from __future__ import annotations

import importlib.util
import random
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "Bio": "biopython",
    "xgboost": "xgboost",
    "optuna": "optuna",
}


def ensure_dependencies() -> None:
    missing_packages = [
        package_name
        for module_name, package_name in REQUIRED_PACKAGES.items()
        if importlib.util.find_spec(module_name) is None
    ]
    if missing_packages:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", *missing_packages]
        )


ensure_dependencies()

import numpy as np
import pandas as pd
import optuna
from Bio.Align import substitution_matrices
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


# =========================
# USER SETTINGS
# =========================

DATA_DIR = Path.cwd() / "Hackathon_data"

# Set this based on how many returned ActiveLearning result files you currently have.
# Examples:
# []  -> before round 1
# ["query_round_1_results.csv"] -> before round 2
# ["query_round_1_results.csv", "query_round_2_results.csv"] -> before round 3
# ["query_round_1_results.csv", "query_round_2_results.csv", "query_round_3_results.csv"] -> final
QUERY_RESULT_FILES = [
     "query_round_1_results.csv",
    "query_round_2_results.csv",
     "query_round_3_results.csv",
]

SEQ_LENGTH = 656
SEED = 0
VAL_RATIO = 0.2
CHECKPOINT_THRESHOLD = 0.02
N_TRIALS = 25
FINAL_MODEL_SEEDS = [SEED + i for i in range(5)]
QUERY_BUDGET = 100

QUERY_ROUND_PLANS = [
    {"likely_good": 50, "uncertain": 30, "diverse": 20, "max_per_position": 3},
    {"likely_good": 60, "uncertain": 25, "diverse": 15, "max_per_position": 4},
    {"likely_good": 70, "uncertain": 20, "diverse": 10, "max_per_position": 5},
]

PLM_SCORE_FILE_CANDIDATES = [
    DATA_DIR / "plm_scores.csv",
    DATA_DIR / "esm_scores.csv",
    Path.cwd() / "plm_scores.csv",
    Path.cwd() / "esm_scores.csv",
]

QUERY_SCORE_WEIGHTS = {"pred_mean": 0.55, "pred_std": 0.25, "plm_score": 0.20}
TOP10_SCORE_WEIGHTS = {"pred_mean": 0.75, "pred_std": -0.15, "plm_score": 0.10}


# =========================
# REPRODUCIBILITY
# =========================

random.seed(SEED)
np.random.seed(SEED)


# =========================
# LOAD BASE DATA
# =========================

with open(DATA_DIR / "sequence.fasta", "r") as f:
    data = f.readlines()

sequence_wt = data[1].strip()
print("WT prefix:", sequence_wt[:20], "...")
print("WT length:", len(sequence_wt))


def get_mutated_sequence(mut: str, sequence_wt: str) -> str:
    wt, pos, mt = mut[0], int(mut[1:-1]), mut[-1]
    sequence = deepcopy(sequence_wt)
    return sequence[:pos] + mt + sequence[pos + 1:]


def extract_mutation_position(mutant: str) -> int:
    return int(mutant[1:-1])


df_train = pd.read_csv(DATA_DIR / "train.csv")
df_train["mutant"] = df_train["mutant"].astype(str)
df_train["sequence"] = df_train["mutant"].apply(lambda x: get_mutated_sequence(x, sequence_wt))

original_train_mutants = set(df_train["mutant"])

df_test = pd.read_csv(DATA_DIR / "test.csv")
df_test["mutant"] = df_test["mutant"].astype(str)
df_test["sequence"] = df_test["mutant"].apply(lambda x: get_mutated_sequence(x, sequence_wt))

print("Initial train shape:", df_train.shape)
print("Test shape:", df_test.shape)


# =========================
# INTEGRATE QUERY RESULTS
# =========================

def load_and_normalize_query_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"mutant", "DMS_score"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{path.name} must contain columns {sorted(required)}; "
            f"found {df.columns.tolist()}"
        )

    df = df[["mutant", "DMS_score"]].copy()
    df["mutant"] = df["mutant"].astype(str)
    df["DMS_score"] = pd.to_numeric(df["DMS_score"], errors="coerce")
    df = df.dropna(subset=["DMS_score"])
    df["sequence"] = df["mutant"].apply(lambda x: get_mutated_sequence(x, sequence_wt))
    df = df.drop_duplicates(subset=["mutant"], keep="last").reset_index(drop=True)
    return df


for fname in QUERY_RESULT_FILES:
    query_path = DATA_DIR / fname
    if not query_path.exists():
        raise FileNotFoundError(f"Missing query result file: {query_path}")

    df_query = load_and_normalize_query_results(query_path)
    df_train = pd.concat([df_train, df_query], ignore_index=True)
    df_train = df_train.drop_duplicates(subset=["mutant"], keep="last").reset_index(drop=True)

print("Train shape after query integration:", df_train.shape)
print("Queries already integrated:", len(QUERY_RESULT_FILES))


# =========================
# OPTIONAL PLM SCORES
# =========================

def resolve_plm_score_file() -> Path | None:
    for candidate in PLM_SCORE_FILE_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def load_plm_scores() -> pd.DataFrame:
    plm_path = resolve_plm_score_file()
    if plm_path is None:
        print("No PLM score file found. Using 0.0 for plm_score.")
        return pd.DataFrame(columns=["mutant", "plm_score"])

    plm_df = pd.read_csv(plm_path)
    required = {"mutant", "plm_score"}
    if not required.issubset(plm_df.columns):
        raise ValueError(
            f"{plm_path.name} must contain columns {sorted(required)}; "
            f"found {plm_df.columns.tolist()}"
        )

    plm_df = plm_df[["mutant", "plm_score"]].copy()
    plm_df["mutant"] = plm_df["mutant"].astype(str)
    plm_df["plm_score"] = pd.to_numeric(plm_df["plm_score"], errors="coerce")
    plm_df = plm_df.dropna(subset=["plm_score"])
    plm_df = plm_df.drop_duplicates(subset=["mutant"], keep="last").reset_index(drop=True)

    print(f"Loaded PLM scores from {plm_path.name} for {len(plm_df)} mutants.")
    return plm_df


plm_scores_df = load_plm_scores()

df_train = df_train.merge(plm_scores_df, on="mutant", how="left")
df_test = df_test.merge(plm_scores_df, on="mutant", how="left")

df_train["plm_score"] = df_train["plm_score"].fillna(0.0).astype(np.float32)
df_test["plm_score"] = df_test["plm_score"].fillna(0.0).astype(np.float32)
df_train["position"] = df_train["mutant"].apply(extract_mutation_position)
df_test["position"] = df_test["mutant"].apply(extract_mutation_position)


# =========================
# FEATURE ENCODING
# =========================

blosum62 = substitution_matrices.load("BLOSUM62")

AA_PROPS = {
    "A": [1.8,   0,   89,  67],
    "C": [2.5,   0,  121,  86],
    "D": [-3.5, -1,  133,  91],
    "E": [-3.5, -1,  147, 109],
    "F": [2.8,   0,  165, 135],
    "G": [-0.4,  0,   75,  48],
    "H": [-3.2,  0.5, 155, 118],
    "I": [4.5,   0,  131, 124],
    "K": [-3.9,  1,  146, 135],
    "L": [3.8,   0,  131, 124],
    "M": [1.9,   0,  149, 124],
    "N": [-3.5,  0,  132,  96],
    "P": [-1.6,  0,  115,  90],
    "Q": [-3.5,  0,  146, 114],
    "R": [-4.5,  1,  174, 148],
    "S": [-0.8,  0,  105,  73],
    "T": [-0.7,  0,  119,  93],
    "V": [4.2,   0,  117, 105],
    "W": [-0.9,  0,  204, 163],
    "Y": [-1.3,  0,  181, 141],
}

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def encode_mutant(mutant: str, seq_length: int = SEQ_LENGTH) -> np.ndarray:
    wildtype_aa = mutant[0]
    position = int(mutant[1:-1])
    mutant_aa = mutant[-1]
    position_idx = position  # mutation positions are 0-indexed

    wildtype_props = np.array(AA_PROPS[wildtype_aa], dtype=np.float32)
    mutant_props = np.array(AA_PROPS[mutant_aa], dtype=np.float32)
    prop_delta = mutant_props - wildtype_props
    prop_abs_delta = np.abs(prop_delta)

    wildtype_onehot = np.zeros(20, dtype=np.float32)
    mutant_onehot = np.zeros(20, dtype=np.float32)
    wildtype_onehot[ALPHABET.index(wildtype_aa)] = 1.0
    mutant_onehot[ALPHABET.index(mutant_aa)] = 1.0

    context_props = []
    for offset in (-2, -1, 1, 2):
        neighbor_idx = position_idx + offset
        if 0 <= neighbor_idx < len(sequence_wt):
            context_props.extend(AA_PROPS[sequence_wt[neighbor_idx]])
        else:
            context_props.extend([0.0] * len(wildtype_props))

    left_span = position_idx / max(seq_length - 1, 1)
    right_span = (seq_length - 1 - position_idx) / max(seq_length - 1, 1)
    center_distance = abs(position_idx - (seq_length - 1) / 2) / max((seq_length - 1) / 2, 1)

    normalized_position = np.array(
        [position / seq_length, left_span, right_span, center_distance],
        dtype=np.float32,
    )
    blosum_score = np.array([blosum62[wildtype_aa][mutant_aa]], dtype=np.float32)

    embedding = np.concatenate([
        normalized_position,
        wildtype_onehot,
        mutant_onehot,
        wildtype_props,
        mutant_props,
        prop_delta,
        prop_abs_delta,
        np.array(context_props, dtype=np.float32),
        blosum_score,
    ]).astype(np.float32)

    return embedding


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.stack(df["mutant"].apply(encode_mutant).values)


def zscore_series(values: pd.Series) -> pd.Series:
    values = values.astype(np.float32)
    std = values.std(ddof=0)
    if np.isnan(std) or std < 1e-8:
        return pd.Series(np.zeros(len(values), dtype=np.float32), index=values.index)
    return (values - values.mean()) / std


def add_combined_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pred_mean_z"] = zscore_series(df["pred_mean"])
    df["pred_std_z"] = zscore_series(df["pred_std"])
    df["plm_score_z"] = zscore_series(df["plm_score"])

    df["query_score"] = (
        QUERY_SCORE_WEIGHTS["pred_mean"] * df["pred_mean_z"]
        + QUERY_SCORE_WEIGHTS["pred_std"] * df["pred_std_z"]
        + QUERY_SCORE_WEIGHTS["plm_score"] * df["plm_score_z"]
    )

    df["top10_score"] = (
        TOP10_SCORE_WEIGHTS["pred_mean"] * df["pred_mean_z"]
        + TOP10_SCORE_WEIGHTS["pred_std"] * df["pred_std_z"]
        + TOP10_SCORE_WEIGHTS["plm_score"] * df["plm_score_z"]
    )
    return df


def greedy_select_with_position_cap(
    candidates: pd.DataFrame,
    n_to_select: int,
    selected_mutants: set[str],
    position_counts: dict[int, int],
    max_per_position: int | None,
) -> pd.DataFrame:
    selected_indices = []

    for idx, row in candidates.iterrows():
        mutant = row["mutant"]
        position = int(row["position"])

        if mutant in selected_mutants:
            continue
        if max_per_position is not None and position_counts.get(position, 0) >= max_per_position:
            continue

        selected_indices.append(idx)
        selected_mutants.add(mutant)
        position_counts[position] = position_counts.get(position, 0) + 1

        if len(selected_indices) >= n_to_select:
            break

    if not selected_indices:
        return candidates.iloc[0:0].copy()

    return candidates.loc[selected_indices].copy()


X_all = build_feature_matrix(df_train)
y_all = df_train["DMS_score"].values.astype(np.float32)
X_test = build_feature_matrix(df_test)

train_idx, val_idx = train_test_split(
    np.arange(len(df_train)),
    test_size=VAL_RATIO,
    random_state=SEED,
)

X_train, y_train = X_all[train_idx], y_all[train_idx]
X_val, y_val = X_all[val_idx], y_all[val_idx]

print(f"Features: {X_train.shape[1]}, Train: {len(X_train)}, Val: {len(X_val)}")


# =========================
# TRAIN MODEL
# =========================

def objective(trial: optuna.Trial) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "early_stopping_rounds": 30,
        "random_state": SEED,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "n_jobs": -1,
    }

    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_pred = model.predict(X_val)
    score = spearmanr(y_val, val_pred)[0]

    if np.isnan(score):
        score = -1.0

    return float(score)


sampler = optuna.samplers.TPESampler(seed=SEED)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print(f"Best Spearman: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

best_model = XGBRegressor(
    **study.best_params,
    early_stopping_rounds=30,
    random_state=SEED,
    objective="reg:squarederror",
    tree_method="hist",
    n_jobs=-1,
)
best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

best_iteration = getattr(best_model, "best_iteration", None)
final_n_estimators = (best_iteration + 1) if best_iteration is not None else study.best_params["n_estimators"]
final_model_params = {**study.best_params, "n_estimators": final_n_estimators}

ensemble_val_predictions = []
for ensemble_seed in FINAL_MODEL_SEEDS:
    ensemble_model = XGBRegressor(
        **final_model_params,
        random_state=ensemble_seed,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
    )
    ensemble_model.fit(X_train, y_train, verbose=False)
    ensemble_val_predictions.append(ensemble_model.predict(X_val))

val_pred = np.mean(np.vstack(ensemble_val_predictions), axis=0)
val_sp = spearmanr(y_val, val_pred)[0]

print(f"Val Spearman with best params: {val_sp:.4f}")
if val_sp >= CHECKPOINT_THRESHOLD:
    print(f"Checkpoint threshold passed on validation split (>= {CHECKPOINT_THRESHOLD:.2f}).")
else:
    print(f"Checkpoint threshold NOT passed on validation split (< {CHECKPOINT_THRESHOLD:.2f}).")


# =========================
# TEST PREDICTIONS
# =========================

df_test = df_test.copy()
final_test_predictions = []

for ensemble_seed in FINAL_MODEL_SEEDS:
    final_model = XGBRegressor(
        **final_model_params,
        random_state=ensemble_seed,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
    )
    final_model.fit(X_all, y_all, verbose=False)
    final_test_predictions.append(final_model.predict(X_test))

test_prediction_matrix = np.vstack(final_test_predictions)
df_test["pred_mean"] = np.mean(test_prediction_matrix, axis=0)
df_test["pred_std"] = np.std(test_prediction_matrix, axis=0)
df_test["DMS_score_predicted"] = df_test["pred_mean"]

submission_df = df_test[["mutant", "DMS_score_predicted"]].copy()
submission_df.to_csv("predictions.csv", index=False)
submission_df.to_csv("test_predictions.csv", index=False)

print("Saved predictions.csv and test_predictions.csv")
print(submission_df.head())


# =========================
# TOP 10 FOR FINAL SUBMISSION
# =========================

top10_candidates = (
    df_test.loc[
        ~df_test["mutant"].isin(original_train_mutants),
        ["mutant", "position", "pred_mean", "pred_std", "plm_score"]
    ]
    .drop_duplicates(subset=["mutant"])
    .reset_index(drop=True)
)

top10_candidates = add_combined_scores(top10_candidates)
top10_df = (
    top10_candidates
    .sort_values(
        ["pred_mean", "pred_std", "plm_score"],
        ascending=[False, True, False],
    )
    .head(10)
    .reset_index(drop=True)
)

with open("top10.txt", "w") as f:
    for mutant in top10_df["mutant"]:
        f.write(f"{mutant}\n")

print("Saved top10.txt")
print(top10_df[["mutant", "pred_mean", "pred_std", "plm_score", "top10_score"]])


# =========================
# BUILD NEXT QUERY FILE
# =========================

queries_completed = len(QUERY_RESULT_FILES)

if queries_completed < 3:
    already_labeled_mutants = set(df_train["mutant"].astype(str))
    round_plan = QUERY_ROUND_PLANS[queries_completed]

    query_candidates = (
        df_test.loc[
            ~df_test["mutant"].isin(already_labeled_mutants),
            ["mutant", "position", "pred_mean", "pred_std", "plm_score"]
        ]
        .drop_duplicates(subset=["mutant"])
        .reset_index(drop=True)
    )

    query_candidates = add_combined_scores(query_candidates)

    if len(query_candidates) < QUERY_BUDGET:
        raise ValueError(
            f"Only {len(query_candidates)} unlabeled candidates left, "
            f"but QUERY_BUDGET={QUERY_BUDGET}"
        )

    selected_mutants: set[str] = set()
    position_counts: dict[int, int] = {}

    likely_good_candidates = query_candidates.sort_values(
        ["pred_mean", "plm_score", "query_score"], ascending=False
    )
    likely_good_df = greedy_select_with_position_cap(
        likely_good_candidates,
        round_plan["likely_good"],
        selected_mutants,
        position_counts,
        round_plan["max_per_position"],
    )

    uncertain_candidates = query_candidates.sort_values(
        ["pred_std", "query_score", "pred_mean"], ascending=False
    )
    uncertain_df = greedy_select_with_position_cap(
        uncertain_candidates,
        round_plan["uncertain"],
        selected_mutants,
        position_counts,
        round_plan["max_per_position"],
    )

    diverse_candidates = (
        query_candidates
        .sort_values(["query_score", "pred_mean"], ascending=False)
        .drop_duplicates(subset=["position"], keep="first")
        .reset_index(drop=True)
    )
    diverse_df = greedy_select_with_position_cap(
        diverse_candidates,
        round_plan["diverse"],
        selected_mutants,
        position_counts,
        round_plan["max_per_position"],
    )

    query_df = pd.concat([likely_good_df, uncertain_df, diverse_df], ignore_index=True)
    query_df = query_df.drop_duplicates(subset=["mutant"])

    relaxed_cap = round_plan["max_per_position"]
    ranked_fillers = query_candidates.sort_values(
        ["query_score", "pred_mean", "plm_score"], ascending=False
    )

    while len(query_df) < QUERY_BUDGET:
        filler_df = greedy_select_with_position_cap(
            ranked_fillers,
            QUERY_BUDGET - len(query_df),
            selected_mutants,
            position_counts,
            relaxed_cap,
        )
        if filler_df.empty:
            relaxed_cap = None if relaxed_cap is None else relaxed_cap + 1
            if relaxed_cap is None:
                break
            continue

        query_df = pd.concat([query_df, filler_df], ignore_index=True)
        query_df = query_df.drop_duplicates(subset=["mutant"])

    query_df = query_df.head(QUERY_BUDGET).reset_index(drop=True)

    next_round = queries_completed + 1
    query_filename = f"query_round_{next_round}.txt"

    with open(query_filename, "w") as f:
        for mutant in query_df["mutant"]:
            f.write(f"{mutant}\n")

    print(f"Saved {query_filename}")
    print(query_df[["mutant", "pred_mean", "pred_std", "plm_score", "query_score"]].head(20))
else:
    print("All 3 query rounds already integrated. No further query file generated.")


# =========================
# SANITY CHECKS
# =========================

assert len(submission_df) == len(pd.read_csv(DATA_DIR / "test.csv")), \
    "predictions file must cover all test mutants"

assert submission_df["mutant"].is_unique, \
    "predictions contain duplicate mutants"

assert len(top10_df) == 10, \
    "top10.txt must contain exactly 10 mutants"

assert top10_df["mutant"].is_unique, \
    "top10 contains duplicate mutants"

assert set(top10_df["mutant"]).issubset(set(df_test["mutant"])), \
    "top10 mutants must come from test.csv"

assert set(top10_df["mutant"]).isdisjoint(original_train_mutants), \
    "top10 cannot include original train.csv mutants"

print("Submission sanity checks passed.")
print("Final files currently written:")
print("- predictions.csv")
print("- test_predictions.csv")
print("- top10.txt")
if queries_completed < 3:
    print(f"- query_round_{queries_completed + 1}.txt")
# =========================
# EXTRA SUBMISSION-FORMAT FILE
# Keep official files untouched; write a separate Kaggle/testing file.
# =========================

submission_format_df = submission_df.rename(
    columns={"DMS_score_predicted": "DMS_score"}
).copy()

if "id" not in submission_format_df.columns:
    submission_format_df = submission_format_df.reset_index().rename(
        columns={"index": "id"}
    )

if "mutant" in submission_format_df.columns:
    submission_format_df = submission_format_df.drop(columns=["mutant"])

submission_format_filename = "test_predictions_submission.csv"
submission_format_df.to_csv(submission_format_filename, index=False)

print(f"- {submission_format_filename}")