# =========================
# Hackathon script with 3-query loop support
# =========================

import os
import random
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import subprocess

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
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from Bio.Align import substitution_matrices
from xgboost import XGBRegressor
import optuna


# =========================
# USER SETTINGS
# =========================

DATA_DIR = Path.cwd() / "Hackathon_data"

# Put returned ActiveLearning CSV files here as you get them back.
# Round 0 / before any query:
QUERY_RESULT_FILES = [
    "query_round_1_results.csv",
    "query_round_2_results.csv",
     "query_round_3_results.csv",
]

# Tunables
SEQ_LENGTH = 656
SEED = 0
VAL_RATIO = 0.2
CHECKPOINT_THRESHOLD = 0.02
N_TRIALS = 25

QUERY_BUDGET = 100
EXPLOIT_COUNT = 70          # top predicted points
EXPLORE_POOL_SIZE = 500     # sample remaining 30 from next-best region


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


df_train = pd.read_csv(DATA_DIR / "train.csv")
df_train["sequence"] = df_train["mutant"].apply(lambda x: get_mutated_sequence(x, sequence_wt))

# Keep this fixed for top10 exclusion
original_train_mutants = set(df_train["mutant"].astype(str))

df_test = pd.read_csv(DATA_DIR / "test.csv")
df_test["sequence"] = df_test["mutant"].apply(lambda x: get_mutated_sequence(x, sequence_wt))

print("Initial train shape:", df_train.shape)
print("Test shape:", df_test.shape)


# =========================
# INTEGRATE QUERY RESULTS
# =========================

def load_and_normalize_query_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # expected columns from returned oracle data
    required = {"mutant", "DMS_score"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{path.name} must contain columns {sorted(required)}; "
            f"found {df.columns.tolist()}"
        )

    df = df[["mutant", "DMS_score"]].copy()
    df["mutant"] = df["mutant"].astype(str)
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

    wildtype_props = np.array(AA_PROPS[wildtype_aa], dtype=np.float32)
    mutant_props = np.array(AA_PROPS[mutant_aa], dtype=np.float32)
    prop_delta = mutant_props - wildtype_props

    wildtype_onehot = np.zeros(20, dtype=np.float32)
    mutant_onehot = np.zeros(20, dtype=np.float32)

    wildtype_onehot[ALPHABET.index(wildtype_aa)] = 1.0
    mutant_onehot[ALPHABET.index(mutant_aa)] = 1.0

    normalized_position = np.array([position / seq_length], dtype=np.float32)
    blosum_score = np.array([blosum62[wildtype_aa][mutant_aa]], dtype=np.float32)

    embedding = np.concatenate([
        normalized_position,
        wildtype_onehot,
        mutant_onehot,
        prop_delta,
        blosum_score,
    ]).astype(np.float32)

    return embedding


X_all = np.stack(df_train["mutant"].apply(encode_mutant).values)
y_all = df_train["DMS_score"].values.astype(np.float32)

X_test = np.stack(df_test["mutant"].apply(encode_mutant).values)

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

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "early_stopping_rounds": 30,
        "random_state": SEED,
        "objective": "reg:squarederror",
        "n_jobs": -1,
    }

    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_pred = model.predict(X_val)
    score = spearmanr(y_val, val_pred)[0]

    if np.isnan(score):
        score = -1.0

    return score


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
    n_jobs=-1,
)

best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

val_pred = best_model.predict(X_val)
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
df_test["DMS_score_predicted"] = best_model.predict(X_test)

submission_df = df_test[["mutant", "DMS_score_predicted"]].copy()
submission_df.to_csv("predictions.csv", index=False)
submission_df.to_csv("test_predictions.csv", index=False)

print("Saved predictions.csv and test_predictions.csv")
print(submission_df.head())


# =========================
# TOP 10 FOR FINAL SUBMISSION
# exclude ONLY original train.csv mutants
# =========================

top10_df = (
    df_test.loc[
        ~df_test["mutant"].isin(original_train_mutants),
        ["mutant", "DMS_score_predicted"]
    ]
    .drop_duplicates(subset=["mutant"])
    .sort_values("DMS_score_predicted", ascending=False)
    .head(10)
    .reset_index(drop=True)
)

with open("top10.txt", "w") as f:
    for mutant in top10_df["mutant"]:
        f.write(f"{mutant}\n")

print("Saved top10.txt")
print(top10_df)


# =========================
# BUILD NEXT QUERY FILE
# exclude everything already labeled so far
# =========================

queries_completed = len(QUERY_RESULT_FILES)

if queries_completed < 3:
    already_labeled_mutants = set(df_train["mutant"].astype(str))

    query_candidates = (
        df_test.loc[
            ~df_test["mutant"].isin(already_labeled_mutants),
            ["mutant", "DMS_score_predicted"]
        ]
        .drop_duplicates(subset=["mutant"])
        .sort_values("DMS_score_predicted", ascending=False)
        .reset_index(drop=True)
    )

    if len(query_candidates) < QUERY_BUDGET:
        raise ValueError(
            f"Only {len(query_candidates)} unlabeled candidates left, "
            f"but QUERY_BUDGET={QUERY_BUDGET}"
        )

    # exploit
    top_part = query_candidates.head(EXPLOIT_COUNT).copy()

    # explore from the next-best region
    explore_start = EXPLOIT_COUNT
    explore_end = min(len(query_candidates), EXPLORE_POOL_SIZE)
    explore_pool = query_candidates.iloc[explore_start:explore_end].copy()

    n_explore = QUERY_BUDGET - len(top_part)

    if len(explore_pool) >= n_explore:
        explore_part = explore_pool.sample(n=n_explore, random_state=SEED, replace=False)
    else:
        explore_part = explore_pool.copy()

    query_df = pd.concat([top_part, explore_part], ignore_index=True)
    query_df = query_df.drop_duplicates(subset=["mutant"])

    # top up to 100 if dedupe reduced count
    if len(query_df) < QUERY_BUDGET:
        already_selected = set(query_df["mutant"])
        filler = query_candidates.loc[~query_candidates["mutant"].isin(already_selected)].head(
            QUERY_BUDGET - len(query_df)
        )
        query_df = pd.concat([query_df, filler], ignore_index=True)

    query_df = query_df.head(QUERY_BUDGET).reset_index(drop=True)

    next_round = queries_completed + 1
    query_filename = f"query_round_{next_round}.txt"

    with open(query_filename, "w") as f:
        for mutant in query_df["mutant"]:
            f.write(f"{mutant}\n")

    print(f"Saved {query_filename}")
    print(query_df.head(20))
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
# Leaves test_predictions.csv untouched and writes a separate file.
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