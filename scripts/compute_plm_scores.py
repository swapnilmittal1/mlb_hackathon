from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmForMaskedLM


DATA_DIR = Path.cwd() / "Hackathon_data"
SEQUENCE_FASTA = DATA_DIR / "sequence.fasta"
TEST_CSV = DATA_DIR / "test.csv"
OUTPUT_CSV = DATA_DIR / "plm_scores.csv"

MODEL_NAME = "facebook/esm1v_t33_650M_UR90S_1"

# Use GPU if available.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Optional batching over mutation positions
BATCH_SIZE_POSITIONS = 128


def read_fasta_sequence(path: Path) -> str:
    lines = path.read_text().strip().splitlines()
    sequence_lines = [line.strip() for line in lines if not line.startswith(">")]
    sequence = "".join(sequence_lines)
    if not sequence:
        raise ValueError(f"No sequence found in {path}")
    return sequence


def parse_mutant(mutant: str) -> Tuple[str, int, str]:
    wildtype_aa = mutant[0]
    position = int(mutant[1:-1])
    mutant_aa = mutant[-1]
    return wildtype_aa, position, mutant_aa


def log_softmax_np(x: np.ndarray) -> np.ndarray:
    x_max = np.max(x, axis=-1, keepdims=True)
    shifted = x - x_max
    logsumexp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    return shifted - logsumexp


def build_position_groups(df_test: pd.DataFrame) -> Dict[int, List[Tuple[str, str, str]]]:
    groups: Dict[int, List[Tuple[str, str, str]]] = {}
    for mutant in df_test["mutant"].astype(str):
        wt, pos, mt = parse_mutant(mutant)
        groups.setdefault(pos, []).append((mutant, wt, mt))
    return groups


def score_positions_with_esm1v(
    sequence_wt: str,
    position_groups: Dict[int, List[Tuple[str, str, str]]],
) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    aa_token_ids = {aa: tokenizer.convert_tokens_to_ids(aa) for aa in amino_acids}

    # Tokenization layout for ESM:
    # [CLS] A A A ... [EOS]
    # so sequence position i maps to token index i + 1
    positions = sorted(position_groups.keys())
    results: List[Dict[str, float | str | int]] = []

    with torch.no_grad():
        for start in range(0, len(positions), BATCH_SIZE_POSITIONS):
            batch_positions = positions[start:start + BATCH_SIZE_POSITIONS]

            masked_sequences = []
            for pos in batch_positions:
                if pos < 0 or pos >= len(sequence_wt):
                    raise ValueError(f"Mutation position {pos} out of range for sequence length {len(sequence_wt)}")
                seq_list = list(sequence_wt)
                seq_list[pos] = tokenizer.mask_token
                masked_sequences.append("".join(seq_list))

            encoded = tokenizer(
                masked_sequences,
                return_tensors="pt",
                padding=True,
                truncation=False,
            ).to(DEVICE)

            outputs = model(**encoded)
            logits = outputs.logits.detach().cpu().numpy()

            for row_idx, pos in enumerate(batch_positions):
                token_pos = pos + 1
                log_probs = log_softmax_np(logits[row_idx, token_pos, :])

                for mutant, wt, mt in position_groups[pos]:
                    if wt != sequence_wt[pos]:
                        raise ValueError(
                            f"Wild-type mismatch for {mutant}: mutant says {wt}, sequence has {sequence_wt[pos]}"
                        )
                    if wt not in aa_token_ids or mt not in aa_token_ids:
                        raise ValueError(f"Unsupported amino acid in mutant {mutant}")

                    wt_token_id = aa_token_ids[wt]
                    mt_token_id = aa_token_ids[mt]

                    logp_wt = float(log_probs[wt_token_id])
                    logp_mt = float(log_probs[mt_token_id])

                    # Main evolutionary plausibility score:
                    # log p(mutant aa at position | context) - log p(wildtype aa at position | context)
                    plm_score = logp_mt - logp_wt

                    results.append(
                        {
                            "mutant": mutant,
                            "position": pos,
                            "wildtype_aa": wt,
                            "mutant_aa": mt,
                            "logp_wildtype": logp_wt,
                            "logp_mutant": logp_mt,
                            "plm_score": plm_score,
                        }
                    )

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(["position", "mutant"]).reset_index(drop=True)
    return result_df


def main() -> None:
    if not SEQUENCE_FASTA.exists():
        raise FileNotFoundError(f"Missing fasta file: {SEQUENCE_FASTA}")
    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Missing test csv: {TEST_CSV}")

    sequence_wt = read_fasta_sequence(SEQUENCE_FASTA)
    df_test = pd.read_csv(TEST_CSV)

    if "mutant" not in df_test.columns:
        raise ValueError("test.csv must contain a 'mutant' column")

    df_test["mutant"] = df_test["mutant"].astype(str)
    position_groups = build_position_groups(df_test)

    result_df = score_positions_with_esm1v(sequence_wt, position_groups)

    # keep the exact two-column file your main pipeline expects,
    # plus write a richer debug file
    result_df[["mutant", "plm_score"]].to_csv(OUTPUT_CSV, index=False)
    result_df.to_csv(DATA_DIR / "plm_scores_full_debug.csv", index=False)

    print(f"Saved {OUTPUT_CSV}")
    print(f"Saved {DATA_DIR / 'plm_scores_full_debug.csv'}")
    print(result_df.head())


if __name__ == "__main__":
    main()