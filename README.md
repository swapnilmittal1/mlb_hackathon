# MLB S26 Hackathon — Protein Fitness Prediction with Active Learning

This project builds an ML pipeline to predict single-mutation protein fitness (⁠ DMS_score ⁠) and recommend top mutations under a limited active-learning query budget.

## Repository Contents

### Core scripts
•⁠  ⁠⁠ hachathon_final_with_plm.py ⁠  
  Final end-to-end pipeline (query integration + model training + ensemble + submission outputs).

•⁠  ⁠⁠ mlb_hackathon_script.py ⁠  
  Main script variant for iterative experimentation and active-learning workflow.

•⁠  ⁠⁠ compute_plm_scores.py ⁠  
  Computes ESM-1v mutation plausibility scores (⁠ plm_score ⁠) for all test mutants.

•⁠  ⁠⁠ hackathonNBFinal-2.ipynb ⁠  
  Notebook version of the workflow.

### Data folder
•⁠  ⁠⁠ Hackathon_data/sequence.fasta ⁠ — wild-type sequence  
•⁠  ⁠⁠ Hackathon_data/train.csv ⁠ — initial labeled data  
•⁠  ⁠⁠ Hackathon_data/test.csv ⁠ — unlabeled test mutants  
•⁠  ⁠⁠ Hackathon_data/query_round_1_results.csv ⁠  
•⁠  ⁠⁠ Hackathon_data/query_round_2_results.csv ⁠  
•⁠  ⁠⁠ Hackathon_data/query_round_3_results.csv ⁠  
•⁠  ⁠⁠ Hackathon_data/plm_scores.csv ⁠ — PLM scores used in final model  
•⁠  ⁠⁠ Hackathon_data/plm_scores_full_debug.csv ⁠ — detailed PLM scoring diagnostics

### Generated outputs
•⁠  ⁠⁠ predictions.csv ⁠  
•⁠  ⁠⁠ test_predictions.csv ⁠  
•⁠  ⁠⁠ test_predictions_submission.csv ⁠  
•⁠  ⁠⁠ top10.txt ⁠  
•⁠  ⁠⁠ query_round_*.txt ⁠

---

## Method Overview

Our final approach combines three components:

1.⁠ ⁠*Feature-engineered XGBoost baseline*
   - mutation position features
   - amino-acid one-hot encoding (WT + mutant)
   - physicochemical descriptors and deltas
   - BLOSUM62 substitution signal
   - local context features near mutation site

2.⁠ ⁠*Active learning integration*
   - merge returned query labels into training set each round
   - deduplicate by ⁠ mutant ⁠ to keep the latest label
   - use model score + uncertainty + diversity criteria for subsequent query generation

3.⁠ ⁠*Protein language model prior (ESM-1v) + ensembling*
   - compute per-mutant ⁠ plm_score ⁠ from log-probability difference
   - include PLM signal in ranking/selection
   - train multiple seeded final models and aggregate predictions (mean/std)

Primary model metric: *Spearman correlation* on validation split.

---

## Setup

Recommended: Python 3.10+ (3.11 also works).

Install dependencies:

```bash
pip install numpy pandas scipy scikit-learn biopython xgboost optuna torch transformers