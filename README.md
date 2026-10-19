# Classifying FDA Medical Device Adverse Events with LLMs

**ADS-509 Applied Large Language Models for Data Science, University of San Diego (Group 2)**

## Team

| Member | Contributions |
|---|---|
| Luisa Gonzalez | Pulled CGM adverse event reports from the FDA MAUDE database with Python and the openFDA API, organized them into a pandas DataFrame, and explored the dataset and event types to prepare the data for classification (notebook 01). |
| Stephanie Smith | Ran additional EDA (bigram analysis), built the group-aware train/test split, generated Sentence-BERT embeddings, trained standard and class-balanced logistic regression models, and evaluated the models and their misclassified reports (notebook 02). |
| Kamran Shirazi | Built the LLM classification pipeline with the Anthropic API (`llm_extraction.py`), developed the v1 and v2 prompts through error analysis, wrote the data cleaning rules, and built the four-model comparison and error analysis (notebook 03). |

This project tests whether a large language model, called through an API with no training data, can classify FDA adverse event reports as accurately as a traditional embedding-based classifier trained on labeled reports. We use narratives from the FDA MAUDE database for continuous glucose monitors (CGMs) and predict FDA's event type: **Malfunction**, **Injury**, or **Death**. The project follows **Option 2 (Generative AI Application)**.

## Key results

All four models were scored on the same 193 held-out test reports.

| Model | Training data | Accuracy | Macro F1 | Errors | Death recall |
|---|---|---|---|---|---|
| LLM v1 (plain-language label definitions) | None | 0.943 | 0.940 | 11 | 15/17 |
| **LLM v2 (MAUDE-aligned label definitions)** | **None** | **0.984** | **0.981** | **3** | **16/17** |
| Sentence-BERT + Logistic Regression | 758 reports | 0.974 | 0.954 | 5 | 14/17 |
| Sentence-BERT + Balanced Logistic Regression | 758 reports | 0.974 | 0.964 | 5 | 15/17 |

- The zero-shot LLM (v2) was the most accurate model. Most of its gain over v1 came from rewriting the label definitions after reviewing v1's errors, not from changing the model.
- The ranking holds when repeated narratives are collapsed to one row each (128 unique narratives: v2 0.977, SBERT 0.961) and when the 3 test reports that informed the v2 prompt are removed (v2 still 0.984).
- Every LLM error predicted a less severe label than FDA. The balanced SBERT model was the only one to over-call (2 Malfunctions predicted as Injury).
- Besides the label, each LLM call extracts a short failure mode and a patient harm flag, which SBERT cannot provide.

Full results are in `03_LLM_Results_and_Comparison.ipynb` and `results/model_comparison.csv`.

## Pipeline

```
openFDA API ──> 01 Data collection ──> 1,000 reports
                        │
                        ▼
                 01 Data cleaning ──> 951 reports
                        │
                        ▼
          02 EDA + group-aware train/test split (758 / 193)
                │                               │
                ▼                               ▼
   02 Sentence-BERT + LogReg        llm_extraction.py (Claude API)
                │                               │
                └──────────────┬────────────────┘
                               ▼
                 03 Model comparison and error analysis
```

## Data

- **Source:** [openFDA Device Adverse Event API](https://open.fda.gov/apis/device/event/), FDA product code `QBJ` (continuous glucose monitoring systems).
- **Sample:** 1,000 reports received between April 2020 and November 2025: 500 Malfunction, 400 Injury, and 100 Death. MAUDE is extremely imbalanced (over 98% of CGM reports are Malfunctions), so each class was requested separately to give the models enough Injury and Death examples. Most reports are for Dexcom G6 devices.
- **Cleaning (notebook 01):** reports were removed when the narrative cannot support the FDA label:
  1. Follow-up reports where the narrative is only an update note ("Subsequent to the initial MDR...")
  2. Injury or Death reports whose narrative says no injury or medical intervention was reported
  3. Death reports whose narrative says there was no report of death
  4. Injury or Death reports whose narrative says there was no patient injury

  This left 951 reports (498 Malfunction, 366 Injury, 87 Death).
- **Train/test split (notebook 02):** about 80/20 using `GroupShuffleSplit` grouped on normalized narrative text, so identical narratives never appear in both train and test. This produced 758 training and 193 test reports.

## Repository contents

### Code

| File | Description |
|---|---|
| `01_MAUDE_Data_Collection.ipynb` | Pulls CGM reports from the openFDA API, explores class balance, manufacturers, brands, and narrative length, builds the 1,000-report sample, and applies the cleaning rules above. |
| `02_MAUDE_Text_Classification.ipynb` | EDA on narrative length and top bigrams by event type, the group-aware train/test split, and two Sentence-BERT (`all-MiniLM-L6-v2`) + logistic regression classifiers (standard and class-balanced). Exports the split and the predictions. |
| `llm_extraction.py` | Sends each narrative to Claude (Anthropic API) and returns a structured prediction: event type, failure mode, and patient harm. Uses a JSON schema so the model can only return valid labels, caches every response so reruns never repeat API calls, and scores the predictions against FDA labels. Contains two prompt versions (v1 and v2). |
| `03_LLM_Results_and_Comparison.ipynb` | Compares all four models on the same test set, runs the two robustness checks, plots confusion matrices and per-class recall, reviews errors, and summarizes the failure modes extracted by the LLM. |

### `data/`

| File | Description |
|---|---|
| `fda_cgm_reports_1000.csv` | Raw 1,000-report sample from the openFDA API (output of notebook 01). |
| `fda_cgm_reports_clean.csv` | After the first three cleaning rules (953 reports). |
| `fda_cgm_reports_clean_2.csv` | **Final dataset** after the fourth cleaning rule (951 reports). Used by notebooks 02 and 03. |
| `train_group_split.csv` | **Final training set** (758 reports): `report_id`, `narrative`, `event_type`. |
| `test_group_split.csv` | **Final test set** (193 reports). All four models are scored on these reports. |
| `fda_cgm_train.csv`, `fda_cgm_test.csv` | Earlier stratified split (800 / 200) from before cleaning and the group-aware split. Kept for reference; not used in final results. |

### `results/`

Final results:

| File | Description |
|---|---|
| `llm_v1_testgroup.csv`, `llm_v2_testgroup.csv` | LLM predictions on the final 193-report test set: `report_id`, `event_type`, `failure_mode`, `patient_harm`. |
| `cache_v1_testgroup.jsonl`, `cache_v2_testgroup.jsonl` | Raw cached API responses for the final test run. |
| `bert_logreg_predictions.csv` | Sentence-BERT predictions on the final test set (standard and balanced). |
| `bert_logreg_results.csv` | Sentence-BERT accuracy, macro F1, and per-class precision, recall, and F1. |
| `bert_logreg_errors.csv` | Test reports misclassified by either Sentence-BERT model. |
| `model_comparison.csv` | Metrics for all four models on all 193 reports, the 128 unique narratives, and the 190 reports excluding prompt-development overlap. |

Earlier runs, kept to document how the v2 prompt was developed:

| File | Description |
|---|---|
| `llm_v1_test.csv`, `llm_v2_test.csv` | LLM predictions on the earlier stratified test split. |
| `llm_errors_v1.csv` | v1's 15 errors on the earlier split. These were reviewed to write the v2 label definitions. Notebook 03 uses this file to check whether any of these reports ended up in the final test set. |
| `llm_v1_train100.csv`, `llm_v2_train100.csv` | v1 and v2 on a stratified sample of 100 training reports, used to check v2 before scoring it on the test set. |
| `cache_v1_test.jsonl`, `cache_v2_test.jsonl`, `cache_v1_test_full1000.jsonl`, `cache_v1_train100.jsonl`, `cache_v2_train100.jsonl` | Cached API responses for the runs above. `llm_extraction.py` reuses these by `report_id` so the same report is never sent to the API twice for the same prompt. |

### `figures/`

Created by notebook 03.

| File | Description |
|---|---|
| `model_comparison.png` | Accuracy and macro F1 for all four models, on all 193 reports and the 128 unique narratives. |
| `confusion_matrices.png` | Confusion matrix for each model. |
| `recall_by_class.png` | Recall for Malfunction, Injury, and Death by model. |
| `failure_modes.png` | Device problems extracted by LLM v2, grouped into categories and broken out by FDA label. |

## How to reproduce

**Requirements:** Python 3.10+ and

```
pip install pandas requests scikit-learn sentence-transformers matplotlib seaborn anthropic python-dotenv jupyter
```

**Notebooks 01 and 02** were developed in Google Colab (use the "Open in Colab" badge at the top of each) and read their input files from `data/`.

**LLM classification** runs locally from the repository root. Add an Anthropic API key to a `.env` file (already in `.gitignore`):

```
ANTHROPIC_API_KEY=your-key-here
```

Then run each prompt version on the final test set:

```
python llm_extraction.py --input data/fda_cgm_reports_clean_2.csv --prompt v1 --test-file data/test_group_split.csv
python llm_extraction.py --input data/fda_cgm_reports_clean_2.csv --prompt v2 --test-file data/test_group_split.csv
```

Because every response is cached in `results/`, rerunning these commands with the repository's cache files makes no new API calls. Other options: `--train-sample N` runs a stratified sample of N training reports, and `--limit N` processes only the first N reports (useful for a quick test).

**Notebook 03** runs locally from the repository root after the LLM results exist:

```
jupyter nbconvert --to notebook --execute --inplace 03_LLM_Results_and_Comparison.ipynb
```

## Use of generative AI

**As part of the analysis.** The LLM classifier calls Claude Haiku through the Anthropic API. The model sees only the narrative text, never the FDA label, and returns a JSON object constrained to the three valid labels. The prompt was developed in two rounds: v1 used plain-language definitions, then v1's errors were reviewed and v2 was rewritten to match how MAUDE event types are actually assigned (for example, a treated low blood sugar counts as Injury, and any death tied to the event counts as Death). v2 was checked on training reports before being scored on the test set.

**As a coding assistant.**
- ChatGPT was used to debug code in notebook 01 (openFDA data collection) and notebook 02 (the `GroupShuffleSplit` train/test split).
- Claude (Anthropic) drafted the initial scaffold of `llm_extraction.py` and notebook 03.

All AI-assisted code was reviewed, modified, run, and checked by the team.

## Limitations and ethical considerations

- **Small test set.** 193 reports with 17 deaths, so a single error changes Death recall by about 6 points.
- **Label noise.** FDA event types reflect manufacturer reporting practice, and some labels don't match their narratives. We removed the clearest contradictions, and a few remaining "errors" appear to be label problems.
- **Prompt tuned on this data.** v2's rules were written after reviewing v1's errors on an earlier split. Notebook 03 checks this and v2's lead holds after removing the overlapping reports.
- **Run-to-run variability is not measured.** Each report was classified once and cached.
- **Severity under-calling.** All LLM errors predicted a less severe label than FDA. In a real surveillance setting, a missed Injury or Death is more costly than a false alarm, so this system should support human reviewers, not replace them.
- **Data privacy.** MAUDE narratives are public and FDA redacts personal information before release. No private data was sent to the API.
