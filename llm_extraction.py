"""
llm_extraction.py
=================
ADS-509 Final Project (Group 2) -- LLM API integration module.

Reads MAUDE adverse event narratives from a CSV, sends each one to the
Anthropic API, and extracts a structured prediction:

    - event_type:   one of {Death, Injury, Malfunction}
                    (matches the FDA-assigned `event_type` label so we can
                    evaluate the LLM against ground truth)
    - failure_mode: short free-text phrase describing what went wrong
                    with the device (e.g. "transmitter failed error")
    - patient_harm: "Yes" / "No" -- did the narrative describe actual
                    harm (not just potential harm)?

The model only ever sees the narrative text, never FDA's label. Labels are
used after the run to score the predictions.

Results are cached to a JSONL file as they complete, so the script can be
safely interrupted and re-run without repeating (and re-paying for) API
calls. Each prompt version and data split gets its own cache and output
file, so runs never mix.

Usage:
    pip install anthropic pandas python-dotenv scikit-learn
    # put ANTHROPIC_API_KEY=sk-... in a .env file (and .gitignore it!)

    # Check a prompt on a stratified sample of TRAINING reports
    python llm_extraction.py --input data/fda_cgm_reports_1000.csv \
        --prompt v2 --train-sample 100

    # Score a prompt on the TEST reports from notebook 02's group split
    python llm_extraction.py --input data/fda_cgm_reports_clean_2.csv \
        --prompt v2 --test-file data/test_group_split.csv

    Outputs go to results/llm_<prompt>_<split>.csv, e.g.
    results/llm_v2_testgroup.csv or results/llm_v1_train100.csv.

Prompt versions:
    v1 -- plain-language definitions of Death / Injury / Malfunction.
    v2 -- aligns label definitions with how MAUDE event types are actually
          assigned (rules inferred from v1's errors, not the regulation;
          the treated-lows rule is looser than FDA's 21 CFR 803.3 'serious
          injury' definition and reflects manufacturer reporting practice): treated highs/lows,
          medical intervention, and physical harm from the device count as
          Injury; any death tied to the event (including a fetus or
          newborn) counts as Death. These gaps were identified from v1's
          errors on the test set, so v2 is checked on training reports
          before being scored on the test set once.

Expected input columns (from the team's prepared fda_cgm_reports_1000.csv):
    report_id   -- unique MAUDE report number
    narrative   -- the event description text from the report
    event_type  -- FDA label (used only for splitting and scoring)

AI assistance disclosure (per course policy): the initial scaffold of this
module was drafted with Claude (Anthropic) and then reviewed, modified, and
tested by Kamran Shirazi. See README for details.
"""

import argparse
import json
import os
import time

import anthropic
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "claude-haiku-5-5"   # small/cheap model; narratives are short and the
                             # task is classification, not open-ended writing
MAX_TOKENS = 2000            # room for the model's optional thinking step
                             # plus the short JSON answer
SLEEP_BETWEEN_CALLS = 0.5    # seconds; stay politely under rate limits
MAX_RETRIES = 3

# FDA's label set for event_type. The prepared dataset only contains these
# three classes (no "Other"), so we don't offer a label that can't be right.
VALID_EVENT_TYPES = {"Death", "Injury", "Malfunction"}
LABEL_ORDER = ["Malfunction", "Injury", "Death"]   # for confusion matrices

ID_COL = "report_id"   # unique key column in the input CSV

# JSON schema passed to the API as a structured output format. The API
# constrains the model's reply to match it, so event_type can only ever be
# one of our three labels and the JSON is always well formed.
# (parse_response() below still validates, as a second safety check.)
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "event_type": {"type": "string",
                       "enum": sorted(VALID_EVENT_TYPES)},
        "failure_mode": {"type": "string"},
        "patient_harm": {"type": "string", "enum": ["Yes", "No"]},
    },
    "required": ["event_type", "failure_mode", "patient_harm"],
    "additionalProperties": False,
}

# Shared opening for every prompt version: task and output format.
PROMPT_HEADER = """\
You are a careful medical device adverse event analyst. You will be given
the narrative text of a single FDA MAUDE report about a continuous glucose
monitor (CGM) used by people with diabetes.

Read the narrative and respond with ONLY a JSON object (no markdown fences,
no commentary) with exactly these keys:

{
  "event_type": one of "Death", "Injury", or "Malfunction",
  "failure_mode": a short phrase (under 10 words) naming the device
                  problem described, or "unknown" if none is described,
  "patient_harm": "Yes" if the narrative describes harm that actually
                  occurred, otherwise "No"
}
"""

PROMPTS = {
    # v1: plain-language definitions (used for the first full test run).
    "v1": PROMPT_HEADER + """
Guidelines:
- "Malfunction": the device failed or performed incorrectly but no actual
  patient harm is described.
- "Injury": the narrative describes actual harm to the patient (e.g.,
  hospitalization, severe hypoglycemia, burns, infection).
- "Death": the patient died.
- Base your answer only on what the narrative states. Do not speculate.
""",

    # v2: label definitions aligned with how MAUDE event types are actually
    # assigned (inferred from v1's error analysis; see module docstring).
    "v2": PROMPT_HEADER + """
Guidelines (these reflect how event types are assigned in MAUDE reports):
- "Death": anyone involved in the event died, including a fetus or
  newborn, even if it is unclear whether the device caused the death.
- "Injury": the narrative describes actual harm to a person. This includes:
  * a low or high blood glucose that was treated or required action
    (for example glucose tablets, juice, candy, carbohydrates, glucagon,
    insulin, or a correction bolus), or that caused symptoms such as
    confusion, sweating, or loss of consciousness;
  * any medical intervention: emergency services, an ER visit,
    hospitalization, or treatment by a healthcare professional;
  * physical harm caused by the device itself, such as pain, bleeding, a
    needle or applicator stuck in the skin, skin reactions, or burns.
- "Malfunction": the device failed or performed incorrectly, but no actual
  harm is described. An inaccurate reading on its own, with no treatment,
  symptoms, or intervention, is a Malfunction.
- Base your answer only on what the narrative states. Do not speculate.
  If the narrative states that no injury occurred and describes no harm,
  choose "Malfunction".
- "patient_harm" should be "Yes" whenever event_type is "Injury" or "Death".
""",
}


# ---------------------------------------------------------------------------
# Data selection
# ---------------------------------------------------------------------------

def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test) rows using the exact split from notebook 02.

    Same arguments as 02_MAUDE_Text_Classification.ipynb (test_size=0.20,
    random_state=20, stratified on event_type). train_test_split keeps the
    original DataFrame index, so rows map back to their report_id. Verified
    to match fda_cgm_train.csv and fda_cgm_test.csv row for row.
    """
    model_df = df[["narrative", "event_type"]].dropna()
    X_train, X_test, _, _ = train_test_split(
        model_df["narrative"],
        model_df["event_type"],
        test_size=0.20,
        random_state=20,
        stratify=model_df["event_type"],
    )
    return df.loc[X_train.index], df.loc[X_test.index]


def sample_train_rows(train: pd.DataFrame, n: int) -> pd.DataFrame:
    """Stratified sample of n training reports (keeps the class balance).

    Fixed random_state so v1 and v2 are always compared on the same reports.
    """
    frac = n / len(train)
    return (train.groupby("event_type", group_keys=False)
                 .sample(frac=frac, random_state=42))


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def parse_response(raw_text: str) -> dict:
    """Parse the model's reply into a validated dict.

    Strips markdown code fences if present, then checks that all keys exist
    and event_type is an allowed label. Raises ValueError if the reply is
    unusable (the caller retries).
    """
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    result = json.loads(cleaned)  # raises ValueError on bad JSON

    missing = {"event_type", "failure_mode", "patient_harm"} - result.keys()
    if missing:
        raise ValueError(f"Model reply missing keys: {missing}")
    if result["event_type"] not in VALID_EVENT_TYPES:
        raise ValueError(f"Invalid event_type: {result['event_type']!r}")
    return result


def extract_one(client: anthropic.Anthropic, narrative: str,
                system_prompt: str) -> dict:
    """Send one narrative to the LLM and return the parsed prediction.

    Retries with exponential backoff on transient API errors (rate limits,
    overloaded server, dropped connections) and on malformed replies.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Note: the current anthropic SDK (1.x) no longer accepts a
            # temperature argument. Reproducibility instead comes from the
            # JSONL cache: each report is classified once per prompt version
            # and every later run reuses the saved answer.
            message = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=system_prompt,
                messages=[{"role": "user", "content": narrative}],
                output_config={
                    "format": {"type": "json_schema",
                               "schema": OUTPUT_SCHEMA},
                },
            )
            # The model sometimes "thinks" before answering, so the reply
            # can start with a thinking block. Use the text block.
            text_blocks = [b.text for b in message.content
                           if b.type == "text"]
            if not text_blocks:
                raise ValueError("Reply had no text block "
                                 f"(stop_reason={message.stop_reason})")
            return parse_response(text_blocks[-1])
        except (anthropic.APIStatusError, anthropic.APIConnectionError,
                ValueError, json.JSONDecodeError) as exc:
            if attempt == MAX_RETRIES:
                raise
            wait = 2 ** attempt  # 2s, 4s
            print(f"  attempt {attempt} failed ({exc}); retrying in {wait}s")
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Batch run, caching, and scoring
# ---------------------------------------------------------------------------

def load_cache(cache_path: str) -> dict:
    """Load previously completed predictions keyed by report ID."""
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                cache[record[ID_COL]] = record
    return cache


def score(pred_df: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Print accuracy, macro F1, and a confusion matrix against FDA labels."""
    merged = pred_df.merge(rows[[ID_COL, "event_type"]], on=ID_COL,
                           suffixes=("_llm", "_fda"))
    if merged.empty:
        return
    y_true, y_pred = merged["event_type_fda"], merged["event_type_llm"]
    print(f"\nScored {len(merged)} of {len(rows)} reports")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1: {f1_score(y_true, y_pred, average='macro'):.4f}")
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    print("Confusion matrix (rows = FDA, columns = LLM):")
    print(pd.DataFrame(cm, index=LABEL_ORDER, columns=LABEL_ORDER))


def run_batch(input_csv: str, prompt_version: str, test_file: str | None,
              train_sample: int | None, limit: int | None) -> None:
    """Classify the selected reports and write predictions to a CSV."""
    load_dotenv()
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    system_prompt = PROMPTS[prompt_version]

    df = pd.read_csv(input_csv, dtype={ID_COL: str})
    if test_file:
        # Use the group split from notebook 02 (GroupShuffleSplit keeps
        # identical narratives on the same side of the split)
        rows = pd.read_csv(test_file, dtype={ID_COL: str})
        split_name = "testgroup"
    elif train_sample:
        train, _ = split_train_test(df)
        rows = sample_train_rows(train, train_sample)
        split_name = f"train{train_sample}"
    else:
        rows, split_name = df, "all"
    if limit:
        rows = rows.head(limit)

    cache_path = f"results/cache_{prompt_version}_{split_name}.jsonl"
    output_csv = f"results/llm_{prompt_version}_{split_name}.csv"
    os.makedirs("results", exist_ok=True)

    cache = load_cache(cache_path)
    # Reuse predictions from earlier runs: same prompt + same report_id
    # means the same input, so there is no need to pay for it again.
    old_caches = [f"results/cache_{prompt_version}_test.jsonl",
                  f"results/cache_{prompt_version}_train100.jsonl"]
    if prompt_version == "v1":
        old_caches.append("results/cache_v1_test_full1000.jsonl")
    for old in old_caches:
        if os.path.exists(old):
            for k, v in load_cache(old).items():
                cache.setdefault(k, v)

    n_cached = rows[ID_COL].isin(cache.keys()).sum()
    print(f"Prompt {prompt_version} | {split_name} | {len(rows)} reports; "
          f"{n_cached} already cached.")

    with open(cache_path, "a", encoding="utf-8") as cache_file:
        for i, row in rows.iterrows():
            key = row[ID_COL]
            if key in cache:
                continue  # already done on a previous run, so skip it

            narrative = str(row["narrative"]).strip()
            if not narrative or narrative.lower() == "nan":
                print(f"[{i}] {key}: empty narrative, skipping")
                continue

            try:
                prediction = extract_one(client, narrative, system_prompt)
            except Exception as exc:
                print(f"[{i}] {key}: FAILED after retries ({exc})")
                continue

            record = {ID_COL: key, **prediction}
            cache[key] = record
            cache_file.write(json.dumps(record) + "\n")
            cache_file.flush()  # survive crashes mid-run

            print(f"[{i}] {key}: {prediction['event_type']}"
                  f" | harm={prediction['patient_harm']}"
                  f" | {prediction['failure_mode']}")
            time.sleep(SLEEP_BETWEEN_CALLS)

    # Keep only predictions for the rows selected in this run
    out_df = pd.DataFrame([cache[k] for k in rows[ID_COL] if k in cache])
    out_df.to_csv(output_csv, index=False)
    print(f"\nWrote {len(out_df)} predictions to {output_csv}")

    score(out_df, rows)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify MAUDE narratives with an LLM via the API")
    parser.add_argument("--input", required=True,
                        help="CSV with report_id, narrative, event_type")
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="v2",
                        help="Prompt version to use (default: v2)")
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--test-file", default=None,
                       help="CSV with report_id, narrative, event_type "
                            "(e.g. data/test_group_split.csv)")
    which.add_argument("--train-sample", type=int, default=None,
                       help="Run a stratified sample of N training reports")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N selected rows")
    args = parser.parse_args()

    run_batch(args.input, args.prompt, args.test_file,
              args.train_sample, args.limit)
