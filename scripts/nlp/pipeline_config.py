"""
Central configuration for the NLP pipeline (steps 05-16).

To create a new run with different settings:
  1. Change RUN_NAME (e.g., "robustness_3win")
  2. Adjust CONFIG values as needed
  3. Run the pipeline: 05 → 07 → 06 → 08 → 09 → 10 → 11 → 14 → 15 → 16

Each run saves outputs to its own directory.
A config.json is saved alongside for reproducibility.

Override mechanism for experiments:
  Set env var PIPELINE_CONFIG_OVERRIDE to a JSON file path.
  The JSON can override any CONFIG value and/or run_name.
  Example: {"run_name": "exp_df100", "newspaper_df_floor": 100}
"""

import json
import os
from pathlib import Path

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

# ── Run name ───────────────────────────────────────────────────────
RUN_NAME = "exp_shvocab_cv"

# ── Pipeline settings ──────────────────────────────────────────────
CONFIG = {
    "run_name": RUN_NAME,
    "congress_range": [99, 108],        # inclusive: speeches from these congresses
    "window_size": 1,                   # 1-congress (Widmer-style) or 2/3 (rolling)
    "ngram_range": (1, 2),              # (2,2)=bigrams only (Widmer), (1,2)=unigrams+bigrams
    "bigrams_only": False,              # True=Widmer, False=unigrams+bigrams
    "use_relative_freq": True,          # True=Widmer (CountVec+L1norm), False=legacy TF-IDF
    "freq_filter_mode": "widmer",       # "widmer" = per-party 0.1%/0.01%, "min_df" = global min_df
    "tfidf_min_df": 0.001,             # only used when freq_filter_mode="min_df"
    "partisan_core_only": False,        # False=all R vs D (Widmer), True=core only
    "min_speech_words": None,           # None = no filter
    "aggregate_to_legislator": False,   # False=individual speeches (Widmer), True=legislator-congress
    "newspaper_df_floor": None,         # None=no floor (Widmer), int=min newspaper DF
    "lasso_Cs": [-1, 4, 20],           # np.logspace(*Cs) args
    "lasso_lambda_selection": "cv",     # "bic" (Widmer) or "cv"
    "lasso_cv_folds": 5,               # only used when lambda_selection="cv"
    "lasso_max_iter": 5000,
    "shared_vocab_min_df": 0.00001,     # None=off, float=min newspaper DF (0.00001=0.001%)
    "filter_gst_procedural": True,     # True=remove GST procedural bigrams from vocabulary
    "sign_consistency_filter": False,   # True=zero out phrases that flip R/D across congresses
    "norm_method": "raw_gap",           # "raw_gap" | "prob_gap" | "prob_direct" (P(R) as score)
    "speech_sample_frac": None,         # None=full sample, float (0,1)=random subsample for dev
    "newspaper_sample_frac": None,      # None=full sample, float (0,1)=random subsample for dev
    "input_speech_dir": "data/processed/runs/exp_unigram_gst/speeches",
    "input_news_dir": "data/processed/runs/exp_unigram_gst/newspapers",
}

# ── Override from environment (for experiments) ────────────────────
_override_path = os.environ.get("PIPELINE_CONFIG_OVERRIDE")
if _override_path:
    with open(_override_path) as f:
        _overrides = json.load(f)
    RUN_NAME = _overrides.pop("run_name", RUN_NAME)
    CONFIG.update(_overrides)
    CONFIG["run_name"] = RUN_NAME
    print(f"  [pipeline_config] Override loaded: run={RUN_NAME}")
    for k, v in _overrides.items():
        print(f"    {k} = {v}")

# ── National newspapers to exclude from analysis ──────────────────
# These are national papers whose content does not reflect local coverage.
# They get mapped to their HQ location in step 11, which is misleading
# for a local trade-shock identification strategy.
NATIONAL_PAPERS = [
    "USA TODAY",
    "CHRISTIAN SCIENCE MONITOR",
]

# ── Non-daily newspapers to exclude from analysis ─────────────────
# These papers are not dailies (weekly, Sunday-only, or archive artifacts).
# Retaining them would mix publication frequencies in the panel.
NON_DAILY_PAPERS = [
    "Quad-City Times, The (IA)",           # archive artifact (<400 articles/yr until 2003)
    "Miami New Times (FL)",                 # alternative weekly
    "Hunterdon Observer (Flemington, NJ)",  # weekly/semi-weekly
    "New Hampshire Sunday News (Manchester, NH)",  # Sunday-only edition
]

# ── Shared input paths (fixed, independent of run) ────────────────
SPEECHES_PATH       = BASE_DIR / "data" / "intermediate" / "speeches" / "01_speeches_merged.parquet"
LABELS_PATH         = BASE_DIR / "data" / "intermediate" / "speeches" / "04_speeches_with_partisan_core.parquet"
VOTEVIEW_PATH       = BASE_DIR / "data" / "raw" / "voteview_nominate" / "HSall_members.csv"
RAW_NEWSPAPERS      = BASE_DIR / "data" / "intermediate" / "newspapers"
NEWSPAPER_LABELS    = BASE_DIR / "data" / "processed" / "newspapers"   # step 04 labels
NEWSPAPER_FEATURES_DIR = BASE_DIR / "data" / "processed" / "newspapers"   # step 07 feature outputs
SPEECH_FEATURES_DIR    = BASE_DIR / "data" / "processed" / "speeches"     # step 05 feature outputs
GST_PROCEDURAL_PATH = BASE_DIR / "data" / "raw" / "speeches" / "vocabulary" / "procedural.txt"

# Econ inputs (steps 12-13, shared across runs)
GEO_PATH         = BASE_DIR / "data" / "geo" / "newspaper_county_map.csv"
CZ_PATH          = BASE_DIR / "data" / "raw" / "econ" / "crosswalk" / "cw_cty_czone" / "cw_cty_czone.dta"
CZ_NAFTA         = BASE_DIR / "data" / "processed" / "econ" / "12_nafta_vars_cz.parquet"
COUNTY_NAFTA     = BASE_DIR / "data" / "processed" / "econ" / "12_nafta_vars_county.parquet"
CHINA_SHOCK_PATH = BASE_DIR / "data" / "processed" / "econ" / "13_china_shock_cz.parquet"

# ── Run-specific output paths ─────────────────────────────────────
if RUN_NAME == "main":
    # Main run: backward-compatible flat structure
    RUN_DIR    = BASE_DIR / "data" / "processed" / "runs" / "main"
    SPEECH_DIR = SPEECH_FEATURES_DIR
    MODEL_DIR  = SPEECH_DIR / "models"
    NEWS_DIR   = BASE_DIR / "data" / "processed" / "newspapers"
    PANEL_DIR  = BASE_DIR / "data" / "processed" / "panel"
    FIG_DIR    = BASE_DIR / "output" / "figures"
    TAB_DIR    = BASE_DIR / "output" / "tables"
else:
    # Experiment: isolated run directory
    RUN_DIR    = BASE_DIR / "data" / "processed" / "runs" / RUN_NAME
    SPEECH_DIR = RUN_DIR / "speeches"
    MODEL_DIR  = RUN_DIR / "models"
    NEWS_DIR   = RUN_DIR / "newspapers"
    PANEL_DIR  = RUN_DIR / "panel"
    FIG_DIR    = RUN_DIR / "output" / "figures"
    TAB_DIR    = RUN_DIR / "output" / "tables"


# ── Input paths for reading shared step 05/07 outputs ────────────
# For "cheap" experiments that reuse feature matrices from another run,
# set these in the override JSON to point to the source run's directories.
_input_speech = CONFIG.get("input_speech_dir")
_input_news = CONFIG.get("input_news_dir")
INPUT_SPEECH_DIR = (BASE_DIR / _input_speech) if _input_speech else SPEECH_DIR
INPUT_NEWS_DIR = (BASE_DIR / _input_news) if _input_news else NEWS_DIR


def save_config():
    """Save the current config alongside run outputs."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUN_DIR / "config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)
    print(f"  Config saved -> {RUN_DIR / 'config.json'}")


def get_congresses():
    """Return list of congresses to score (e.g., 100-108)."""
    lo, hi = CONFIG["congress_range"]
    return list(range(lo + 1, hi + 1))


def get_windows():
    """Generate rolling windows based on config."""
    congresses = get_congresses()
    ws = CONFIG["window_size"]
    if ws == 1:
        return [(c,) for c in congresses]
    elif ws == 2:
        return [(c - 1, c) for c in congresses]
    elif ws == 3:
        return [(c - 2, c - 1, c) for c in congresses]
    else:
        raise ValueError(f"Unsupported window_size: {ws}")
