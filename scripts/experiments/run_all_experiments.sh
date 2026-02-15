#!/bin/bash
# Run all 6 experiments with shared vocabulary
# Speech-level step 05 already done (exp_unigram_gst)
# Strategy: 1x step 07, 1x step 05b, 6x step 06, 6x steps 08-16

set -e
export SHIFTING_SLANT_DIR="C:/Users/ymw04/Dropbox/shifting_slant"
export PYTHONPATH="$SHIFTING_SLANT_DIR/scripts/utils"
PYTHON="/c/Users/ymw04/miniconda3/envs/nlp/python.exe"
SCRIPTS="$SHIFTING_SLANT_DIR/scripts"
EXP_DIR="$SHIFTING_SLANT_DIR/experiments"

echo "=========================================="
echo "  PHASE 1: Step 05b - Legislator-level DTM"
echo "=========================================="
# Create override for legislator-level (pointing to speech-level for input)
cat > "$EXP_DIR/exp_uni_gst_leg.json" << 'EOF'
{
    "run_name": "exp_uni_gst_leg",
    "bigrams_only": false,
    "filter_gst_procedural": true,
    "aggregate_to_legislator": true,
    "partisan_core_only": false,
    "lasso_lambda_selection": "bic",
    "speech_sample_frac": null,
    "newspaper_sample_frac": null,
    "input_speech_dir": "data/processed/runs/exp_unigram_gst/speeches"
}
EOF

PIPELINE_CONFIG_OVERRIDE="$EXP_DIR/exp_uni_gst_leg.json" \
    $PYTHON "$SCRIPTS/nlp/05b_aggregate_dtm.py"

echo ""
echo "=========================================="
echo "  PHASE 2: Step 07 - Newspaper Transform"
echo "  (using speech-level vocabulary, one time)"
echo "=========================================="
# Use exp_unigram_gst (speech-level) for step 07
PIPELINE_CONFIG_OVERRIDE="$EXP_DIR/exp_unigram_gst.json" \
    $PYTHON "$SCRIPTS/nlp/07_prepare_newspapers.py"

echo ""
echo "=========================================="
echo "  PHASE 3: Step 06 - Train LASSO (6 variants)"
echo "=========================================="

# Experiment 1: speech, all, BIC (uses exp_unigram_gst speech dir directly)
cat > "$EXP_DIR/exp_unigram_gst.json" << 'EOF'
{
    "run_name": "exp_unigram_gst",
    "bigrams_only": false,
    "filter_gst_procedural": true,
    "aggregate_to_legislator": false,
    "partisan_core_only": false,
    "lasso_lambda_selection": "bic",
    "speech_sample_frac": null,
    "newspaper_sample_frac": null
}
EOF
echo "--- Exp 1: Speech, All R/D, BIC ---"
PIPELINE_CONFIG_OVERRIDE="$EXP_DIR/exp_unigram_gst.json" \
    $PYTHON "$SCRIPTS/nlp/06_train_lasso.py"

# Experiment 4: speech, all, CV
cat > "$EXP_DIR/exp_uni_gst_cv.json" << 'EOF'
{
    "run_name": "exp_uni_gst_cv",
    "bigrams_only": false,
    "filter_gst_procedural": true,
    "aggregate_to_legislator": false,
    "partisan_core_only": false,
    "lasso_lambda_selection": "cv",
    "speech_sample_frac": null,
    "newspaper_sample_frac": null,
    "input_speech_dir": "data/processed/runs/exp_unigram_gst/speeches",
    "input_news_dir": "data/processed/runs/exp_unigram_gst/newspapers"
}
EOF
echo "--- Exp 4: Speech, All R/D, CV ---"
PIPELINE_CONFIG_OVERRIDE="$EXP_DIR/exp_uni_gst_cv.json" \
    $PYTHON "$SCRIPTS/nlp/06_train_lasso.py"

# Experiment 2: legislator, all, BIC
echo "--- Exp 2: Legislator, All R/D, BIC ---"
PIPELINE_CONFIG_OVERRIDE="$EXP_DIR/exp_uni_gst_leg.json" \
    $PYTHON "$SCRIPTS/nlp/06_train_lasso.py"

# Experiment 3: legislator, core, BIC
cat > "$EXP_DIR/exp_uni_gst_leg_core.json" << 'EOF'
{
    "run_name": "exp_uni_gst_leg_core",
    "bigrams_only": false,
    "filter_gst_procedural": true,
    "aggregate_to_legislator": true,
    "partisan_core_only": true,
    "lasso_lambda_selection": "bic",
    "speech_sample_frac": null,
    "newspaper_sample_frac": null,
    "input_speech_dir": "data/processed/runs/exp_uni_gst_leg/speeches",
    "input_news_dir": "data/processed/runs/exp_unigram_gst/newspapers"
}
EOF
echo "--- Exp 3: Legislator, Core 20%, BIC ---"
PIPELINE_CONFIG_OVERRIDE="$EXP_DIR/exp_uni_gst_leg_core.json" \
    $PYTHON "$SCRIPTS/nlp/06_train_lasso.py"

# Experiment 5: legislator, all, CV
cat > "$EXP_DIR/exp_uni_gst_leg_cv.json" << 'EOF'
{
    "run_name": "exp_uni_gst_leg_cv",
    "bigrams_only": false,
    "filter_gst_procedural": true,
    "aggregate_to_legislator": true,
    "partisan_core_only": false,
    "lasso_lambda_selection": "cv",
    "speech_sample_frac": null,
    "newspaper_sample_frac": null,
    "input_speech_dir": "data/processed/runs/exp_uni_gst_leg/speeches",
    "input_news_dir": "data/processed/runs/exp_unigram_gst/newspapers"
}
EOF
echo "--- Exp 5: Legislator, All R/D, CV ---"
PIPELINE_CONFIG_OVERRIDE="$EXP_DIR/exp_uni_gst_leg_cv.json" \
    $PYTHON "$SCRIPTS/nlp/06_train_lasso.py"

# Experiment 6: legislator, core, CV
cat > "$EXP_DIR/exp_uni_gst_leg_core_cv.json" << 'EOF'
{
    "run_name": "exp_uni_gst_leg_core_cv",
    "bigrams_only": false,
    "filter_gst_procedural": true,
    "aggregate_to_legislator": true,
    "partisan_core_only": true,
    "lasso_lambda_selection": "cv",
    "speech_sample_frac": null,
    "newspaper_sample_frac": null,
    "input_speech_dir": "data/processed/runs/exp_uni_gst_leg/speeches",
    "input_news_dir": "data/processed/runs/exp_unigram_gst/newspapers"
}
EOF
echo "--- Exp 6: Legislator, Core 20%, CV ---"
PIPELINE_CONFIG_OVERRIDE="$EXP_DIR/exp_uni_gst_leg_core_cv.json" \
    $PYTHON "$SCRIPTS/nlp/06_train_lasso.py"

echo ""
echo "=========================================="
echo "  PHASE 4: Steps 08-16 for all experiments"
echo "=========================================="

# Run steps 08-16 for each experiment
for exp_json in \
    exp_unigram_gst.json \
    exp_uni_gst_cv.json \
    exp_uni_gst_leg.json \
    exp_uni_gst_leg_core.json \
    exp_uni_gst_leg_cv.json \
    exp_uni_gst_leg_core_cv.json
do
    exp_name="${exp_json%.json}"
    echo ""
    echo "--- $exp_name: Steps 08-16 ---"

    for step in \
        nlp/08_project_slant.py \
        nlp/09_normalize_slant.py \
        nlp/10_aggregate_slant.py \
        econ/11_merge_geography.py \
        econ/14_merge_panel.py \
        analysis/15_event_study.py \
        analysis/16_did_regression.py
    do
        echo "  Running $step ..."
        PIPELINE_CONFIG_OVERRIDE="$EXP_DIR/$exp_json" \
            $PYTHON "$SCRIPTS/$step" || { echo "  FAILED: $step for $exp_name"; break; }
    done
done

echo ""
echo "=========================================="
echo "  PHASE 5: Comparison"
echo "=========================================="
$PYTHON -c "
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
from pathlib import Path
BASE = Path('$SHIFTING_SLANT_DIR')
exps = [
    ('exp_unigram_gst',          'Speech, All, BIC'),
    ('exp_uni_gst_cv',           'Speech, All, CV'),
    ('exp_uni_gst_leg',          'Leg, All, BIC'),
    ('exp_uni_gst_leg_core',     'Leg, Core, BIC'),
    ('exp_uni_gst_leg_cv',       'Leg, All, CV'),
    ('exp_uni_gst_leg_core_cv',  'Leg, Core, CV'),
]
results = []
for name, label in exps:
    did_path = BASE / 'data/processed/runs' / name / 'output/tables/did_results.csv'
    if did_path.exists():
        df = pd.read_csv(did_path)
        df['experiment'] = name
        df['exp_label'] = label
        results.append(df)
    else:
        print(f'  Missing: {did_path}')

if results:
    all_did = pd.concat(results, ignore_index=True)
    out = BASE / 'output/tables/experiment_comparison.csv'
    all_did.to_csv(out, index=False)
    print(f'Saved: {out}')

    # Training summary comparison
    print()
    print('=== Training Summary ===')
    for name, label in exps:
        train_path = BASE / 'data/processed/runs' / name / 'models/06_training_summary.csv'
        if train_path.exists():
            t = pd.read_csv(train_path)
            avg_acc = t['train_accuracy'].mean()
            avg_k = t['n_nonzero_coefs'].mean()
            print(f'  {label:25s}  acc={avg_acc:.3f}  avg_k={avg_k:.0f}')

    # DiD comparison for key outcomes
    for outcome in ['int_R', 'int_D', 'net_slant_norm', 'ext_nonzero']:
        sub = all_did[(all_did['depvar'] == outcome) & (all_did['spec'] == 'spec1')]
        if len(sub) > 0:
            print(f'\n  {outcome} (Spec 1):')
            for _, row in sub.iterrows():
                sig = '**' if row['pval'] < 0.05 else '*' if row['pval'] < 0.1 else ''
                print(f'    {row[\"exp_label\"]:25s}  b={row[\"coef\"]:+.4f}  se={row[\"se\"]:.4f}  p={row[\"pval\"]:.3f}{sig}')
"

echo ""
echo "ALL EXPERIMENTS COMPLETE"
