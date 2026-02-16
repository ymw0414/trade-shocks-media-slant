# Trade Shocks and Media Slant

Measuring how NAFTA trade exposure shifted U.S. newspaper slant using text analysis of Congressional speeches and newspaper articles (1987–2000).

## Method

1. **Partisan language model** — Build a vocabulary of partisan phrases from Congressional Record speeches (Congresses 99–108) using rolling-window LASSO on L1-normalized count features, trained on ideologically extreme legislators identified via Nokken-Poole scores.

2. **Newspaper scoring** — Project the partisan language model onto newspaper articles to produce four measures per article: right intensity, left intensity, net slant, and politicization. Normalize by the Congressional partisan gap for cross-year comparability.

3. **NAFTA exposure** — Construct commuting-zone-level NAFTA tariff vulnerability following Choi et al. (2024): employment-and-RCA-weighted average tariff reductions, scaled by the interquartile range.

4. **Event study & DiD** — Estimate the causal effect of NAFTA vulnerability on newspaper slant using two-way fixed effects (newspaper + year + division×year), with China shock and manufacturing share as controls.

## Pipeline

```
scripts/
├── nlp/                         # Congressional speech → partisan language
│   ├── 01_load_speeches.py          Load Hein-Bound speech files
│   ├── 02_merge_speaker_map.py      Map speaker IDs to metadata
│   ├── 03_add_party_label.py        Add party affiliation
│   ├── 04_label_partisan_core.py    Identify extreme partisans (Nokken-Poole)
│   ├── 05_build_features.py          Build feature matrix (CountVectorizer + L1)
│   ├── 06_train_lasso.py            Train rolling-window LASSO classifiers
│   ├── 07_prepare_newspapers.py     Vectorize newspaper articles
│   ├── 08_project_slant.py          Score articles with LASSO coefficients
│   ├── 09_normalize_slant.py        Normalize by partisan gap
│   └── 10_aggregate_slant.py        Aggregate to newspaper-year panel
│
├── preprocessing/               # Newspaper data cleaning
│   ├── 01_standardize_yearly_csv.py
│   ├── 02_standardize_paper_names.py
│   ├── 03_apply_crosswalk.py
│   └── 04_label_articles.py
│
├── econ/                        # Economic exposure variables
│   ├── 11_merge_geography.py        County FIPS & commuting zones
│   ├── 12_build_nafta_vars.py       NAFTA tariff vulnerability
│   ├── 13_build_china_shock.py      ADH China import shock
│   └── 14_merge_panel.py            Combine into regression panel
│
├── analysis/                    # Estimation
│   ├── 15_event_study.py            Event study (baseline + controls)
│   └── 16_did_regression.py         DiD with sequential robustness specs
│
├── figures/
│   └── map_nafta_exposure.py        Choropleth of NAFTA vulnerability
│
└── utils/
    ├── text_analyzer.py             Shared text processing utilities
    └── validate_nafta_vars.py       Validation against replication data
```

## Setup

```bash
# Set environment variable pointing to the project root
export SHIFTING_SLANT_DIR=/path/to/shifting_slant

# Install dependencies
pip install -r requirements.txt
```

Scripts are designed to be run sequentially by step number. Each script reads from and writes to `data/` subdirectories relative to `SHIFTING_SLANT_DIR`.

## Data

Not included in this repository. Raw data sources:

- **Congressional Record**: Hein-Bound (Gentzkow, Shapiro & Taddy)
- **Newspaper articles**: NewsLibrary
- **Trade data**: USITC, Feenstra (1996), UN Comtrade
- **Employment**: County Business Patterns (CBP)
- **China shock**: Autor, Dorn & Hanson (2013)
- **Geography**: David Dorn CZ crosswalks, Widmer newspaper-county mapping

## References

- Choi, J., Kuziemko, I., Washington, E., & Wright, G. (2024). "Local Economic and Political Effects of Trade Deals: Evidence from NAFTA." *American Economic Review*.
- Gentzkow, M. & Shapiro, J. (2010). "What Drives Media Slant? Evidence from U.S. Daily Newspapers." *Econometrica*.
- Autor, D., Dorn, D. & Hanson, G. (2013). "The China Syndrome." *American Economic Review*.
