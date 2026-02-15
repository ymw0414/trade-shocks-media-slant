"""
01_mine_topics.py — GST-filtered topic mining with partisan phrase labeling.

Per 2-congress window:
  Phase A: Load GST procedural bigrams (46K) and partisan phrases
  Phase B: Filter speeches (partisan core + GST procedural ratio + word count)
  Phase C: K-means on filtered SBERT embeddings -> 200 clusters
  Phase D: Select top-30 polarized clusters, label with GST partisan phrases
  Phase E: Extract R/D indicator speeches (closest to party centroid)

Output per window:
  topics_window_{prev}_{curr}.parquet   — topic info + indicators + GST labels
  topic_centroids_window_{prev}_{curr}.npy — cluster centroids for topic retrieval
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.prism.config import (
    SBERT_DIR, SPEECHES_PATH, LABELS_PATH, OUTPUT_DIR,
    GST_PROCEDURAL_PATH, GST_PARTISAN_DIR,
    N_CLUSTERS, N_POLARIZED_TOPICS, INDICATOR_MAX_WORDS, MIN_CLUSTER_SIZE,
    PARTISAN_CORE_ONLY, MIN_WORDS_AFTER_CLEAN, COMMON_WORD_RATIO_THRESHOLD,
    PROCEDURAL_RATIO_THRESHOLD, COMMON_WORDS,
    get_windows,
)

from nltk.stem import PorterStemmer

STEMMER = PorterStemmer()


# ── Phase A: Load GST data ──────────────────────────────────────────────

def load_gst_procedural():
    """Load GST procedural bigrams (46K stemmed bigrams)."""
    procedural = set()
    with open(GST_PROCEDURAL_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0 and "phrase|type" in line:
                continue  # skip header
            parts = line.strip().split("|")
            if len(parts) >= 2:
                procedural.add(parts[0].strip())
    print(f"  GST procedural bigrams loaded: {len(procedural):,}")
    return procedural


def load_gst_partisan_phrases(congresses):
    """Load GST partisan phrases for given congresses.

    Returns dict: {phrase: partisanship_score} averaged across congresses.
    """
    all_phrases = {}
    counts = {}
    for cong in congresses:
        path = GST_PARTISAN_DIR / f"partisan_phrases_{cong:03d}.txt"
        if not path.exists():
            print(f"  WARNING: {path.name} not found")
            continue
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == 0 and "phrase|partisanship" in line:
                    continue
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    phrase = parts[0].strip()
                    try:
                        score = float(parts[1].strip())
                    except ValueError:
                        continue
                    all_phrases[phrase] = all_phrases.get(phrase, 0) + score
                    counts[phrase] = counts.get(phrase, 0) + 1

    # Average across congresses
    for phrase in all_phrases:
        all_phrases[phrase] /= counts[phrase]

    print(f"  GST partisan phrases loaded: {len(all_phrases):,} "
          f"(from {len(congresses)} congresses)")
    return all_phrases


# ── Phase B: Speech filtering ───────────────────────────────────────────

def stem_and_bigram(text):
    """Tokenize, stem, and extract bigrams from text."""
    tokens = text.lower().split()
    stemmed = [STEMMER.stem(t) for t in tokens]
    bigrams = [f"{stemmed[i]} {stemmed[i+1]}" for i in range(len(stemmed) - 1)]
    return stemmed, bigrams


def compute_procedural_ratio(bigrams, procedural_set):
    """Fraction of bigrams that are procedural."""
    if not bigrams:
        return 1.0
    n_proc = sum(1 for b in bigrams if b in procedural_set)
    return n_proc / len(bigrams)


def compute_common_word_ratio(tokens):
    """Fraction of tokens that are common/stop words."""
    if not tokens:
        return 1.0
    n_common = sum(1 for t in tokens if t in COMMON_WORDS)
    return n_common / len(tokens)


def filter_speeches(meta, speech_texts, procedural_set):
    """Apply 3-level filtering to speeches.

    Level 1: GST procedural ratio < threshold (replaces 151 regex)
    Level 2: Min word count after cleaning
    Level 3: Common word ratio (not too generic)

    Returns boolean mask aligned with meta index.
    """
    n_total = len(meta)
    keep = np.ones(n_total, dtype=bool)

    # Merge texts
    text_map = dict(zip(speech_texts["speech_id"], speech_texts["speech"]))

    proc_ratios = []
    common_ratios = []
    word_counts = []

    for idx, row in enumerate(meta.itertuples()):
        sid = row.speech_id
        text = text_map.get(sid, "")
        if not text or not isinstance(text, str):
            keep[idx] = False
            proc_ratios.append(1.0)
            common_ratios.append(1.0)
            word_counts.append(0)
            continue

        tokens = text.lower().split()
        word_counts.append(len(tokens))

        # Level 2: min words
        if len(tokens) < MIN_WORDS_AFTER_CLEAN:
            keep[idx] = False
            proc_ratios.append(0.0)
            common_ratios.append(0.0)
            continue

        stemmed, bigrams = stem_and_bigram(text)

        # Level 1: GST procedural
        pr = compute_procedural_ratio(bigrams, procedural_set)
        proc_ratios.append(pr)
        if pr > PROCEDURAL_RATIO_THRESHOLD:
            keep[idx] = False
            continue

        # Level 3: common word ratio
        cr = compute_common_word_ratio(tokens)
        common_ratios.append(cr)
        if cr > (1 - COMMON_WORD_RATIO_THRESHOLD):
            keep[idx] = False
            continue

    print(f"  Filtering: {n_total:,} -> {keep.sum():,} speeches "
          f"({keep.sum()/n_total*100:.1f}%)")
    print(f"    Procedural filter removed: {sum(1 for p in proc_ratios if p > PROCEDURAL_RATIO_THRESHOLD):,}")
    print(f"    Short speech filter: {sum(1 for w in word_counts if w < MIN_WORDS_AFTER_CLEAN):,}")

    return keep


# ── Phase C: K-means clustering ────────────────────────────────────────

def cluster_speeches(embeddings, n_clusters=N_CLUSTERS):
    """K-means clustering on speech embeddings."""
    from sklearn.cluster import KMeans

    print(f"  K-means clustering: {len(embeddings):,} speeches -> {n_clusters} clusters")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(embeddings)
    centroids = km.cluster_centers_
    inertia = km.inertia_

    # Cluster size stats
    sizes = np.bincount(labels, minlength=n_clusters)
    print(f"    Inertia: {inertia:.0f}")
    print(f"    Cluster sizes: min={sizes.min()}, median={np.median(sizes):.0f}, "
          f"max={sizes.max()}")

    return labels, centroids


# ── Phase D: Select polarized topics + GST labeling ────────────────────

def select_polarized_topics(cluster_labels, meta, n_top=N_POLARIZED_TOPICS,
                            min_size=MIN_CLUSTER_SIZE):
    """Select most polarized clusters based on R/D composition."""
    n_clusters = cluster_labels.max() + 1
    topics = []

    for k in range(n_clusters):
        mask = cluster_labels == k
        n_k = mask.sum()
        if n_k < min_size:
            continue

        parties = meta.iloc[mask.nonzero()[0]]["party"].values
        n_R = (parties == "R").sum()
        n_D = (parties == "D").sum()
        n_partisan = n_R + n_D
        if n_partisan == 0:
            continue

        r_prop = n_R / n_partisan
        polarization = abs(r_prop - 0.5) * 2  # 0=balanced, 1=all one party

        topics.append({
            "cluster_id": k,
            "n_speeches": n_k,
            "n_R": n_R,
            "n_D": n_D,
            "r_proportion": r_prop,
            "polarization": polarization,
        })

    topics_df = pd.DataFrame(topics).sort_values("polarization", ascending=False)
    selected = topics_df.head(n_top).copy()
    selected["topic_idx"] = range(len(selected))

    print(f"  Selected {len(selected)} polarized topics "
          f"(from {len(topics)} valid clusters)")
    print(f"    Polarization range: {selected['polarization'].min():.3f} - "
          f"{selected['polarization'].max():.3f}")
    print(f"    R-leaning: {(selected['r_proportion'] > 0.5).sum()}, "
          f"D-leaning: {(selected['r_proportion'] <= 0.5).sum()}")

    return selected


def label_topics_with_gst(selected_topics, cluster_labels, meta,
                          speech_texts, partisan_phrases):
    """Label each selected topic using GST partisan phrases.

    For each cluster: stem+bigram all speeches -> find most frequent
    GST partisan phrases -> use top phrases as topic label.
    """
    text_map = dict(zip(speech_texts["speech_id"], speech_texts["speech"]))
    topic_labels = []
    top_r_phrases_list = []
    top_d_phrases_list = []

    for _, topic in selected_topics.iterrows():
        k = topic["cluster_id"]
        mask = cluster_labels == k
        speech_ids = meta.iloc[mask.nonzero()[0]]["speech_id"].values

        # Collect bigrams from all speeches in cluster
        phrase_counts = Counter()
        for sid in speech_ids:
            text = text_map.get(sid, "")
            if not text or not isinstance(text, str):
                continue
            _, bigrams = stem_and_bigram(text)
            for bg in bigrams:
                if bg in partisan_phrases:
                    phrase_counts[bg] += 1

        # Top partisan phrases
        top_phrases = phrase_counts.most_common(20)

        # Separate R (positive score) and D (negative score) phrases
        r_phrases = [(p, c, partisan_phrases[p])
                     for p, c in top_phrases if partisan_phrases[p] > 0]
        d_phrases = [(p, c, partisan_phrases[p])
                     for p, c in top_phrases if partisan_phrases[p] < 0]

        # Topic label = top 3 most frequent partisan phrases
        label_parts = [p for p, c in top_phrases[:3]]
        label = " / ".join(label_parts) if label_parts else f"cluster_{k}"

        topic_labels.append(label)
        top_r_phrases_list.append(
            "; ".join(f"{p}({s:+.1f})" for p, c, s in r_phrases[:5])
        )
        top_d_phrases_list.append(
            "; ".join(f"{p}({s:+.1f})" for p, c, s in d_phrases[:5])
        )

    selected_topics = selected_topics.copy()
    selected_topics["topic_label"] = topic_labels
    selected_topics["top_r_phrases"] = top_r_phrases_list
    selected_topics["top_d_phrases"] = top_d_phrases_list

    print(f"  Topic labeling complete. Top 5 topics:")
    for _, t in selected_topics.head(5).iterrows():
        print(f"    [{t['topic_idx']}] {t['topic_label']} "
              f"(R={t['n_R']}, D={t['n_D']}, pol={t['polarization']:.3f})")

    return selected_topics


# ── Phase E: Extract indicator speeches ─────────────────────────────────

def extract_indicators(selected_topics, cluster_labels, embeddings,
                       meta, speech_texts, centroids):
    """Extract R/D indicator speeches for each selected topic.

    Indicator = speech closest to party centroid within cluster.
    Truncate to INDICATOR_MAX_WORDS.
    """
    text_map = dict(zip(speech_texts["speech_id"], speech_texts["speech"]))

    r_texts, d_texts = [], []
    r_ids, d_ids = [], []

    for _, topic in selected_topics.iterrows():
        k = topic["cluster_id"]
        mask = cluster_labels == k
        idxs = mask.nonzero()[0]

        cluster_meta = meta.iloc[idxs]
        cluster_emb = embeddings[idxs]

        # Split R/D within cluster
        r_mask = cluster_meta["party"].values == "R"
        d_mask = cluster_meta["party"].values == "D"

        r_text, d_text = "", ""
        r_id, d_id = "", ""

        if r_mask.sum() > 0:
            r_emb = cluster_emb[r_mask]
            r_centroid = r_emb.mean(axis=0)
            # Closest to R centroid
            dists = np.linalg.norm(r_emb - r_centroid, axis=1)
            best_r_local = dists.argmin()
            best_r_global = idxs[r_mask.nonzero()[0][best_r_local]]
            r_id = meta.iloc[best_r_global]["speech_id"]
            raw = text_map.get(r_id, "")
            r_text = " ".join(raw.split()[:INDICATOR_MAX_WORDS]) if raw else ""

        if d_mask.sum() > 0:
            d_emb = cluster_emb[d_mask]
            d_centroid = d_emb.mean(axis=0)
            dists = np.linalg.norm(d_emb - d_centroid, axis=1)
            best_d_local = dists.argmin()
            best_d_global = idxs[d_mask.nonzero()[0][best_d_local]]
            d_id = meta.iloc[best_d_global]["speech_id"]
            raw = text_map.get(d_id, "")
            d_text = " ".join(raw.split()[:INDICATOR_MAX_WORDS]) if raw else ""

        r_texts.append(r_text)
        d_texts.append(d_text)
        r_ids.append(r_id)
        d_ids.append(d_id)

    selected_topics = selected_topics.copy()
    selected_topics["r_indicator_text"] = r_texts
    selected_topics["d_indicator_text"] = d_texts
    selected_topics["r_speech_id"] = r_ids
    selected_topics["d_speech_id"] = d_ids

    # Stats
    n_both = sum(1 for r, d in zip(r_texts, d_texts) if r and d)
    avg_r_len = np.mean([len(t.split()) for t in r_texts if t])
    avg_d_len = np.mean([len(t.split()) for t in d_texts if t])
    print(f"  Indicators extracted: {n_both}/{len(selected_topics)} topics have both R+D")
    print(f"    Avg R indicator: {avg_r_len:.0f} words, Avg D indicator: {avg_d_len:.0f} words")

    return selected_topics


# ── Main ────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PRISM Step 1: Topic Mining with GST Filtering + Labeling")
    print("=" * 60)

    # Load GST procedural bigrams (once)
    print("\nPhase A: Loading GST data")
    procedural_set = load_gst_procedural()

    # Load speech texts (once, filter to relevant congresses)
    print("\nLoading speech texts...")
    all_speeches = pd.read_parquet(SPEECHES_PATH)
    # Congress in speeches is string like '099'
    relevant_congs = set()
    for prev, curr in get_windows():
        relevant_congs.add(f"{prev:03d}")
        relevant_congs.add(f"{curr:03d}")
    speech_texts = all_speeches[all_speeches["congress"].isin(relevant_congs)].copy()
    print(f"  Loaded {len(speech_texts):,} speeches for congresses {sorted(relevant_congs)}")

    for window in get_windows():
        prev_cong, curr_cong = window
        print(f"\n{'=' * 60}")
        print(f"Window ({prev_cong}, {curr_cong})")
        print("=" * 60)

        # ── Load SBERT speech embeddings + meta ──────────────────
        embs, metas = [], []
        for c in [prev_cong, curr_cong]:
            emb_path = SBERT_DIR / f"speech_embeddings_cong_{c}.npz"
            meta_path = SBERT_DIR / f"speech_meta_cong_{c}.parquet"
            if not emb_path.exists():
                print(f"  Congress {c}: no embeddings found")
                continue
            e = np.load(emb_path)["embeddings"]
            m = pd.read_parquet(meta_path)
            embs.append(e)
            metas.append(m)
            print(f"  Congress {c}: {len(m):,} speeches")

        if not embs:
            print("  Skipping window — no data")
            continue

        all_emb = np.vstack(embs)
        all_meta = pd.concat(metas, ignore_index=True)

        # ── Partisan core filter ─────────────────────────────────
        if PARTISAN_CORE_ONLY:
            core_mask = (all_meta["label_rep_core"] == 1) | (all_meta["label_dem_core"] == 1)
            all_emb = all_emb[core_mask.values]
            all_meta = all_meta[core_mask].reset_index(drop=True)
            print(f"  Partisan core: {len(all_meta):,} speeches")

        # ── Phase B: GST procedural filtering ────────────────────
        print("\nPhase B: Speech filtering")
        # Get texts for these speeches
        window_speech_ids = set(all_meta["speech_id"].values)
        window_texts = speech_texts[speech_texts["speech_id"].isin(window_speech_ids)]
        print(f"  Matched texts: {len(window_texts):,} / {len(all_meta):,}")

        keep_mask = filter_speeches(all_meta, window_texts, procedural_set)
        filtered_emb = all_emb[keep_mask]
        filtered_meta = all_meta[keep_mask].reset_index(drop=True)
        print(f"  After filtering: {len(filtered_meta):,} speeches")

        # ── Phase C: K-means clustering ──────────────────────────
        print("\nPhase C: K-means clustering")
        cluster_labels, centroids = cluster_speeches(filtered_emb, N_CLUSTERS)

        # ── Phase D: Select polarized topics + GST labeling ──────
        print("\nPhase D: Topic selection + GST labeling")
        selected = select_polarized_topics(cluster_labels, filtered_meta)

        # Load partisan phrases for this window's congresses
        partisan_phrases = load_gst_partisan_phrases([prev_cong, curr_cong])

        selected = label_topics_with_gst(
            selected, cluster_labels, filtered_meta,
            window_texts, partisan_phrases,
        )

        # ── Phase E: Extract indicators ──────────────────────────
        print("\nPhase E: Extracting indicator speeches")
        selected = extract_indicators(
            selected, cluster_labels, filtered_emb,
            filtered_meta, window_texts, centroids,
        )

        # ── Save ─────────────────────────────────────────────────
        # Topics parquet
        out_path = OUTPUT_DIR / f"topics_window_{prev_cong}_{curr_cong}.parquet"
        selected.to_parquet(out_path, index=False)
        print(f"\n  Saved: {out_path.name} ({len(selected)} topics)")

        # Centroids for topic retrieval (all 200 clusters, for bi-encoder search)
        centroid_path = OUTPUT_DIR / f"topic_centroids_window_{prev_cong}_{curr_cong}.npy"
        np.save(centroid_path, centroids.astype(np.float32))
        print(f"  Saved: {centroid_path.name} ({centroids.shape})")

        # Print sample indicators
        print(f"\n  Sample topic indicators:")
        for _, t in selected.head(3).iterrows():
            print(f"\n    Topic {t['topic_idx']}: {t['topic_label']}")
            print(f"      R-leaning: {t['r_proportion']:.2f}, N={t['n_speeches']}")
            r_preview = t['r_indicator_text'][:120] + "..." if len(t['r_indicator_text']) > 120 else t['r_indicator_text']
            d_preview = t['d_indicator_text'][:120] + "..." if len(t['d_indicator_text']) > 120 else t['d_indicator_text']
            print(f"      R indicator: {r_preview}")
            print(f"      D indicator: {d_preview}")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Topic mining complete in {elapsed:.0f}s")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
