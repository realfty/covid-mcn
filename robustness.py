
# -*- coding: utf-8 -*-
"""
robustness.py
-------------
Run robustness checks for module control networks across multiple phases.

What it does
1) N-bootstrap (with replacement, same sample size as original) per phase.
2) Case-dropping stability (retain ratios: 0.9, 0.8, 0.7, 0.6, 0.5), each repeated many times.
3) Correlates each replicate's metrics (Node strength, ACF, AMCS) with the full-sample metrics.
4) Saves per-replicate results and summary JSONs under a results directory.

Assumptions
- CSV files for each phase are in `DATA_PATH` (same format as used by module_control.py).
- Graph construction follows your main analysis:
    GraphicalLasso(alpha=ALPHA) → precision → partial correlation → abs() & diag→0 → weighted undirected graph.
- Module/Control metrics reuse functions from tools.py (already in your project).

Usage
    python robustness.py

Edit the CONFIG section below if paths or parameters differ.
"""

import os
import json
import math
import time
import random
import datetime
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import networkx as nx
import community.community_louvain as louvain

from sklearn.covariance import GraphicalLasso
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr

# --- import project helpers ---
from tools import (
    all_min_dominating_set,
    dominating_frequency,
    matrix_preprocess,
    module_controllability,
    network_analysis
)

# =====================
# ====== CONFIG =======
# =====================
DATA_PATH   = r"/home/a001/fantianyi/globalMindProject/US_18-24_data/filter_columns"  # same as module_control.py
RESULT_PATH = r"/home/a001/fantianyi/globalMindProject/鲁棒性/重复抽样"
ALPHA       = 0.4                 # keep consistent with your main result
N_BOOT      = 1000                # bootstrap repeats per phase
N_DROP      = 1000                # case-dropping repeats per fraction per phase
DROP_RETAIN = [0.9, 0.8, 0.7, 0.6, 0.5]  # retain ratios (case-dropping)
SEED        = 42                  # reproducibility for resampling

# If exact MDSet enumeration is too slow, you can switch to "greedy" (not recommended for main results).
MDS_METHOD  = "precise"   # "precise" (all_min_dominating_set) or "greedy" (fallback not wired by default here)

# =====================
# ===== UTILITIES =====
# =====================

def log(msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} | {msg}")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def list_phase_files(data_path):
    return sorted([f[:-4] for f in os.listdir(data_path) if f.endswith(".csv") and "nan" not in f])

def build_graph_from_dataframe(df: pd.DataFrame, alpha: float) -> nx.Graph:
    """Fit GraphicalLasso, compute partial correlations, preprocess and return weighted undirected graph."""
    est = GraphicalLasso(alpha=alpha)
    est.fit(df.values)

    prec = est.precision_
    # partial correlation
    diag_sqrt = np.sqrt(np.diag(prec))
    pcorr = -prec / np.outer(diag_sqrt, diag_sqrt)
    np.fill_diagonal(pcorr, 1.0)

    # abs + zero diag (as in your pipeline)
    W = matrix_preprocess(pcorr)
    G = nx.Graph(W)
    return G

def compute_acf(cf_counts: dict, communities: dict) -> dict:
    """Average Control Frequency per module (normalize CF by #MDSets)."""
    if len(cf_counts) == 0:
        return {}
    # cf_counts are integer counts across all minimal dominating sets
    # normalize by max possible count (= number of MDSets)
    max_count = max(cf_counts.values()) if cf_counts else 1
    if max_count == 0:  # fallback
        max_count = 1
    cf_norm = {n: cf_counts.get(n, 0) / max_count for n in communities}
    # group by module
    mod_ids = set(communities.values())
    acf = {}
    for m in mod_ids:
        nodes_in_m = [n for n, lab in communities.items() if lab == m]
        if len(nodes_in_m) == 0:
            acf[m] = 0.0
        else:
            acf[m] = float(np.mean([cf_norm[n] for n in nodes_in_m]))
    return acf

def relabel_modules_to_full(full_labels: dict, boot_labels: dict) -> dict:
    """
    Align community labels of a replicate to the full-sample labels using Hungarian matching on the overlap matrix.
    Returns a mapping: boot_label -> full_label
    """
    full_ids = sorted(set(full_labels.values()))
    boot_ids = sorted(set(boot_labels.values()))
    nF = len(full_ids)
    nB = len(boot_ids)

    # Build overlap matrix (nF x nB): count of nodes overlapping in (full=i, boot=j)
    idxF = {lab:i for i, lab in enumerate(full_ids)}
    idxB = {lab:j for j, lab in enumerate(boot_ids)}

    M = np.zeros((nF, nB), dtype=int)
    for node in full_labels.keys():
        i = idxF[ full_labels[node] ]
        j = idxB[ boot_labels[node] ]
        M[i, j] += 1

    # Hungarian to maximize total overlap -> minimize negative overlap
    cost = -M
    row_ind, col_ind = linear_sum_assignment(cost)

    mapping = {}
    for i, j in zip(row_ind, col_ind):
        mapping[ boot_ids[j] ] = full_ids[i]

    # Unmatched boot labels (if any) -> map to nearest full by overlap
    unmatched = [b for b in boot_ids if b not in mapping]
    for b in unmatched:
        col = M[:, idxB[b]]
        mapping[b] = full_ids[int(np.argmax(col))]

    return mapping

def remap_acf_to_full(acf_boot: dict, boot_to_full_map: dict) -> dict:
    return { boot_to_full_map.get(k, k): v for k, v in acf_boot.items() }

def remap_amcs_to_full(amcs_boot: dict, boot_to_full_map: dict, n_modules_full: int) -> np.ndarray:
    """
    Convert AMCS dict like {'0_1': val, ...} using boot_to_full mapping onto full labels,
    then return a dense (n_modules_full x n_modules_full) matrix in full label order 0..K-1.
    """
    # parse boot dict
    temp = {}
    for key, val in amcs_boot.items():
        a, b = key.split('_')
        a, b = int(a), int(b)
        a2 = boot_to_full_map.get(a, a)
        b2 = boot_to_full_map.get(b, b)
        temp[(a2, b2)] = val

    M = np.zeros((n_modules_full, n_modules_full), dtype=float)
    for (i, j), v in temp.items():
        if 0 <= i < n_modules_full and 0 <= j < n_modules_full:
            M[i, j] = v
    return M

def vectorize_matrix_row_major(M: np.ndarray) -> np.ndarray:
    return M.flatten(order='C')

def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) != len(y) or len(x) == 0:
        return np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    r, _ = pearsonr(x, y)
    return float(r)

def summarize_ci(series: pd.Series, alpha=0.05):
    lo = series.quantile(alpha/2.0)
    hi = series.quantile(1 - alpha/2.0)
    return float(series.mean()), float(lo), float(hi)

# ===========================
# ===== CORE PIPELINES ======
# ===========================

def one_pass_metrics(G: nx.Graph):
    """
    Compute all metrics needed for correlation tracking against full-sample.
    Returns:
        dict with keys: edges, modules, mdset_size, mdset_count, strength (dict),
                        CF (dict), ACF (dict), AMCS (dict), communities (dict)
    """
    # community detection
    comm = louvain.best_partition(G)
    n_modules = max(comm.values()) + 1 if len(comm) else 0

    # MDSets
    if MDS_METHOD == "precise":
        all_dom_set, _ = all_min_dominating_set(G)
    else:
        # You can wire your greedy method here if needed.
        all_dom_set, _ = all_min_dominating_set(G)

    mdset_count = len(all_dom_set)
    mdset_size  = len(all_dom_set[0]) if mdset_count > 0 else 0

    # CF and controllability
    cf_counts = dominating_frequency(all_dom_set, G) if mdset_count > 0 else {n:0 for n in G.nodes}
    amcs = module_controllability(G, all_dom_set, comm) if mdset_count > 0 else {}

    # Node strength & co.
    net_stats = network_analysis(G)
    strength = net_stats["average_strength"]  # dict by node

    acf = compute_acf(cf_counts, comm)

    return dict(
        edges=int(G.number_of_edges()),
        modules=int(n_modules),
        mdset_size=int(mdset_size),
        mdset_count=int(mdset_count),
        strength=strength,
        CF=cf_counts,
        ACF=acf,
        AMCS=amcs,
        communities=comm
    )

def bootstrap_phase(df: pd.DataFrame, alpha: float, n_boot: int, rng: random.Random):
    """Return dataframe of per-replicate metrics and correlations vs full sample."""
    n = len(df)
    # full-sample baseline
    G_full = build_graph_from_dataframe(df, alpha)
    base = one_pass_metrics(G_full)

    # Prepare vectors for correlation
    nodes_order = sorted(G_full.nodes())
    strength_full = np.array([base["strength"][i] for i in nodes_order])
    acf_full_dict = base["ACF"]
    n_modules_full = base["modules"]
    # For AMCS, build dense matrix (K x K)
    M_full = np.zeros((n_modules_full, n_modules_full), dtype=float)
    for k, v in base["AMCS"].items():
        a, b = map(int, k.split('_'))
        if a < n_modules_full and b < n_modules_full:
            M_full[a, b] = v
    amcs_full_vec = vectorize_matrix_row_major(M_full)

    rows = []
    for rep in range(1, n_boot + 1):
        # resample with replacement
        idx = rng.choices(range(n), k=n)
        df_b = df.iloc[idx].reset_index(drop=True)
        G_b  = build_graph_from_dataframe(df_b, alpha)
        m    = one_pass_metrics(G_b)

        # correlations
        strength_b = np.array([m["strength"].get(i, 0.0) for i in nodes_order])
        # align modules
        boot_to_full = relabel_modules_to_full(base["communities"], m["communities"])
        acf_b = remap_acf_to_full(m["ACF"], boot_to_full)
        acf_b_vec = np.array([acf_b.get(i, 0.0) for i in sorted(acf_full_dict.keys())])

        M_b = remap_amcs_to_full(m["AMCS"], boot_to_full, n_modules_full)
        amcs_b_vec = vectorize_matrix_row_major(M_b)

        r_strength = safe_pearson(strength_full, strength_b)
        r_acf      = safe_pearson(np.array([acf_full_dict[i] for i in sorted(acf_full_dict.keys())]), acf_b_vec)
        r_amcs     = safe_pearson(amcs_full_vec, amcs_b_vec)

        rows.append(dict(
            rep=rep,
            edges=m["edges"],
            modules=m["modules"],
            mdset_size=m["mdset_size"],
            mdset_count=m["mdset_count"],
            r_strength=r_strength,
            r_acf=r_acf,
            r_amcs=r_amcs
        ))

    return base, pd.DataFrame(rows)

def case_dropping_phase(df: pd.DataFrame, alpha: float, retain_levels, n_rep: int, rng: random.Random):
    """
    For each retain ratio, build networks n_rep times and correlate metrics with full-sample metrics.
    Return: dict retain_ratio -> DataFrame of correlations
    """
    n = len(df)
    # full baseline
    G_full = build_graph_from_dataframe(df, alpha)
    base = one_pass_metrics(G_full)

    nodes_order = sorted(G_full.nodes())
    strength_full = np.array([base["strength"][i] for i in nodes_order])
    acf_full_dict = base["ACF"]
    n_modules_full = base["modules"]
    M_full = np.zeros((n_modules_full, n_modules_full), dtype=float)
    for k, v in base["AMCS"].items():
        a, b = map(int, k.split('_'))
        if a < n_modules_full and b < n_modules_full:
            M_full[a, b] = v
    amcs_full_vec = vectorize_matrix_row_major(M_full)

    out = {}
    for retain in retain_levels:
        k = int(math.floor(retain * n))
        rows = []
        for rep in range(1, n_rep + 1):
            # sample without replacement
            idx = rng.sample(range(n), k=k)
            df_c = df.iloc[idx].reset_index(drop=True)
            G_c  = build_graph_from_dataframe(df_c, alpha)
            m    = one_pass_metrics(G_c)

            strength_c = np.array([m["strength"].get(i, 0.0) for i in nodes_order])
            boot_to_full = relabel_modules_to_full(base["communities"], m["communities"])
            acf_c = remap_acf_to_full(m["ACF"], boot_to_full)
            acf_c_vec = np.array([acf_c.get(i, 0.0) for i in sorted(acf_full_dict.keys())])

            M_c = remap_amcs_to_full(m["AMCS"], boot_to_full, n_modules_full)
            amcs_c_vec = vectorize_matrix_row_major(M_c)

            r_strength = safe_pearson(strength_full, strength_c)
            r_acf      = safe_pearson(np.array([acf_full_dict[i] for i in sorted(acf_full_dict.keys())]), acf_c_vec)
            r_amcs     = safe_pearson(amcs_full_vec, amcs_c_vec)

            rows.append(dict(
                rep=rep,
                retain=retain,
                r_strength=r_strength,
                r_acf=r_acf,
                r_amcs=r_amcs
            ))
        out[retain] = pd.DataFrame(rows)
    return base, out

def cs_coefficient(corr_df_by_retain, threshold=0.7, quantile=0.05):
    """
    Compute CS-coefficient: the largest proportion of cases that can be dropped
    while the correlation with the full-sample metrics remains >= threshold
    for at least (1-quantile) of replications.
    Returns dict for strength/acf/amcs.
    """
    # corr_df_by_retain: dict retain -> df with columns r_strength, r_acf, r_amcs
    # We need to find max drop (1 - retain) satisfying quantile condition.
    metrics = ["r_strength", "r_acf", "r_amcs"]
    result = {}
    for m in metrics:
        best_drop = 0.0
        for retain, df in sorted(corr_df_by_retain.items()):
            q = df[m].quantile(quantile)  # e.g., 5th percentile
            if q >= threshold:
                best_drop = max(best_drop, 1.0 - retain)
        result[m] = best_drop
    return result

# ===========================
# =========  MAIN  ==========
# ===========================

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    ensure_dir(RESULT_PATH)
    phases = list_phase_files(DATA_PATH)
    log(f"Found {len(phases)} phase files.")

    for pi, phase in enumerate(phases, 1):
        log(f"[{pi}/{len(phases)}] Phase: {phase}")
        phase_dir = ensure_dir(os.path.join(RESULT_PATH, phase))
        boot_dir  = ensure_dir(os.path.join(phase_dir, "bootstrap"))
        cd_dir    = ensure_dir(os.path.join(phase_dir, "case_dropping"))

        df = pd.read_csv(os.path.join(DATA_PATH, f"{phase}.csv"))

        # ---- Bootstrap ----
        t0 = time.time()
        base_boot, boot_df = bootstrap_phase(df, ALPHA, N_BOOT, random)
        boot_df.to_csv(os.path.join(boot_dir, "bootstrap_metrics.csv"), index=False, encoding="utf-8")

        # summarize bootstrap (edges/modules/mdset/ correlations)
        summary = {
            "edges_mean_ci": summarize_ci(boot_df["edges"]),
            "modules_mean_ci": summarize_ci(boot_df["modules"]),
            "mdset_size_mean_ci": summarize_ci(boot_df["mdset_size"]),
            "mdset_count_mean_ci": summarize_ci(boot_df["mdset_count"]),
            "r_strength_mean_ci": summarize_ci(boot_df["r_strength"].dropna()),
            "r_acf_mean_ci": summarize_ci(boot_df["r_acf"].dropna()),
            "r_amcs_mean_ci": summarize_ci(boot_df["r_amcs"].dropna()),
        }
        with open(os.path.join(boot_dir, "bootstrap_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        log(f"Bootstrap done in {time.time() - t0:.2f}s")

        # ---- Case-dropping ----
        t1 = time.time()
        base_cd, cd_dict = case_dropping_phase(df, ALPHA, DROP_RETAIN, N_DROP, random)
        # write each retain level
        for retain, dfx in cd_dict.items():
            dfx.to_csv(os.path.join(cd_dir, f"case_drop_retain_{retain:.2f}.csv"), index=False, encoding="utf-8")
        # CS coefficients (0.7 @ 95% default)
        cs = cs_coefficient(cd_dict, threshold=0.7, quantile=0.05)
        with open(os.path.join(cd_dir, "cs_coefficients.json"), "w", encoding="utf-8") as f:
            json.dump(cs, f, ensure_ascii=False, indent=2)

        log(f"Case-dropping done in {time.time() - t1:.2f}s")

        # ---- Save full-sample basics for reference ----
        base_out = {
            "edges": base_boot["edges"],
            "modules": base_boot["modules"],
            "mdset_size": base_boot["mdset_size"],
            "mdset_count": base_boot["mdset_count"]
        }
        with open(os.path.join(phase_dir, "full_sample_summary.json"), "w", encoding="utf-8") as f:
            json.dump(base_out, f, ensure_ascii=False, indent=2)

    log("All phases completed.")

if __name__ == "__main__":
    main()
