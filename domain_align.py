#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
domain_align.py  (minimal, static mapping + ACF aggregation)

Two subcommands:
  1) validate        : check & normalize a static node→domain mapping CSV.
  2) aggregate-acf   : use the mapping to aggregate node-level CF to domain-level ACF per phase.

We intentionally do NOT read ACF/AMCS/MCN to avoid circularity.
Later we can add an 'auto' mode (anchor + spectral/k-medoids on P matrix) as a separate PR.

Expected inputs
---------------
mapping CSV: at least columns: ['node_id','domain'].
             Optional: ['node_name','source','confidence'].
             Domains must be in {'EMO','STR','SPF','CSF'}.

CF files: under --cf-dir/<phase_name>/cf.csv, with columns including 'node_id' and one of:
          - 'cf_norm' (preferred, [0,1])
          - else 'cf' or 'CF' or 'cf_count' (will be normalized per phase to [0,1] by dividing by per-phase max).

Outputs
-------
- <outdir>/node2domain.normalized.csv
- <outdir>/domain_acf.csv  with columns: ['phase','domain','acf_mean','n_nodes','coverage']

Usage
-----
python domain_align.py validate --mapping examples/domain_alignment/node2domain.csv --outdir out/
python domain_align.py aggregate-acf --mapping out/node2domain.normalized.csv --cf-dir /path/to/results --outdir out/
"""

from __future__ import annotations
import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ALLOWED_DOMAINS = {"EMO", "STR", "SPF", "CSF"}

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def load_and_normalize_mapping(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = [c.strip() for c in df.columns]
    df.columns = cols
    # minimal check
    required = {"node_id", "domain"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"mapping must contain columns {required}, got {df.columns.tolist()}")

    # normalize domain to upper
    df["domain"] = df["domain"].astype(str).str.strip().str.upper()
    bad = sorted(set(df["domain"]) - ALLOWED_DOMAINS)
    if bad:
        raise ValueError(f"Invalid domain values found: {bad}. Allowed: {sorted(ALLOWED_DOMAINS)}")

    # normalize node_id to int
    try:
        df["node_id"] = df["node_id"].astype(int)
    except Exception:
        raise ValueError("column 'node_id' must be integer-like")

    # drop duplicates (keep first)
    before = len(df)
    df = df.drop_duplicates(subset=["node_id"], keep="first").copy()
    if len(df) != before:
        eprint(f"[warn] removed {before - len(df)} duplicated node_id rows in mapping")

    # sort by node_id
    df = df.sort_values("node_id").reset_index(drop=True)
    return df

def cmd_validate(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = load_and_normalize_mapping(Path(args.mapping))

    # small report
    counts = df["domain"].value_counts().reindex(sorted(ALLOWED_DOMAINS), fill_value=0)
    report = {
        "n_nodes": int(len(df)),
        "domain_counts": counts.to_dict(),
    }
    # save normalized mapping & report
    norm_path = outdir / "node2domain.normalized.csv"
    df.to_csv(norm_path, index=False)
    (outdir / "mapping_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[ok] normalized mapping saved -> {norm_path}")
    print(f"[ok] domain counts: {report['domain_counts']} (n_nodes={report['n_nodes']})")

def find_phase_dirs(cf_root: Path) -> List[Path]:
    # expect subdirs like phase_Early_Phase_integrated/, etc., each containing cf.csv
    cands = []
    for p in sorted(cf_root.iterdir()):
        if p.is_dir() and (p / "cf.csv").exists():
            cands.append(p)
    if not cands:
        raise FileNotFoundError(f"No <phase>/cf.csv found under {cf_root}")
    return cands

def load_cf_normalized(cf_path: Path) -> pd.DataFrame:
    df = pd.read_csv(cf_path)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    # find node id column
    nid_col = None
    for c in ("node_id", "node", "id", "nodeId"):
        if c in df.columns:
            nid_col = c
            break
    if nid_col is None:
        raise ValueError(f"{cf_path}: missing node_id column (expected one of ['node_id','node','id','nodeId'])")
    df = df[[nid_col] + [c for c in df.columns if c != nid_col]].copy()
    df.rename(columns={nid_col: "node_id"}, inplace=True)
    df["node_id"] = df["node_id"].astype(int)

    # find CF-like column
    cf_norm_col = None
    if "cf_norm" in df.columns:
        cf_norm_col = "cf_norm"
        df["cf_norm"] = df["cf_norm"].astype(float)
    else:
        # fallback to raw counts; normalize by per-phase max
        for c in ("cf", "CF", "cf_count", "count"):
            if c in df.columns:
                cf_norm_col = c
                break
        if cf_norm_col is None:
            raise ValueError(f"{cf_path}: cannot find CF column (tried cf_norm/cf/CF/cf_count/count)")
        # normalize
        m = df[cf_norm_col].astype(float).max()
        if m <= 0:
            df["cf_norm"] = 0.0
        else:
            df["cf_norm"] = df[cf_norm_col].astype(float) / float(m)

    return df[["node_id", "cf_norm"]].copy()

def cmd_aggregate_acf(args: argparse.Namespace) -> None:
    mapping = load_and_normalize_mapping(Path(args.mapping)).set_index("node_id")["domain"].to_dict()
    cf_root = Path(args.cf_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ph_dir in find_phase_dirs(cf_root):
        phase_name = ph_dir.name
        cf_path = ph_dir / "cf.csv"
        df = load_cf_normalized(cf_path)

        # join mapping
        df["domain"] = df["node_id"].map(mapping)
        missing = df["domain"].isna().sum()
        if missing:
            eprint(f"[warn] {phase_name}: {missing} nodes in CF not found in mapping; they will be dropped.")
        df = df.dropna(subset=["domain"]).copy()

        # aggregate to domain-level ACF (mean of cf_norm)
        g = df.groupby("domain")["cf_norm"].agg(["mean", "count"])
        g = g.reindex(sorted(ALLOWED_DOMAINS)).fillna({"mean": np.nan, "count": 0})
        # coverage = matched nodes / expected nodes in mapping for that domain
        dom_expected = pd.Series(mapping).value_counts()
        coverage = (g["count"] / dom_expected.reindex(g.index).fillna(0)).replace([np.inf, -np.inf], np.nan)

        for dom in g.index:
            rows.append({
                "phase": phase_name,
                "domain": dom,
                "acf_mean": float(g.loc[dom, "mean"]) if not np.isnan(g.loc[dom, "mean"]) else np.nan,
                "n_nodes": int(g.loc[dom, "count"]),
                "coverage": float(coverage.loc[dom]) if not np.isnan(coverage.loc[dom]) else 0.0
            })

    out_df = pd.DataFrame(rows).sort_values(["phase", "domain"]).reset_index(drop=True)
    out_path = outdir / "domain_acf.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[ok] domain-level ACF saved -> {out_path}")
    # tiny preview
    print(out_df.head(10).to_string(index=False))

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Static domain mapping & ACF aggregation (minimal).")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_val = sub.add_parser("validate", help="Validate & normalize node→domain mapping CSV")
    p_val.add_argument("--mapping", required=True, help="Path to node2domain.csv")
    p_val.add_argument("--outdir",  required=True, help="Output directory for normalized mapping/report")
    p_val.set_defaults(func=cmd_validate)

    p_acf = sub.add_parser("aggregate-acf", help="Aggregate node-level CF to domain-level ACF per phase")
    p_acf.add_argument("--mapping", required=True, help="Path to normalized node2domain.csv")
    p_acf.add_argument("--cf-dir",  required=True, help="Root dir with <phase>/cf.csv produced by module_control")
    p_acf.add_argument("--outdir",  required=True, help="Output directory for domain_acf.csv")
    p_acf.set_defaults(func=cmd_aggregate_acf)

    return p

def main():
    parser = build_argparser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()