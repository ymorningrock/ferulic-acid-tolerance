import os
import re
import ast
from pathlib import Path
import numpy as np
import pandas as pd

import cobra
from cobra.flux_analysis import pfba


ROOT = Path(__file__).resolve().parents[1]
SBML_PATH = ROOT / "models" / "iMT1026_glyc.xml"
TPM_WT_PATH = ROOT / "data" / "TPM_WT_24h.csv"
TPM_FENI_PATH = ROOT / "data" / "TPM_FE-NI_24h.csv"
DEG_CSV = ROOT / "data" / "All Data_24h_FE_vs_WT.csv"
OUT_DIR = ROOT / "results"

TOPN_RXN = 30
TOPN_GENE = 30
MIN_SCALE = 1e-6
USE_ROBUST_MAX = True
ROBUST_Q = 0.99
MAX_SCALE = 10.0
PFBA_RELAX_FACTORS = [0.999, 0.99, 0.95]
DELTA_V_EPS = 1e-9
ATTRIB_TOL_ABS = 1e-8
ATTRIB_TOL_REL = 1e-8
MAX_DNF_CLAUSES = 256
ENABLE_PSEUDOGENE_BUCKET = True
PSEUDO_PREFIX = "__"
WRITE_EFLUX_SCALING_TSV = False
REP_AGG = "median"


D_COL_GENE = "gene_symbol"
D_COL_NAME = "gene_name"
D_COL_LOG2FC = "log2fc"
D_COL_QVALUE = "qvalue"


def mkdir(p: str):
    os.makedirs(p, exist_ok=True)

def to_num(x):
    return pd.to_numeric(x, errors="coerce")

def agg_reps(arr: np.ndarray, method: str):
    if method == "median":
        return float(np.nanmedian(arr))
    if method == "mean":
        return float(np.nanmean(arr))
    raise ValueError("REP_AGG must be 'median' or 'mean'")


def read_tpm_noheader(path: str, label: str):
    df = pd.read_csv(path, header=None)
    gene = df.iloc[:, 0].astype(str)
    reps = df.iloc[:, 1:].apply(to_num).fillna(0.0)

    expr = []
    for i in range(reps.shape[0]):
        expr.append(agg_reps(reps.iloc[i].values.astype(float), REP_AGG))

    out = pd.DataFrame({"gene": gene.values, f"expr_{label}": np.array(expr, dtype=float)})
    return out


def safe_gene_rule_to_reaction_expr(rule: str, gene_ids, gene2expr):
    if not rule or not rule.strip() or not gene_ids:
        return None

    genes_sorted = sorted(set(gene_ids), key=len, reverse=True)
    rep = {}
    expr = rule

    for i, gid in enumerate(genes_sorted):
        ph = f"g{i}"
        rep[ph] = gid
        expr = re.sub(rf"(?<!\w){re.escape(gid)}(?!\w)", ph, expr)

    expr = expr.replace("AND", "and").replace("OR", "or")

    try:
        node = ast.parse(expr, mode="eval").body
    except Exception:
        return None

    def eval_node(n):
        if isinstance(n, ast.Name):
            gid = rep.get(n.id, None)
            if gid is None:
                return None
            v = gene2expr.get(gid, None)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            return float(v)

        if isinstance(n, ast.BoolOp):
            vals = [eval_node(x) for x in n.values]
            if any(v is None for v in vals):
                return None
            if isinstance(n.op, ast.And):
                return float(np.min(vals))
            if isinstance(n.op, ast.Or):
                return float(np.max(vals))
            return None

        return None

    return eval_node(node)

def compute_reaction_expression(model: cobra.Model, gene2expr: dict) -> dict:
    rxn2expr = {}
    for rxn in model.reactions:
        if not rxn.genes:
            continue
        rule = rxn.gene_reaction_rule
        if not rule or not rule.strip():
            continue
        gids = [g.id for g in rxn.genes]
        v = safe_gene_rule_to_reaction_expr(rule, gids, gene2expr)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        rxn2expr[rxn.id] = float(v)
    return rxn2expr


def apply_eflux(model: cobra.Model, rxn2expr: dict, mode_name: str, out_dir: str = None) -> cobra.Model:
    vals = np.array(list(rxn2expr.values()), dtype=float)
    vals = vals[np.isfinite(vals)]

    denom = float(np.max(vals))
    if USE_ROBUST_MAX:
        denom = float(np.quantile(vals, ROBUST_Q))
        if denom <= 0:
            denom = float(np.max(vals))

    m = model.copy()
    scales = []
    minscale_hits = 0
    cap_hits = 0
    changed = 0

    for rxn in m.reactions:
        if rxn.id not in rxn2expr:
            continue

        raw_scale = rxn2expr[rxn.id] / denom
        scale = max(raw_scale, MIN_SCALE)
        if scale == MIN_SCALE and raw_scale < MIN_SCALE:
            minscale_hits += 1
        if scale > MAX_SCALE:
            scale = MAX_SCALE
            cap_hits += 1

        rxn.lower_bound *= scale
        rxn.upper_bound *= scale
        changed += 1
        scales.append((rxn.id, raw_scale, scale))

    arr = np.array([s[2] for s in scales], dtype=float)
    print(f"[E-Flux] {mode_name}: changed={changed}, denom={'q'+str(int(ROBUST_Q*100)) if USE_ROBUST_MAX else 'max'}={denom:.4g}, "
          f"scale(min/median/max)={arr.min():.3g}/{np.median(arr):.3g}/{arr.max():.3g}, "
          f"MIN_SCALE_hits={minscale_hits}, MAX_SCALE_hits={cap_hits}")

    if out_dir and WRITE_EFLUX_SCALING_TSV and scales:
        df_scale = pd.DataFrame(scales, columns=["reaction", "raw_scale", "applied_scale"])
        df_scale.to_csv(os.path.join(out_dir, f"eflux_scaling_{mode_name}.tsv"), sep="\t", index=False)

    return m


def solve_pfba_common(model: cobra.Model, mode_name: str, obj_common: float) -> cobra.Solution:
    sol = model.optimize()
    if sol.status != "optimal":
        raise ValueError(f"[{mode_name}] FBA non-optimal: {sol.status}")
    obj_star = float(sol.objective_value)

    if obj_star == 0 or obj_common <= 0:
        for frac in PFBA_RELAX_FACTORS:
            psol = pfba(model, fraction_of_optimum=frac)
            if psol.status == "optimal":
                print(f"[{mode_name}] pFBA optimal with fraction_of_optimum={frac}")
                return psol
        return sol

    base_frac = obj_common / obj_star
    if base_frac > 1:
        base_frac = 1.0
    if base_frac < 0:
        base_frac = 0.0

    print(f"[{mode_name}] obj*={obj_star:.6g}, obj_common={obj_common:.6g}, base_frac={base_frac:.6g}")

    for relax in PFBA_RELAX_FACTORS:
        frac = base_frac * relax
        psol = pfba(model, fraction_of_optimum=frac)
        if psol.status == "optimal":
            return psol

    return sol


def gpr_to_clauses(rule: str, gene_ids):
    if not rule or not rule.strip() or not gene_ids:
        return []

    genes_sorted = sorted(set(gene_ids), key=len, reverse=True)
    rep = {}
    expr = rule
    for i, gid in enumerate(genes_sorted):
        ph = f"g{i}"
        rep[ph] = gid
        expr = re.sub(rf"(?<!\w){re.escape(gid)}(?!\w)", ph, expr)

    expr = expr.replace("AND", "and").replace("OR", "or")
    try:
        node = ast.parse(expr, mode="eval").body
    except Exception:
        return []

    def to_dnf(n):
        if isinstance(n, ast.Name):
            gid = rep.get(n.id, None)
            return [set([gid])] if gid else []

        if isinstance(n, ast.BoolOp):
            child_dnfs = [to_dnf(x) for x in n.values]
            if any(len(x) == 0 for x in child_dnfs):
                return []

            if isinstance(n.op, ast.Or):
                out = []
                for part in child_dnfs:
                    out.extend(part)
                return out

            if isinstance(n.op, ast.And):
                out = [set()]
                for part in child_dnfs:
                    new_out = []
                    for a in out:
                        for b in part:
                            new_out.append(a.union(b))
                    out = new_out
                    if len(out) > MAX_DNF_CLAUSES:
                        return ["__TOO_MANY__"]
                return out

        return []

    clauses = to_dnf(node)
    if clauses == ["__TOO_MANY__"]:
        return ["__TOO_MANY__"]

    uniq = []
    seen = set()
    for c in clauses:
        c = frozenset([x for x in c if x])
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def dominant_clause_and_bottleneck(clauses, gene2expr: dict):
    best_clause = None
    best_clause_expr = None

    for c in clauses:
        vals = []
        ok = True
        for g in c:
            v = gene2expr.get(g, None)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                ok = False
                break
            vals.append(float(v))
        if (not ok) or (len(vals) == 0):
            continue

        clause_expr = float(np.min(vals))  
        if (best_clause is None) or (clause_expr > best_clause_expr):
            best_clause = c
            best_clause_expr = clause_expr
        elif clause_expr == best_clause_expr and best_clause is not None:
            if len(c) < len(best_clause):
                best_clause = c

    if best_clause is None:
        return (None, None, None)

    min_v = min(float(gene2expr[g]) for g in best_clause)
    bottlenecks = [g for g in best_clause if float(gene2expr[g]) == min_v]
    bottleneck_gene = sorted(bottlenecks)[0] if bottlenecks else None
    return (bottleneck_gene, best_clause_expr, best_clause)


def build_gene_contrib_from_deltaflux(model, rxn_df: pd.DataFrame, gene2wt: dict, gene2feni: dict):
    contrib_rows = []
    bottleneck_rows = []
    skipped_rows = []
    status_rows = []

    stats = {
        "skipped_no_gpr": 0,
        "skipped_too_complex": 0,
        "skipped_no_clause": 0,
        "skipped_no_bottleneck": 0,
        "skipped_rxn_not_in_model": 0,
    }

    def add_pseudo(rid, dv, reason):
        if not ENABLE_PSEUDOGENE_BUCKET:
            return
        pseudo_gene = f"{PSEUDO_PREFIX}{reason.upper()}{PSEUDO_PREFIX}"
        contrib_rows.append((pseudo_gene, rid, float(dv), abs(float(dv))))

    for _, r in rxn_df.iterrows():
        rid = r["reaction"]
        dv = float(r["delta_v"])
        adv = float(r["abs_delta_v"])
        if adv <= DELTA_V_EPS:
            continue

        try:
            rr = model.reactions.get_by_id(rid)
        except KeyError:
            stats["skipped_rxn_not_in_model"] += 1
            skipped_rows.append((rid, "rxn_not_in_model"))
            status_rows.append((rid, "skipped", "rxn_not_in_model"))
            add_pseudo(rid, dv, "rxn_not_in_model")
            continue

        rule = rr.gene_reaction_rule
        if not rule or not rr.genes:
            stats["skipped_no_gpr"] += 1
            skipped_rows.append((rid, "no_gpr"))
            status_rows.append((rid, "skipped", "no_gpr"))
            add_pseudo(rid, dv, "no_gpr")
            continue

        gene_ids = [g.id for g in rr.genes]
        clauses = gpr_to_clauses(rule, gene_ids)

        if clauses == ["__TOO_MANY__"]:
            stats["skipped_too_complex"] += 1
            skipped_rows.append((rid, "too_complex_dnf"))
            status_rows.append((rid, "skipped", "too_complex_dnf"))
            add_pseudo(rid, dv, "too_complex_dnf")
            continue

        if not clauses:
            stats["skipped_no_clause"] += 1
            skipped_rows.append((rid, "no_clause_after_parse"))
            status_rows.append((rid, "skipped", "no_clause_after_parse"))
            add_pseudo(rid, dv, "no_clause_after_parse")
            continue

        g_wt, expr_wt, clause_wt = dominant_clause_and_bottleneck(clauses, gene2wt)
        g_feni, expr_feni, clause_feni = dominant_clause_and_bottleneck(clauses, gene2feni)

        if (not g_wt) and (not g_feni):
            stats["skipped_no_bottleneck"] += 1
            skipped_rows.append((rid, "no_bottleneck_both_conditions"))
            status_rows.append((rid, "skipped", "no_bottleneck_both_conditions"))
            add_pseudo(rid, dv, "no_bottleneck_both_conditions")
            continue

        share_wt = 0.0
        share_feni = 0.0

        if g_wt and g_feni:
            if g_wt == g_feni:
                share_wt = dv
                share_feni = 0.0
                attribution_mode = "same_gene_no_split"
            else:
                if dv > 0:
                    share_wt = dv
                    share_feni = 0.0
                    attribution_mode = "directional_to_WT"
                elif dv < 0:
                    share_wt = 0.0
                    share_feni = dv
                    attribution_mode = "directional_to_FENI"
                else:
                    share_wt = 0.0
                    share_feni = 0.0
                    attribution_mode = "directional_zero"
        elif g_wt and (not g_feni):
            share_wt = dv
            share_feni = 0.0
            attribution_mode = "only_WT"
        else:
            share_wt = 0.0
            share_feni = dv
            attribution_mode = "only_FENI"

        bottleneck_rows.append({
            "reaction": rid,
            "delta_v": dv,
            "abs_delta_v": adv,
            "bottleneck_WT": g_wt if g_wt else "",
            "bottleneck_FENI": g_feni if g_feni else "",
            "dom_clause_expr_WT": expr_wt if expr_wt is not None else np.nan,
            "dom_clause_expr_FENI": expr_feni if expr_feni is not None else np.nan,
            "dom_clause_genes_WT": ";".join(sorted(list(clause_wt))) if clause_wt else "",
            "dom_clause_genes_FENI": ";".join(sorted(list(clause_feni))) if clause_feni else "",
            "attribution_mode": attribution_mode,
        })

        if g_wt and share_wt != 0.0:
            contrib_rows.append((g_wt, rid, share_wt, abs(share_wt)))
        if g_feni and share_feni != 0.0:
            contrib_rows.append((g_feni, rid, share_feni, abs(share_feni)))

        status_rows.append((rid, "attributed", ""))

    contrib_df = pd.DataFrame(contrib_rows, columns=["gene", "reaction", "contrib_signed", "contrib_abs"])
    bottleneck_df = pd.DataFrame(bottleneck_rows)
    skipped_df = pd.DataFrame(skipped_rows, columns=["reaction", "reason"])
    rxn_status_df = pd.DataFrame(status_rows, columns=["reaction", "status", "reason"])

    return contrib_df, bottleneck_df, skipped_df, rxn_status_df, stats


def attribution_balance_check(rxn_df: pd.DataFrame, contrib_df: pd.DataFrame, rxn_status_df: pd.DataFrame, out_path: str):
    base = rxn_df[["reaction", "delta_v", "abs_delta_v"]].copy()
    s = contrib_df.groupby("reaction", as_index=False)["contrib_signed"].sum().rename(columns={"contrib_signed": "sum_contrib_signed"})
    base = base.merge(s, on="reaction", how="left")
    base["sum_contrib_signed"] = base["sum_contrib_signed"].fillna(0.0)
    base["residual"] = base["delta_v"] - base["sum_contrib_signed"]

    base = base.merge(rxn_status_df, on="reaction", how="left")
    base["status"] = base["status"].fillna("not_evaluated")
    base["reason"] = base["reason"].fillna("")

    active = base[base["abs_delta_v"] > DELTA_V_EPS].copy()
    if active.empty:
        active.to_csv(out_path, sep="\t", index=False)
        print("[Balance] active reactions empty")
        return

    tol = np.maximum(ATTRIB_TOL_ABS, ATTRIB_TOL_REL * active["abs_delta_v"].values)
    active["pass_balance"] = np.abs(active["residual"].values) <= tol

    attributed = active[active["status"] == "attributed"]
    skipped = active[active["status"] == "skipped"]

    fail_attr = int((~attributed["pass_balance"]).sum()) if not attributed.empty else 0
    fail_skip = int((~skipped["pass_balance"]).sum()) if not skipped.empty else 0

    active.to_csv(out_path, sep="\t", index=False)
    print(f"[Balance] active={active.shape[0]}, attributed={attributed.shape[0]}, skipped={skipped.shape[0]}, "
          f"attributed_fail={fail_attr}, skipped_fail={fail_skip}, max|residual|={float(np.max(np.abs(active['residual'].values))):.3g} -> {out_path}")

def attribution_coverage_report(rxn_df: pd.DataFrame, rxn_status_df: pd.DataFrame, out_path: str):
    rxn_active = rxn_df[rxn_df["abs_delta_v"] > DELTA_V_EPS][["reaction", "abs_delta_v"]].copy()
    total_abs = float(rxn_active["abs_delta_v"].sum()) if not rxn_active.empty else 0.0

    st = rxn_status_df.merge(rxn_active, on="reaction", how="inner")
    if st.empty or total_abs == 0:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# summary\n")
            pd.DataFrame(columns=["category", "n_reactions", "sum_abs_delta_v", "pct_of_total"]).to_csv(f, sep="\t", index=False)
            f.write("\n# skipped_by_reason\n")
            pd.DataFrame(columns=["reason", "n_reactions", "sum_abs_delta_v", "pct_of_total"]).to_csv(f, sep="\t", index=False)
        print(f"[Coverage] total Σ|Δv|=0 -> {out_path}")
        return

    summary_rows = []
    for cat in ["attributed", "skipped"]:
        sub = st[st["status"] == cat]
        n = int(sub["reaction"].nunique())
        sabs = float(sub["abs_delta_v"].sum())
        pct = sabs / total_abs * 100.0
        summary_rows.append((cat, n, sabs, pct))
    summary_rows.append(("total_active", int(st["reaction"].nunique()), total_abs, 100.0))
    summary = pd.DataFrame(summary_rows, columns=["category", "n_reactions", "sum_abs_delta_v", "pct_of_total"])

    sk = st[st["status"] == "skipped"].copy()
    if sk.empty:
        skip_reason = pd.DataFrame(columns=["reason", "n_reactions", "sum_abs_delta_v", "pct_of_total"])
    else:
        grp = sk.groupby("reason", as_index=False).agg(
            n_reactions=("reaction", "nunique"),
            sum_abs_delta_v=("abs_delta_v", "sum"),
        )
        grp["pct_of_total"] = grp["sum_abs_delta_v"] / total_abs * 100.0
        skip_reason = grp.sort_values("sum_abs_delta_v", ascending=False)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# summary\n")
        summary.to_csv(f, sep="\t", index=False)
        f.write("\n# skipped_by_reason\n")
        skip_reason.to_csv(f, sep="\t", index=False)

    print(f"[Coverage] total Σ|Δv|={total_abs:.6g}, attributed={float(summary.loc[summary['category']=='attributed','pct_of_total'].values[0]):.2f}%, "
          f"skipped={float(summary.loc[summary['category']=='skipped','pct_of_total'].values[0]):.2f}% -> {out_path}")


def main():
    mkdir(OUT_DIR)

    model = cobra.io.read_sbml_model(SBML_PATH)


    wt = read_tpm_noheader(TPM_WT_PATH, "WT")
    feni = read_tpm_noheader(TPM_FENI_PATH, "FENI")
    df_expr = wt.merge(feni, on="gene", how="outer").fillna(0.0)

    gene2wt = dict(zip(df_expr["gene"], df_expr["expr_WT"]))
    gene2feni = dict(zip(df_expr["gene"], df_expr["expr_FENI"]))
    print(f"[TPM->Expr] genes={df_expr.shape[0]}, REP_AGG={REP_AGG}")


    d = pd.read_csv(DEG_CSV)
    required_d = {D_COL_GENE, D_COL_NAME, D_COL_LOG2FC, D_COL_QVALUE}
    missing_d = required_d - set(d.columns)
    if missing_d:
        raise ValueError(
            f"DEG_CSV missing required columns: {sorted(missing_d)}. "
            f"Available: {list(d.columns)}"
        )

    deg_map = (
        d[[D_COL_GENE, D_COL_NAME, D_COL_LOG2FC, D_COL_QVALUE]]
        .rename(columns={
            D_COL_GENE: "gene",
            D_COL_NAME: "gene_name",
            D_COL_LOG2FC: "log2fc",
            D_COL_QVALUE: "qvalue",
        })
        .drop_duplicates(subset=["gene"])
    )
    deg_map["log2fc"] = to_num(deg_map["log2fc"])
    deg_map["qvalue"] = to_num(deg_map["qvalue"])

    df_expr = df_expr.merge(deg_map, on="gene", how="left")
    n_anno = int(df_expr["log2fc"].notna().sum())
    print(f"[DEG annotate] rows_in_DEG={deg_map.shape[0]}, annotated_genes={n_anno}/{df_expr.shape[0]}")


    wt_rxn2expr = compute_reaction_expression(model, gene2wt)
    feni_rxn2expr = compute_reaction_expression(model, gene2feni)
    print(f"[RxnExpr coverage] WT_rxn2expr={len(wt_rxn2expr)}, FENI_rxn2expr={len(feni_rxn2expr)}")

    wt_model = apply_eflux(model, wt_rxn2expr, "WT_24h", out_dir=OUT_DIR)
    feni_model = apply_eflux(model, feni_rxn2expr, "FENI_24h", out_dir=OUT_DIR)


    sol_wt = wt_model.optimize()
    sol_feni = feni_model.optimize()
    obj_wt = float(sol_wt.objective_value)
    obj_feni = float(sol_feni.objective_value)
    obj_common = min(obj_wt, obj_feni)
    print(f"[Common objective] obj*_WT={obj_wt:.6g}, obj*_FENI={obj_feni:.6g}, obj_common={obj_common:.6g}")


    wt_sol_pfba = solve_pfba_common(wt_model, "WT_24h", obj_common)
    feni_sol_pfba = solve_pfba_common(feni_model, "FENI_24h", obj_common)


    flux_wt = wt_sol_pfba.fluxes.rename("v_WT").to_frame()
    flux_feni = feni_sol_pfba.fluxes.rename("v_FENI").to_frame()


    rxn_meta = []
    for rxn in model.reactions:
        rxn_meta.append({
            "reaction": rxn.id,
            "name": rxn.name,
            "subsystem": getattr(rxn, "subsystem", "") if getattr(rxn, "subsystem", None) else "",
            "gpr": rxn.gene_reaction_rule if rxn.gene_reaction_rule else "",
        })
    rxn_meta = pd.DataFrame(rxn_meta)

    rxn = flux_wt.join(flux_feni, how="outer").fillna(0.0)
    rxn["delta_v"] = rxn["v_FENI"] - rxn["v_WT"]
    rxn["abs_delta_v"] = rxn["delta_v"].abs()
    rxn = rxn.reset_index().rename(columns={"index": "reaction"})
    rxn = rxn.merge(rxn_meta, on="reaction", how="left")


    contrib_df, bottleneck_df, skipped_df, rxn_status_df, stats = build_gene_contrib_from_deltaflux(model, rxn, gene2wt, gene2feni)


    gene_agg = contrib_df.groupby("gene", as_index=False).agg(
        hit_reactions_n=("reaction", "nunique"),
        sum_abs_delta_v=("contrib_abs", "sum"),
        sum_signed_delta_v=("contrib_signed", "sum"),
    )

    gene_agg["flux_direction"] = np.where(
        gene_agg["sum_signed_delta_v"] > 0, "FENI_increase_flux",
        np.where(gene_agg["sum_signed_delta_v"] < 0, "FENI_decrease_flux", "zero")
    )

    gene_agg_real = gene_agg[~gene_agg["gene"].astype(str).str.startswith(PSEUDO_PREFIX)].copy()
    gene_agg_pseudo = gene_agg[gene_agg["gene"].astype(str).str.startswith(PSEUDO_PREFIX)].copy()

    gene_meta = df_expr[["gene", "expr_WT", "expr_FENI", "log2fc", "qvalue", "gene_name"]].copy()
    gene_agg_real = gene_agg_real.merge(gene_meta, on="gene", how="left")

    rxn_list = contrib_df.groupby("gene")["reaction"].apply(lambda s: ";".join(sorted(set(s)))).reset_index()
    gene_agg_real = gene_agg_real.merge(rxn_list, on="gene", how="left").rename(columns={"reaction": "hit_reactions"})
    gene_agg_pseudo = gene_agg_pseudo.merge(rxn_list, on="gene", how="left").rename(columns={"reaction": "hit_reactions"})

    gene_agg_real["abs_log2fc"] = gene_agg_real["log2fc"].abs()

    gene_agg_real = gene_agg_real.sort_values(
        ["sum_abs_delta_v", "hit_reactions_n", "abs_log2fc"],
        ascending=[False, False, False]
    )
    gene_agg_pseudo = gene_agg_pseudo.sort_values(
        ["sum_abs_delta_v", "hit_reactions_n"],
        ascending=[False, False]
    )


    columns = ["gene", "gene_name", "log2fc", "qvalue", "sum_abs_delta_v", "sum_signed_delta_v"]
    output = OUT_DIR / "FE-NI_gene_scores.tsv"
    gene_agg_real[columns].to_csv(output, sep="\t", index=False)
    print("[DONE]", output)


if __name__ == "__main__":
    main()
