import os
import re
import ast
from pathlib import Path
import numpy as np
import pandas as pd

import cobra
from cobra.flux_analysis import pfba


ROOT = Path(__file__).resolve().parents[1]
SBML_PATH = ROOT / "models" / "iMT1026_meoh.xml"
TPM_FEI0_PATH = ROOT / "data" / "TPM_FE-I_0h.csv"
TPM_FEI24_PATH = ROOT / "data" / "TPM_FE-I_24h.csv"


DEG_CSV = ROOT / "data" / "All Data_FE(I)_24h_vs_0h.csv"

OUT_DIR = ROOT / "results"

TOPN_RXN = 30
TOPN_GENE = 30

MIN_SCALE = 3e-3
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


def read_deg_table(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)


    D_COL_GENE = "gene_symbol"
    D_COL_NAME = "gene_name"
    D_COL_LOG2FC = "log2fc"
    D_COL_QVALUE = "qvalue"

    required = {D_COL_GENE, D_COL_NAME, D_COL_LOG2FC, D_COL_QVALUE}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"DEG_CSV missing required columns: {sorted(missing)}. Available: {list(d.columns)}")

    d = d[[D_COL_GENE, D_COL_NAME, D_COL_LOG2FC, D_COL_QVALUE]].copy()
    d[D_COL_GENE] = d[D_COL_GENE].astype(str)
    d[D_COL_LOG2FC] = to_num(d[D_COL_LOG2FC])
    d[D_COL_QVALUE] = to_num(d[D_COL_QVALUE])


    d = d.drop_duplicates(subset=[D_COL_GENE], keep="first").rename(
        columns={
            D_COL_GENE: "gene",
            D_COL_NAME: "gene_name",
            D_COL_LOG2FC: "log2fc",
            D_COL_QVALUE: "qvalue",
        }
    )
    return d


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
    print(
        f"[E-Flux] {mode_name}: changed={changed}, "
        f"denom={'q'+str(int(ROBUST_Q*100)) if USE_ROBUST_MAX else 'max'}={denom:.4g}, "
        f"scale(min/median/max)={arr.min():.3g}/{np.median(arr):.3g}/{arr.max():.3g}, "
        f"MIN_SCALE_hits={minscale_hits}, MAX_SCALE_hits={cap_hits}"
    )

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


def build_gene_contrib_from_deltaflux(model, rxn_df: pd.DataFrame, gene2fei0: dict, gene2fei24: dict):
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

        g_0h, expr_0h, clause_0h = dominant_clause_and_bottleneck(clauses, gene2fei0)
        g_24h, expr_24h, clause_24h = dominant_clause_and_bottleneck(clauses, gene2fei24)

        if (not g_0h) and (not g_24h):
            stats["skipped_no_bottleneck"] += 1
            skipped_rows.append((rid, "no_bottleneck_both_conditions"))
            status_rows.append((rid, "skipped", "no_bottleneck_both_conditions"))
            add_pseudo(rid, dv, "no_bottleneck_both_conditions")
            continue

        share_0h = 0.0
        share_24h = 0.0

        if g_0h and g_24h:
            if g_0h == g_24h:
                share_0h = dv
                share_24h = 0.0
                attribution_mode = "same_gene_no_split"
            else:

                if dv > 0:
                    share_0h = dv
                    share_24h = 0.0
                    attribution_mode = "directional_to_FEI0"
                elif dv < 0:
                    share_0h = 0.0
                    share_24h = dv
                    attribution_mode = "directional_to_FEI24"
                else:
                    share_0h = 0.0
                    share_24h = 0.0
                    attribution_mode = "directional_zero"
        elif g_0h and (not g_24h):
            share_0h = dv
            share_24h = 0.0
            attribution_mode = "only_FEI0"
        else:
            share_0h = 0.0
            share_24h = dv
            attribution_mode = "only_FEI24"

        bottleneck_rows.append(
            {
                "reaction": rid,
                "delta_v": dv,
                "abs_delta_v": adv,
                "bottleneck_FEI0": g_0h if g_0h else "",
                "bottleneck_FEI24": g_24h if g_24h else "",
                "dom_clause_expr_FEI0": expr_0h if expr_0h is not None else np.nan,
                "dom_clause_expr_FEI24": expr_24h if expr_24h is not None else np.nan,
                "dom_clause_genes_FEI0": ";".join(sorted(list(clause_0h))) if clause_0h else "",
                "dom_clause_genes_FEI24": ";".join(sorted(list(clause_24h))) if clause_24h else "",
                "attribution_mode": attribution_mode,
            }
        )

        if g_0h and share_0h != 0.0:
            contrib_rows.append((g_0h, rid, share_0h, abs(share_0h)))
        if g_24h and share_24h != 0.0:
            contrib_rows.append((g_24h, rid, share_24h, abs(share_24h)))

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
    print(
        f"[Balance] active={active.shape[0]}, attributed={attributed.shape[0]}, skipped={skipped.shape[0]}, "
        f"attributed_fail={fail_attr}, skipped_fail={fail_skip}, max|residual|={float(np.max(np.abs(active['residual'].values))):.3g} -> {out_path}"
    )


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

    print(
        f"[Coverage] total Σ|Δv|={total_abs:.6g}, "
        f"attributed={float(summary.loc[summary['category']=='attributed','pct_of_total'].values[0]):.2f}%, "
        f"skipped={float(summary.loc[summary['category']=='skipped','pct_of_total'].values[0]):.2f}% -> {out_path}"
    )


def main():
    mkdir(OUT_DIR)

    model = cobra.io.read_sbml_model(SBML_PATH)


    fei0 = read_tpm_noheader(TPM_FEI0_PATH, "FEI0")
    fei24 = read_tpm_noheader(TPM_FEI24_PATH, "FEI24")
    df_expr = fei0.merge(fei24, on="gene", how="outer").fillna(0.0)

    gene2fei0 = dict(zip(df_expr["gene"], df_expr["expr_FEI0"]))
    gene2fei24 = dict(zip(df_expr["gene"], df_expr["expr_FEI24"]))

    print(f"[TPM->Expr] genes={df_expr.shape[0]}, REP_AGG={REP_AGG}")


    deg = read_deg_table(DEG_CSV)
    df_expr = df_expr.merge(deg, on="gene", how="left")


    fei0_rxn2expr = compute_reaction_expression(model, gene2fei0)
    fei24_rxn2expr = compute_reaction_expression(model, gene2fei24)
    print(f"[RxnExpr coverage] FEI0_rxn2expr={len(fei0_rxn2expr)}, FEI24_rxn2expr={len(fei24_rxn2expr)}")

    fei0_model = apply_eflux(model, fei0_rxn2expr, "FEI0_0h", out_dir=OUT_DIR)
    fei24_model = apply_eflux(model, fei24_rxn2expr, "FEI24_24h", out_dir=OUT_DIR)


    sol_0h = fei0_model.optimize()
    sol_24h = fei24_model.optimize()
    obj_0h = float(sol_0h.objective_value)
    obj_24h = float(sol_24h.objective_value)
    obj_common = min(obj_0h, obj_24h)
    print(f"[Common objective] obj*_FEI0={obj_0h:.6g}, obj*_FEI24={obj_24h:.6g}, obj_common={obj_common:.6g}")


    sol_pfba_0h = solve_pfba_common(fei0_model, "FEI0_0h", obj_common)
    sol_pfba_24h = solve_pfba_common(fei24_model, "FEI24_24h", obj_common)


    flux_0h = sol_pfba_0h.fluxes.rename("v_FEI0").to_frame()
    flux_24h = sol_pfba_24h.fluxes.rename("v_FEI24").to_frame()


    rxn_meta = []
    for rxn in model.reactions:
        rxn_meta.append(
            {
                "reaction": rxn.id,
                "name": rxn.name,
                "subsystem": getattr(rxn, "subsystem", "") if getattr(rxn, "subsystem", None) else "",
                "gpr": rxn.gene_reaction_rule if rxn.gene_reaction_rule else "",
            }
        )
    rxn_meta = pd.DataFrame(rxn_meta)

    rxn = flux_0h.join(flux_24h, how="outer").fillna(0.0)
    rxn["delta_v"] = rxn["v_FEI24"] - rxn["v_FEI0"]  
    rxn["abs_delta_v"] = rxn["delta_v"].abs()
    rxn = rxn.reset_index().rename(columns={"index": "reaction"})
    rxn = rxn.merge(rxn_meta, on="reaction", how="left")


    contrib_df, bottleneck_df, skipped_df, rxn_status_df, stats = build_gene_contrib_from_deltaflux(
        model, rxn, gene2fei0, gene2fei24
    )


    gene_agg = contrib_df.groupby("gene", as_index=False).agg(
        hit_reactions_n=("reaction", "nunique"),
        sum_abs_delta_v=("contrib_abs", "sum"),
        sum_signed_delta_v=("contrib_signed", "sum"),
    )

    gene_agg["flux_direction"] = np.where(
        gene_agg["sum_signed_delta_v"] > 0,
        "FEI24_increase_flux",
        np.where(gene_agg["sum_signed_delta_v"] < 0, "FEI24_decrease_flux", "zero"),
    )

    gene_agg_real = gene_agg[~gene_agg["gene"].astype(str).str.startswith(PSEUDO_PREFIX)].copy()
    gene_agg_pseudo = gene_agg[gene_agg["gene"].astype(str).str.startswith(PSEUDO_PREFIX)].copy()


    gene_meta = df_expr[["gene", "expr_FEI0", "expr_FEI24", "log2fc", "qvalue", "gene_name"]].copy()
    gene_agg_real = gene_agg_real.merge(gene_meta, on="gene", how="left")

    rxn_list = contrib_df.groupby("gene")["reaction"].apply(lambda s: ";".join(sorted(set(s)))).reset_index()
    gene_agg_real = gene_agg_real.merge(rxn_list, on="gene", how="left").rename(columns={"reaction": "hit_reactions"})
    gene_agg_pseudo = gene_agg_pseudo.merge(rxn_list, on="gene", how="left").rename(columns={"reaction": "hit_reactions"})

    gene_agg_real["abs_log2fc"] = gene_agg_real["log2fc"].abs()

    gene_agg_real = gene_agg_real.sort_values(
        ["sum_abs_delta_v", "hit_reactions_n", "abs_log2fc"],
        ascending=[False, False, False],
    )
    gene_agg_pseudo = gene_agg_pseudo.sort_values(
        ["sum_abs_delta_v", "hit_reactions_n"],
        ascending=[False, False],
    )


    columns = ["gene", "gene_name", "log2fc", "qvalue", "sum_abs_delta_v", "sum_signed_delta_v"]
    output = OUT_DIR / "FE-I_gene_scores.tsv"
    gene_agg_real[columns].to_csv(output, sep="\t", index=False)
    print("[DONE]", output)


if __name__ == "__main__":
    main()
