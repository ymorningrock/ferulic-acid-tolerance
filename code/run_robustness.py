from __future__ import annotations

import ast
import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import clarabel
import cobra
import numpy as np
import pandas as pd
from cobra.util.solver import linear_reaction_coefficients
from scipy import sparse
from scipy.optimize import linprog
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class Comparison:
    name: str
    model_file: str
    expression_a: str
    expression_b: str
    label_a: str
    label_b: str
    deg_file: str
    target_locus: str
    target_name: str
    min_scale: float
    carbon_reaction: str
    carbon_fixed_flux: float


@dataclass(frozen=True)
class Scenario:
    name: str
    normalization: str
    medium: str = "closed"
    or_rule: str = "max"
    biomass_mode: str = "common"


COMPARISONS = (
    Comparison(
        name="FENI_vs_WT",
        model_file="iMT1026_glyc.xml",
        expression_a="TPM_WT_24h.csv",
        expression_b="TPM_FE-NI_24h.csv",
        label_a="WT_24h",
        label_b="FENI_24h",
        deg_file="All Data_24h_FE_vs_WT.csv",
        target_locus="PAS_chr3_0434",
        target_name="AGC1",
        min_scale=1e-6,
        carbon_reaction="Ex_glyc",
        carbon_fixed_flux=-1.0,
    ),
    Comparison(
        name="FEI_24h_vs_0h",
        model_file="iMT1026_meoh.xml",
        expression_a="TPM_FE-I_0h.csv",
        expression_b="TPM_FE-I_24h.csv",
        label_a="FEI_0h",
        label_b="FEI_24h",
        deg_file="All Data_FE(I)_24h_vs_0h.csv",
        target_locus="PAS_chr3_0834",
        target_name="PAS_chr3_0834",
        min_scale=3e-3,
        carbon_reaction="Ex_meoh",
        carbon_fixed_flux=-3.0,
    ),
)


SCENARIOS = (
    Scenario("max_independent", "max_independent"),
    Scenario("q99_independent", "q99_independent"),
    Scenario("log1p_max_independent", "log1p_max_independent"),
    Scenario("rank_independent", "rank_independent"),
)


def load_legacy_module():
    path = ROOT / "code" / "Eflux_FE-NI.py"
    spec = importlib.util.spec_from_file_location("legacy_eflux", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


LEGACY_MODULE = load_legacy_module()


def read_expression(path: Path) -> pd.Series:
    frame = pd.read_csv(path, header=None)
    values = frame.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    medians = values.median(axis=1).astype(float)
    return pd.Series(medians.to_numpy(), index=frame.iloc[:, 0].astype(str), dtype=float)


def rank_transform(values: pd.Series) -> pd.Series:
    positive = values > 0
    transformed = pd.Series(0.0, index=values.index, dtype=float)
    if positive.any():
        transformed.loc[positive] = rankdata(values.loc[positive], method="average") / positive.sum()
    return transformed


def transform_pair(a: pd.Series, b: pd.Series, normalization: str) -> tuple[pd.Series, pd.Series]:
    if normalization == "log1p_max_independent":
        return np.log1p(a), np.log1p(b)
    if normalization == "rank_independent":
        return rank_transform(a), rank_transform(b)
    return a.copy(), b.copy()


def reaction_expression(model: cobra.Model, gene_values: pd.Series, or_rule: str) -> dict[str, float]:
    mapping = gene_values.to_dict()
    output: dict[str, float] = {}
    for reaction in model.reactions:
        rule = reaction.gene_reaction_rule
        gene_ids = [gene.id for gene in reaction.genes]
        if not rule or not gene_ids:
            continue
        sorted_genes = sorted(set(gene_ids), key=len, reverse=True)
        placeholder_to_gene: dict[str, str] = {}
        expression = rule
        for index, gene_id in enumerate(sorted_genes):
            placeholder = f"g{index}"
            placeholder_to_gene[placeholder] = gene_id
            expression = re.sub(rf"(?<!\w){re.escape(gene_id)}(?!\w)", placeholder, expression)
        expression = expression.replace("AND", "and").replace("OR", "or")
        try:
            node = ast.parse(expression, mode="eval").body
        except SyntaxError:
            continue

        def evaluate(current):
            if isinstance(current, ast.Name):
                gene_id = placeholder_to_gene.get(current.id)
                return None if gene_id not in mapping else float(mapping[gene_id])
            if isinstance(current, ast.BoolOp):
                values = [evaluate(child) for child in current.values]
                if any(value is None for value in values):
                    return None
                if isinstance(current.op, ast.And):
                    return float(np.min(values))
                if isinstance(current.op, ast.Or):
                    if or_rule == "max":
                        return float(np.max(values))
                    if or_rule == "sum":
                        return float(np.sum(values))
            return None

        value = evaluate(node)
        if value is not None and np.isfinite(value):
            output[reaction.id] = float(value)
    return output


def scaling_denominators(
    reaction_a: dict[str, float], reaction_b: dict[str, float], normalization: str
) -> tuple[float, float]:
    a = np.asarray(list(reaction_a.values()), dtype=float)
    b = np.asarray(list(reaction_b.values()), dtype=float)
    if normalization == "max_independent":
        return float(np.max(a)), float(np.max(b))
    if normalization == "log1p_max_independent":
        return float(np.max(a)), float(np.max(b))
    if normalization == "q99_independent":
        return float(np.quantile(a, 0.99)), float(np.quantile(b, 0.99))
    if normalization == "rank_independent":
        return float(np.max(a)), float(np.max(b))
    raise ValueError(f"Unknown normalization: {normalization}")


def configure_environment(model: cobra.Model, comparison: Comparison, scenario: Scenario) -> dict:
    if scenario.medium != "closed":
        raise ValueError(f"Unknown medium: {scenario.medium}")
    carbon = model.reactions.get_by_id(comparison.carbon_reaction)
    carbon.bounds = (comparison.carbon_fixed_flux, comparison.carbon_fixed_flux)

    return {
        "carbon_reaction": comparison.carbon_reaction,
        "carbon_bounds": list(carbon.bounds),
        "complex_organic_nutrient_uptake": "closed",
    }


def apply_expression_constraints(
    model: cobra.Model,
    reaction_values: dict[str, float],
    denominator: float,
    min_scale: float,
) -> tuple[cobra.Model, pd.DataFrame]:
    constrained = model.copy()
    rows = []
    for reaction in constrained.reactions:
        if reaction.id not in reaction_values:
            continue
        raw_scale = reaction_values[reaction.id] / denominator if denominator > 0 else 0.0
        applied_scale = min(10.0, max(min_scale, raw_scale))
        original_lower, original_upper = reaction.bounds
        reaction.bounds = (original_lower * applied_scale, original_upper * applied_scale)
        rows.append(
            {
                "reaction": reaction.id,
                "reaction_expression": reaction_values[reaction.id],
                "raw_scale": raw_scale,
                "applied_scale": applied_scale,
                "original_lower": original_lower,
                "original_upper": original_upper,
                "scaled_lower": reaction.lower_bound,
                "scaled_upper": reaction.upper_bound,
            }
        )
    return constrained, pd.DataFrame(rows)


def objective_vector(model: cobra.Model) -> np.ndarray:
    coefficients = linear_reaction_coefficients(model)
    return np.asarray([float(coefficients.get(reaction, 0.0)) for reaction in model.reactions])


def solve_biomass_lp(model: cobra.Model) -> tuple[pd.Series, float, dict]:
    stoichiometry = stoichiometric_matrix(model)
    objective = objective_vector(model)
    lower = np.asarray([reaction.lower_bound for reaction in model.reactions], dtype=float)
    upper = np.asarray([reaction.upper_bound for reaction in model.reactions], dtype=float)
    result = linprog(
        -objective,
        A_eq=stoichiometry,
        b_eq=np.zeros(stoichiometry.shape[0]),
        bounds=list(zip(lower, upper)),
        method="highs",
        options={
            "primal_feasibility_tolerance": 1e-8,
            "dual_feasibility_tolerance": 1e-8,
        },
    )
    if not result.success:
        raise RuntimeError(f"Biomass LP failed: {result.message}")
    flux = np.asarray(result.x, dtype=float)
    mass_balance = stoichiometry @ flux
    value = float(objective @ flux)
    metadata = {
        "solver_status": str(result.message),
        "iterations": int(result.nit),
        "objective_biomass": value,
        "max_mass_balance_residual": float(np.max(np.abs(mass_balance))),
    }
    return (
        pd.Series(flux, index=[reaction.id for reaction in model.reactions]),
        value,
        metadata,
    )


def stoichiometric_matrix(model: cobra.Model) -> sparse.csc_matrix:
    metabolite_index = {metabolite.id: i for i, metabolite in enumerate(model.metabolites)}
    rows = []
    columns = []
    data = []
    for j, reaction in enumerate(model.reactions):
        for metabolite, coefficient in reaction.metabolites.items():
            rows.append(metabolite_index[metabolite.id])
            columns.append(j)
            data.append(float(coefficient))
    matrix = sparse.coo_matrix(
        (data, (rows, columns)), shape=(len(model.metabolites), len(model.reactions))
    )
    return matrix.tocsc()


def solve_l2(
    model: cobra.Model,
    biomass_value: float,
    initial_flux: pd.Series | None = None,
) -> tuple[pd.Series, dict]:
    n_reactions = len(model.reactions)
    stoichiometry = stoichiometric_matrix(model)
    objective = objective_vector(model)
    lower_bounds = np.asarray([reaction.lower_bound for reaction in model.reactions], dtype=float)
    upper_bounds = np.asarray([reaction.upper_bound for reaction in model.reactions], dtype=float)

    fixed = np.isclose(lower_bounds, upper_bounds, atol=1e-12, rtol=0.0)
    free = ~fixed
    fixed_flux = lower_bounds[fixed]
    free_indices = np.flatnonzero(free)
    fixed_indices = np.flatnonzero(fixed)

    mass_balance_rhs = -(stoichiometry[:, fixed] @ fixed_flux)
    biomass_rhs = biomass_value - float(objective[fixed] @ fixed_flux)
    equality_matrix_full = sparse.vstack(
        [stoichiometry[:, free], sparse.csc_matrix(objective[free].reshape(1, -1))],
        format="csc",
    )
    equality_rhs_full = np.concatenate([np.asarray(mass_balance_rhs).ravel(), [biomass_rhs]])
    equality_matrix = equality_matrix_full.copy()
    equality_rhs = equality_rhs_full.copy()

    nonzero_rows = np.asarray(abs(equality_matrix).max(axis=1).toarray()).ravel() > 0
    if np.any(np.abs(equality_rhs[~nonzero_rows]) > 1e-10):
        raise RuntimeError("Fixed reaction bounds create an inconsistent equality constraint")
    equality_matrix = equality_matrix[nonzero_rows, :].tocsc()
    equality_rhs = equality_rhs[nonzero_rows]
    retained_rows = np.arange(equality_matrix.shape[0], dtype=int)

    row_maximum = np.asarray(abs(equality_matrix).max(axis=1).toarray()).ravel()
    row_maximum[row_maximum == 0] = 1.0
    constraints = (sparse.diags(1.0 / row_maximum) @ equality_matrix).tocsc()
    row_bounds = equality_rhs / row_maximum

    identity = sparse.eye(len(free_indices), format="csc")
    conic_matrix = sparse.vstack([constraints, -identity, identity], format="csc")
    conic_rhs = np.concatenate(
        [row_bounds, -lower_bounds[free], upper_bounds[free]]
    )
    cones = [
        clarabel.ZeroConeT(constraints.shape[0]),
        clarabel.NonnegativeConeT(2 * len(free_indices)),
    ]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = 500
    settings.max_threads = 4
    settings.direct_solve_method = "faer"
    settings.time_limit = 120.0
    settings.tol_gap_abs = 1e-12
    settings.tol_gap_rel = 1e-12
    settings.tol_feas = 1e-12
    solver = clarabel.DefaultSolver(
        2.0 * identity,
        np.zeros(len(free_indices)),
        conic_matrix,
        conic_rhs,
        cones,
        settings,
    )
    solution = solver.solve()
    if solution.status not in {
        clarabel.SolverStatus.Solved,
        clarabel.SolverStatus.AlmostSolved,
    }:
        raise RuntimeError(f"Clarabel failed: {solution.status}")

    flux = np.empty(n_reactions, dtype=float)
    flux[fixed_indices] = fixed_flux
    optimized_flux = np.clip(
        np.asarray(solution.x, dtype=float),
        lower_bounds[free],
        upper_bounds[free],
    )

    

    
    # Correct rescaling residuals with the smallest bound-preserving L1 adjustment.
    polish_matrix = equality_matrix_full[nonzero_rows, :].tocsc()
    polish_rhs = equality_rhs_full[nonzero_rows] - polish_matrix @ optimized_flux
    positive_room = np.maximum(upper_bounds[free] - optimized_flux, 0.0)
    negative_room = np.maximum(optimized_flux - lower_bounds[free], 0.0)
    polish = linprog(
        np.ones(2 * len(free_indices)),
        A_eq=sparse.hstack([polish_matrix, -polish_matrix], format="csc"),
        b_eq=np.asarray(polish_rhs).ravel(),
        bounds=[(0.0, value) for value in positive_room]
        + [(0.0, value) for value in negative_room],
        method="highs",
        options={
            "primal_feasibility_tolerance": 1e-9,
            "dual_feasibility_tolerance": 1e-9,
        },
    )
    if polish.success:
        correction = polish.x[: len(free_indices)] - polish.x[len(free_indices) :]
        optimized_flux = optimized_flux + correction
        correction_norm = float(np.linalg.norm(correction))
        correction_l1 = float(np.abs(correction).sum())
        polish_status = str(polish.message)
    else:
        correction_norm = 0.0
        correction_l1 = 0.0
        polish_status = f"not applied: {polish.message}"
    flux[free_indices] = optimized_flux
    mass_balance = stoichiometry @ flux
    maximum_mass_balance_row = int(np.argmax(np.abs(mass_balance)))
    biomass_observed = float(objective @ flux)
    metadata = {
        "solver_status": str(solution.status),
        "iterations": int(solution.iterations),
        "solver_primal_residual": float(solution.r_prim),
        "solver_dual_residual": float(solution.r_dual),
        "feasibility_correction_l2_norm": correction_norm,
        "feasibility_correction_l1_norm": correction_l1,
        "feasibility_correction_status": polish_status,
        "objective_l2_squared": float(np.dot(flux, flux)),
        "max_mass_balance_residual": float(np.max(np.abs(mass_balance))),
        "max_mass_balance_metabolite": model.metabolites[maximum_mass_balance_row].id,
        "reactions_total": int(n_reactions),
        "reactions_fixed": int(fixed.sum()),
        "reactions_optimized": int(free.sum()),
        "equality_rows_before_reduction": int(nonzero_rows.sum()),
        "equality_rows_retained": int(len(retained_rows)),
        "requested_biomass": float(biomass_value),
        "observed_biomass": biomass_observed,
        "biomass_residual": biomass_observed - biomass_value,
        "max_lower_bound_violation": float(np.max(np.maximum(lower_bounds - flux, 0.0))),
        "max_upper_bound_violation": float(np.max(np.maximum(flux - upper_bounds, 0.0))),
    }
    if metadata["max_mass_balance_residual"] > 2e-5 or abs(metadata["biomass_residual"]) > 2e-8:
        raise RuntimeError(f"QP validation failed: {metadata}")
    return pd.Series(flux, index=[reaction.id for reaction in model.reactions]), metadata


def read_deg(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    selected = frame[["gene_symbol", "gene_name", "log2fc", "qvalue"]].copy()
    selected = selected.rename(columns={"gene_symbol": "gene"}).drop_duplicates("gene")
    selected["gene"] = selected["gene"].astype(str)
    selected["log2fc"] = pd.to_numeric(selected["log2fc"], errors="coerce")
    selected["qvalue"] = pd.to_numeric(selected["qvalue"], errors="coerce")
    return selected


def build_rankings(
    model: cobra.Model,
    flux_a: pd.Series,
    flux_b: pd.Series,
    expression_a: pd.Series,
    expression_b: pd.Series,
    deg: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    reaction_meta = pd.DataFrame(
        {
            "reaction": [reaction.id for reaction in model.reactions],
            "name": [reaction.name for reaction in model.reactions],
            "gpr": [reaction.gene_reaction_rule for reaction in model.reactions],
        }
    )
    reactions = pd.concat([flux_a.rename("v_a"), flux_b.rename("v_b")], axis=1).fillna(0.0)
    reactions["delta_v"] = reactions["v_b"] - reactions["v_a"]
    reactions["abs_delta_v"] = reactions["delta_v"].abs()
    reactions = reactions.reset_index().rename(columns={"index": "reaction"}).merge(
        reaction_meta, on="reaction", how="left"
    )

    contribution, _, skipped, status, statistics = LEGACY_MODULE.build_gene_contrib_from_deltaflux(
        model,
        reactions,
        expression_a.to_dict(),
        expression_b.to_dict(),
    )
    genes = contribution[~contribution["gene"].astype(str).str.startswith("__")].copy()
    aggregated = genes.groupby("gene", as_index=False).agg(
        hit_reactions_n=("reaction", "nunique"),
        sum_abs_delta_v=("contrib_abs", "sum"),
        sum_signed_delta_v=("contrib_signed", "sum"),
    )
    reaction_list = genes.groupby("gene")["reaction"].apply(
        lambda series: ";".join(sorted(set(series)))
    )
    aggregated = aggregated.merge(reaction_list.rename("hit_reactions"), on="gene", how="left")
    aggregated = aggregated.merge(deg, on="gene", how="left")
    aggregated["expr_a"] = aggregated["gene"].map(expression_a)
    aggregated["expr_b"] = aggregated["gene"].map(expression_b)
    aggregated["sum_abs_delta_v"] = pd.to_numeric(
        aggregated["sum_abs_delta_v"], errors="coerce"
    )
    aggregated["sum_signed_delta_v"] = pd.to_numeric(
        aggregated["sum_signed_delta_v"], errors="coerce"
    )
    aggregated["abs_log2fc"] = aggregated["log2fc"].abs()
    aggregated["_score_abs_sort"] = aggregated["sum_abs_delta_v"].round(6)
    aggregated["_score_signed_sort"] = aggregated["sum_signed_delta_v"].round(6)
    aggregated = aggregated.sort_values(
        ["_score_abs_sort", "hit_reactions_n", "abs_log2fc"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    aggregated["rank_all"] = np.arange(1, len(aggregated) + 1)
    aggregated["is_deg"] = (aggregated["log2fc"].abs() > 1.0) & (aggregated["qvalue"] < 0.05)
    aggregated["rank_deg"] = np.nan
    deg_indices = aggregated.index[aggregated["is_deg"]]
    aggregated.loc[deg_indices, "rank_deg"] = np.arange(1, len(deg_indices) + 1)
    aggregated["percentile_deg"] = np.nan
    if len(deg_indices):
        aggregated.loc[deg_indices, "percentile_deg"] = 100.0 * (
            1.0 - (aggregated.loc[deg_indices, "rank_deg"] - 1.0) / len(deg_indices)
        )
    aggregated["flux_side"] = np.where(
        aggregated["sum_signed_delta_v"] > 0,
        "positive",
        np.where(aggregated["sum_signed_delta_v"] < 0, "negative", "zero"),
    )
    aggregated["rank_side_deg"] = np.nan
    aggregated["percentile_side_deg"] = np.nan
    for side, ascending in (("positive", False), ("negative", True)):
        side_indices = aggregated.index[aggregated["is_deg"] & (aggregated["flux_side"] == side)]
        ordered = aggregated.loc[side_indices].sort_values(
            ["_score_signed_sort", "hit_reactions_n", "abs_log2fc"],
            ascending=[ascending, False, False],
        )
        ranks = pd.Series(np.arange(1, len(ordered) + 1), index=ordered.index, dtype=float)
        if len(ordered):
            aggregated.loc[ranks.index, "rank_side_deg"] = ranks
            aggregated.loc[ranks.index, "percentile_side_deg"] = 100.0 * (
                1.0 - (ranks - 1.0) / len(ordered)
            )
    aggregated = aggregated.drop(columns=["_score_abs_sort", "_score_signed_sort"])

    active = reactions[reactions["abs_delta_v"] > LEGACY_MODULE.DELTA_V_EPS]
    status_active = status.merge(active[["reaction", "abs_delta_v"]], on="reaction", how="inner")
    total = float(active["abs_delta_v"].sum())
    skipped_flux = float(status_active.loc[status_active["status"] == "skipped", "abs_delta_v"].sum())
    coverage = {
        **statistics,
        "active_reactions": int(len(active)),
        "total_abs_delta_v": total,
        "attributed_fraction": 1.0 - skipped_flux / total if total else np.nan,
        "skipped_reactions_table_rows": int(len(skipped)),
    }
    return aggregated, reactions, coverage


def run_one(comparison: Comparison, scenario: Scenario) -> pd.DataFrame:
    model_path = MODELS / comparison.model_file
    base_a = cobra.io.read_sbml_model(str(model_path))
    base_b = cobra.io.read_sbml_model(str(model_path))
    environment_a = configure_environment(base_a, comparison, scenario)
    environment_b = configure_environment(base_b, comparison, scenario)

    raw_a = read_expression(DATA / comparison.expression_a)
    raw_b = read_expression(DATA / comparison.expression_b)
    transformed_a, transformed_b = transform_pair(raw_a, raw_b, scenario.normalization)
    reaction_a = reaction_expression(base_a, transformed_a, scenario.or_rule)
    reaction_b = reaction_expression(base_b, transformed_b, scenario.or_rule)
    denominator_a, denominator_b = scaling_denominators(
        reaction_a, reaction_b, scenario.normalization
    )

    model_a, scaling_a = apply_expression_constraints(
        base_a, reaction_a, denominator_a, comparison.min_scale
    )
    model_b, scaling_b = apply_expression_constraints(
        base_b, reaction_b, denominator_b, comparison.min_scale
    )
    lp_flux_a, maximum_a, lp_a = solve_biomass_lp(model_a)
    lp_flux_b, maximum_b, lp_b = solve_biomass_lp(model_b)
    if scenario.biomass_mode == "common":
        biomass_a = biomass_b = min(maximum_a, maximum_b)
    elif scenario.biomass_mode == "own":
        biomass_a, biomass_b = maximum_a, maximum_b
    else:
        raise ValueError(f"Unknown biomass mode: {scenario.biomass_mode}")

    biomass_safety_fraction = 0.999
    biomass_a *= biomass_safety_fraction
    biomass_b *= biomass_safety_fraction
    flux_a, qp_a = solve_l2(model_a, biomass_a, lp_flux_a)
    flux_b, qp_b = solve_l2(model_b, biomass_b, lp_flux_b)
    deg = read_deg(DATA / comparison.deg_file)
    ranking, reaction_changes, coverage = build_rankings(
        model_a, flux_a, flux_b, transformed_a, transformed_b, deg
    )

    return ranking


EXPRESSION_LABELS = {
    "max_independent": "Maximum scaling",
    "q99_independent": "99th percentile scaling",
    "log1p_max_independent": "Log-transformation",
    "rank_independent": "Rank-based scaling",
}


def save_results(method: str, records: list[tuple]) -> None:

    RESULTS.mkdir(parents=True, exist_ok=True)
    columns = ["gene", "gene_name", "log2fc", "qvalue", "sum_abs_delta_v",
               "sum_signed_delta_v", "rank_deg"]
    frames = []
    target_rows = {scenario.name: {
        "Flux solution method": method,
        "Expression processing method": EXPRESSION_LABELS[scenario.name],
    } for scenario in SCENARIOS}
    for comparison, scenario, ranking in records:
        label = EXPRESSION_LABELS[scenario.name]
        if ranking is None:
            value = "IF"
        else:
            frame = ranking[columns].copy()
            frame.insert(0, "expression_processing", label)
            frame.insert(0, "flux_objective", method)
            frame.insert(0, "comparison", comparison.name)
            frames.append(frame)
            target = ranking.loc[ranking["gene"] == comparison.target_locus]
            if target.empty or target["rank_deg"].isna().all():
                value = "NP"
            else:
                value = f"{int(target.iloc[0]['rank_deg'])}/{int(ranking['rank_deg'].notna().sum())}"
        target_rows[scenario.name][comparison.target_name] = value

    output = pd.concat(frames, ignore_index=True)
    path = RESULTS / "sensitivity_rankings.tsv"
    if path.exists():
        previous = pd.read_csv(path, sep="\t")
        output = pd.concat([previous.loc[previous["flux_objective"] != method], output], ignore_index=True)

    method_order = {"pFBA": 0, "L2": 1}
    comparison_order = {item.name: i for i, item in enumerate(COMPARISONS)}
    expression_order = {label: i for i, label in enumerate(EXPRESSION_LABELS.values())}
    output = output.sort_values(
        ["flux_objective", "comparison", "expression_processing"],
        key=lambda values: values.map({"flux_objective": method_order,
            "comparison": comparison_order, "expression_processing": expression_order}[values.name]),
        kind="stable",
    )
    output.to_csv(path, sep="\t", index=False)

    summary = pd.DataFrame(target_rows.values())
    path = RESULTS / "target_robustness.tsv"
    if path.exists():
        previous = pd.read_csv(path, sep="\t", keep_default_na=False)
        summary = pd.concat([previous.loc[previous["Flux solution method"] != method], summary], ignore_index=True)
    summary = summary.sort_values(
        ["Flux solution method", "Expression processing method"],
        key=lambda values: values.map(method_order if values.name == "Flux solution method" else expression_order),
        kind="stable",
    )
    summary.to_csv(path, sep="\t", index=False)


def main() -> None:
    records = []
    for comparison in COMPARISONS:
        for scenario in SCENARIOS:
            print(f"[RUN] {comparison.name} / {scenario.name}", flush=True)
            try:
                ranking = run_one(comparison, scenario)
            except RuntimeError as error:

                if "Biomass LP failed:" not in str(error) or "infeasible" not in str(error).lower():
                    raise
                print(f"[IF] {error}", flush=True)
                ranking = None
            records.append((comparison, scenario, ranking))
    save_results("L2", records)


if __name__ == "__main__":
    sys.exit(main())
