from __future__ import annotations

import sys

import cobra
import numpy as np
import pandas as pd
from cobra.flux_analysis import pfba

import run_robustness as core


def solve_pfba_at_common_biomass(
    model: cobra.Model,
    maximum_biomass: float,
    common_biomass: float,
) -> tuple[pd.Series, dict]:
    fraction = common_biomass / maximum_biomass
    solution = pfba(model, fraction_of_optimum=fraction)
    if solution.status != "optimal":
        raise RuntimeError(f"pFBA failed: {solution.status}")
    flux = solution.fluxes.reindex([reaction.id for reaction in model.reactions]).astype(float)
    stoichiometry = core.stoichiometric_matrix(model)
    objective = core.objective_vector(model)
    observed_biomass = float(objective @ flux.to_numpy())
    mass_balance = stoichiometry @ flux.to_numpy()
    lower = np.asarray([reaction.lower_bound for reaction in model.reactions], dtype=float)
    upper = np.asarray([reaction.upper_bound for reaction in model.reactions], dtype=float)
    metadata = {
        "solver_status": solution.status,
        "fraction_of_condition_optimum": fraction,
        "requested_minimum_biomass": common_biomass,
        "observed_biomass": observed_biomass,
        "max_mass_balance_residual": float(np.max(np.abs(mass_balance))),
        "max_lower_bound_violation": float(
            np.max(np.maximum(lower - flux.to_numpy(), 0.0))
        ),
        "max_upper_bound_violation": float(
            np.max(np.maximum(flux.to_numpy() - upper, 0.0))
        ),
    }
    return flux, metadata


def run_one(comparison: core.Comparison, scenario: core.Scenario) -> pd.DataFrame:
    model_path = core.MODELS / comparison.model_file
    base_a = cobra.io.read_sbml_model(str(model_path))
    base_b = cobra.io.read_sbml_model(str(model_path))
    environment_a = core.configure_environment(base_a, comparison, scenario)
    environment_b = core.configure_environment(base_b, comparison, scenario)

    raw_a = core.read_expression(core.DATA / comparison.expression_a)
    raw_b = core.read_expression(core.DATA / comparison.expression_b)
    transformed_a, transformed_b = core.transform_pair(raw_a, raw_b, scenario.normalization)
    reaction_a = core.reaction_expression(base_a, transformed_a, scenario.or_rule)
    reaction_b = core.reaction_expression(base_b, transformed_b, scenario.or_rule)
    denominator_a, denominator_b = core.scaling_denominators(
        reaction_a, reaction_b, scenario.normalization
    )
    model_a, scaling_a = core.apply_expression_constraints(
        base_a, reaction_a, denominator_a, comparison.min_scale
    )
    model_b, scaling_b = core.apply_expression_constraints(
        base_b, reaction_b, denominator_b, comparison.min_scale
    )
    _, maximum_a, lp_a = core.solve_biomass_lp(model_a)
    _, maximum_b, lp_b = core.solve_biomass_lp(model_b)
    common_biomass = 0.999 * min(maximum_a, maximum_b)
    flux_a, pfba_a = solve_pfba_at_common_biomass(model_a, maximum_a, common_biomass)
    flux_b, pfba_b = solve_pfba_at_common_biomass(model_b, maximum_b, common_biomass)

    deg = core.read_deg(core.DATA / comparison.deg_file)
    ranking, reaction_changes, coverage = core.build_rankings(
        model_a, flux_a, flux_b, transformed_a, transformed_b, deg
    )
    return ranking


def main() -> int:
    records = []
    for comparison in core.COMPARISONS:
        for scenario in core.SCENARIOS:
            print(f"[RUN] {comparison.name} / {scenario.name}", flush=True)
            try:
                ranking = run_one(comparison, scenario)
            except RuntimeError as error:
                if "Biomass LP failed:" not in str(error) or "infeasible" not in str(error).lower():
                    raise
                print(f"[IF] {error}", flush=True)
                ranking = None
            records.append((comparison, scenario, ranking))
    core.save_results("pFBA", records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
