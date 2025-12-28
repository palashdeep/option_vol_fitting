# Project: Arbitrage-Free Volatility Surface Construciton

This project builds a consistent implied volatility surface from noisy option quotes while enforcing known no-arbitrage constraints. The emphasis is on correctness, invairants, and auditability, rather than any single volatility model.

Design rationale and invariants are documented in [`DESIGN.md`](DESIGN.md)

## Motivation

Raw option quotes:
- violate put-call parity
- admit static arbitrage
- are often illiquid or incomplete

This system recovers a surface that:
- respects market prices when reliable
- enforces arbitrage constraints explicitly
- remains stable across strikes and maturities

## Approach

The pipeline consists of three layers:

1. IV inference
    - Market mid-price inversion
    - Parity-based reconstruction
    - Dataset fallback for illiquid quotes

2. Arbitrage Enforcement
    - Vertical monotonicity
    - Butterfly convexity
    - Calendar consistency (total variance)

3. Surface Modelling
    - Spline
    - SVI
    - SABR
    - Heston

All models operate on arbitrage-free inputs.

### Arbitrage Enforcement

The system enforces vertical, butterfly, and calendar arbitrage constraints explicitly.
The mathematical formulation and enforcement order are detailed in §3–§6 of [`DESIGN.md`](DESIGN.md).

## Design & Invariants

This project is intentionally organized around explicit invariants rather than around any single volatility model.

The full design rationale including the choice of representations, the ordering of arbitrage repairs, and the separation between inference, constraints, and modeling is documented in [`DESIGN.md`](DESIGN.md).

Readers interested in *why* particular constraints are enforced, *where* they are applied in the pipeline, and *how* edge cases are handled should start there.

## Key Features
- Hybrid implied volatility inference with provenance tracking
- Explicitly enforcement of static arbitrage constraints
- Cross-expiry consistency via calendar monotonicity
- Modular, testable design
- Model-agnostic surface construction

## Repository Structure
```graphql
core/        # invariant-driven logic
models/      # surface parametrizations
research/    # exploratory analysis; not part of invariant core
tests/       # invariant tests
scripts/     # orchestration
```

## Intended Use

This project is designed for:
- derivative pricing research
- volatility modeling
- risk and surface consistency analysis

It is not a trading strategy.

A short demonstration notebook is available in `research/`.
