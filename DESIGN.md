# Arbitrage Free Volatility Surface Construction from Noisy Option Quotes

### Relationship to README

This document complements [`README.md`](README.md).

- `README.md` describes *what* the system does and *how* to use it at a high level.
- `DESIGN.md` explains *why* the system is structured the way it is, with emphasis on invariants, correctness, and failure modes.

Readers evaluating architectural decisions or correctness guarantees should start here before reading the implementation.

## Problem Statement

Market option quotes are noisy, incomplete and frequently violate known no-arbitrage constraints. In particular:
- Call and put quotes may violate put-call parity due to liquidity asymmetry
- Bid-ask spreads vary widely across strikes and maturities
- Static arbitrage (vertical, butterfly, calendar) is observed in raw data

The goal of this project is to construct a consistent implied volatility surface that:
1. Remains faithful to reliable market prices
2. Explicitly enforces known no-arbitrage constraints
3. Is stable across strikes and maturities
4. Is independent of any signle volatility model

## Design Principles

This project follows five principles:
1. **Invariants before models** - Arbitrage constraints are enforced explicitly and independently of any parametric model
2. **Separation of concerns** - Price inference, constraint enforcement, and surface modelling are isolated layers
3. **Minimal repairs** - When violations occur, the smallest possible local adjustment is applied
4. **Auditability** - Every inferred implied volatility is annotated with its provenance
5. **Determinism and testability** - Each transformation is deterministic and unit-testable in isolation

## Core Invariants

The system enforces following invariants:

### 1. Put-Call Parity

For European options:

$$ C - P = Se^{-qT} - Ke^{-rT} $$

Used to:
- Reconstruct put prices from liquid calls (or vice versa)
- Resolve asymmetric liquidity across option sides

### 2. Vertical Spread Arbitrage

Call prices must be non-increasing in strike:

$$ \frac{\partial C}{\partial K} \le 0 $$

Violations imply negative call spreads and are repaired by monotic projection

### 3. Butterfly Arbitrage (Convexity)

Call prices must be conves in strike:

$$ \frac{\partial^2 C}{\partial K^2} \ge 0 $$

Equivalent to a non-negative risk neutral density. Enforced via convexity repair on total variance.

### 4. Calendar Arbitrage

Total variance must be non-decreasing in maturity:

$$ w(k,T)=\sigma^2(k,T)T \\quad \text{ is non-decreasing in } T $$

Enforced across maturities on a common log-moneyness grid

## System Architecture

The sytem is decomposed into three layers:

```css
Raw Quotes
   ↓
[IV Inference Layer]
   ↓
[Arbitrage Enforcement Layer]
   ↓
[Surface Representation Layer]
```

## IV Inference Layer

### Responsibilities

- Convert option prices into implied volatilities
- Handle noisy or illiquid quotes

### Strategy

For each option:
1. Use market mid-price if quote passes liquidity checks
2. If illiquid and option is a put, reconstruct price via parity from call
3. If neither is reliable, fall back to reference IV data

Each IV is tagged with:
- `market_mid`
- `parity_from_call_market`
- `dataset_fallback`

## Arbitrage Enforcement Layer

See “Arbitrage Enforcement” in [`README.md`](README.md) for a high-level overview of how these constraints are applied in practice.

### Representation Choice

Arbitrage constraints are enforced on total variance:

$$ w(k,T)=\sigma^2(k,T)T $$

This simplifies:
- Convexity enforcement (butterfly)
- Calendar monotonicity

### Repairs

- **Veritcal:** monotone projection of prices
- **Butterfly:** local convexity repair on total variance
- **Calendar:** monotone adjustment across maturities at fixed log-moneyness

Repairs are applied in the following order:
1. Calendar
2. Butterfly
3. Vertical

## Surface Representation Layer

Surface models consume already arbitrage-free inputs

Supported representations:
- Non-parametric spline on total variance
- SVI parameterization
- SABR (Hagan approximation)
- Heston (Fourier/COS pricing)

Models are interchangeable and evaluated independently.

## Testing Strategy

Each invariant is tested independently:
- Parity reconstruction tests
- Vertical monotonicity tests
- Convexity tests
- Calendar monotonicity tests

Synthetic surfaces are used to verify that:
- Repairs remove violations
- No new violations are introduced

## Non-Goals

This project explicitly does not aim to:
- Predict future volatility
- Optimize trading strategies
- Claim superior model performance

The focus is correctness and consistency, not alpha.

## Extensions

Future extenstions include:
- Local volatility extraction (Dupire)
- Jump-diffusion models
- Probabilistic surface regularization
- Real-time surface updates
