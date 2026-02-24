# SocialIdentity.jl

A Julia package implementing the mathematical formalism for the **Social Identity Approach** — a family of social-psychological theories explaining inter- and intra-group dynamics.

The formalism represents individual social identities as unit vectors in a complex vector space Cⁿ, group identities as density matrices, and the social network of identification as a weighted directed graph with a stochastic adjacency matrix. Group identities emerge from DeGroot social learning on this network.

## Background

The package implements the formalism introduced in:

> de Haan, F.J. (2025). *Towards a Formalism for the Social Identity Approach*. In: Advances in Social Simulation, Springer Proceedings in Complexity.

and extended in the companion synthesis document *Identity, Density Matrices, and Quantum Cognition* (2026), which grounds the framework in quantum information theory and quantum cognition, and motivates the use of complex amplitudes and density matrices.

Key framework choices:
- **Individual identities** are unit vectors ψ ∈ Cⁿ (pure states). The squared modulus |ψᵢ|² of each coordinate is the salience probability of normative dimension i. Relative phases encode the integration vs. compartmentalisation of identity dimensions.
- **Group identities** are density matrices ρ = Σ wᵢ ψᵢψᵢ†, the convex mixture of pure states weighted by the stationary distribution of the social network. This is closed under convex combination, linear, and distinguishes coexistence of opposing identities from their cancellation.
- **Social identity networks** are weighted directed graphs with stochastic adjacency matrices. Each agent distributes one unit of identification weight across those they identify with. Self-loops represent self-identification.
- **Opposing identities** (e.g. cat-lover vs. cat-hater) are treated as distinct dimensions, not as sign-reversed versions of a single axis. This eliminates the anti-identity problem and is consistent with empirical findings in affective psychology.

## Installation

```julia
] add https://github.com/fjalar/SocialIdentity.jl
```

Or in development mode from a local clone:

```julia
] dev path/to/SocialIdentity.jl
```

## Quick Start

```julia
using SocialIdentity

# ── Individual identities ──────────────────────────────────────────────────────

# Random 4-dimensional complex identity (Haar-uniform on unit sphere in C⁴)
ψ₁ = Identity(4)

# From a vector
ψ₂ = Identity([0.6, 0.8, 0.0, 0.0])   # normalised automatically

# Real-valued identity
ψ₃ = Identity(4, Real)

# With bias (for constructing shared-bias groups, Theorem 1 de Haan 2025)
biasdir = e(1, 4)   # first standard basis vector
ψ_biased = Identity(4, biasdir, π/6)   # within π/6 of biasdir

# ── Group identities ───────────────────────────────────────────────────────────

# Pure state (single coherent identity)
ρ_pure = GroupIdentity(ψ₁)

# Mixed state (ensemble)
ρ_mixed = GroupIdentity([0.7, 0.3], [ψ₁, ψ₂])

purity(ρ_mixed)              # ∈ [1/N, 1]
von_neumann_entropy(ρ_mixed) # in nats

# ── Social identity network ────────────────────────────────────────────────────

sin = SocialIdentityNetwork{Symbol, 4, ComplexF64}()

add_identity!(sin, :alice, Identity(4))
add_identity!(sin, :bob,   Identity(4))
add_identity!(sin, :carol, Identity(4))

# Alice identifies 60% with herself, 40% with Bob
set_identifications!(sin, :alice, [:alice, :bob], [0.6, 0.4])
# Bob identifies 80% with Carol, 20% with himself
set_identifications!(sin, :bob,   [:carol, :bob], [0.8, 0.2])
# Carol is self-identifying (fully autonomous identity)
set_identifications!(sin, :carol, [:carol], [1.0])

# DeGroot dynamics
W  = stochastic_matrix(sin)
W5 = degroot(sin, 5)         # W^5

# Stable group identities
rcs = stationary(sin)        # Vector{RecurrentClass}
rcs[1].group_identity        # GroupIdentity of the recurrent class
rcs[1].members               # [:carol]
rcs[1].weights               # [1.0]

influence(sin, :carol)       # 1.0
influence(sin, :alice)       # 0.0 (transient)

# ── Observables ────────────────────────────────────────────────────────────────

angle_between(ψ₁, ψ₂)   # ∈ [0, π/2]; fundamental comparison (also: φ(ψ₁, ψ₂))
overlap(ψ₁, ψ₂)         # |⟨ψ₁|ψ₂⟩|² ∈ [0, 1]
salience(ψ₁, ψ₂)        # probability of ψ₁ expressing along direction ψ₂

# ── Contexts (projection postulate) ───────────────────────────────────────────

# Collapse onto a 2-dimensional subspace (dimensions 1 and 2)
ψ_collapsed = collapse(ψ₁, [1, 2])   # Definition 5, de Haan 2025

# Collapse onto an arbitrary subspace
A = [1.0 0.0; 0.0 1.0; 0.0 0.0; 0.0 0.0]   # 4×2 matrix
ψ_c = collapse(ψ₁, A)

# Order-dependent context sequence (non-commutativity)
A1 = [1.0 0.0; 0.0 1.0; 0.0 0.0; 0.0 0.0]
A2 = [1.0 0.0; 0.0 0.0; 0.0 1.0; 0.0 0.0]
ψ_AB = context_sequence(ψ₁, [A1, A2])
ψ_BA = context_sequence(ψ₁, [A2, A1])
# In general: ψ_AB ≠ ψ_BA (order effects, Moore 2002)

# ── Bases ──────────────────────────────────────────────────────────────────────

e(1, 4)              # first standard basis vector in C⁴
standard_basis(4)    # all four basis vectors
represent(ψ₁, standard_basis(4))   # coordinates of ψ₁ in standard basis
```

## Package Structure

```
src/
  SocialIdentity.jl   # module entry point, exports, aliases
  identities.jl       # Identity{N,T} type
  groups.jl           # GroupIdentity{N,T} type
  bases.jl            # standard basis vectors, Gram-Schmidt, basis change
  observables.jl      # pure scalar functions (angle, overlap, entropy, ...)
  contexts.jl         # project, collapse, context_sequence
  networks.jl         # SocialIdentityNetwork, RecurrentClass, DeGroot
```

## Aliases

| Alias | Full name | Notes |
|-------|-----------|-------|
| `ψ`   | `Identity` | Also callable as `ψ(sin, label)` to retrieve an agent's identity |
| `SIN` | `SocialIdentityNetwork` | |
| `φ`   | `angle_between` | Angle between two identities ∈ [0, π/2] |

## Dependencies

- [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) — fixed-size array types
- [Graphs.jl](https://github.com/JuliaGraphs/Graphs.jl) — graph algorithms
- [MetaGraphsNext.jl](https://github.com/JuliaGraphs/MetaGraphsNext.jl) — graphs with vertex/edge metadata
- [LinearAlgebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/) — standard library

## References

- de Haan, F.J. (2025). Towards a Formalism for the Social Identity Approach. *Advances in Social Simulation*, Springer.
- DeGroot, M.H. (1974). Reaching a consensus. *JASA*, 69(345), 118–121.
- Golub, B. & Jackson, M.O. (2010). Naïve learning in social networks. *AEJ: Microeconomics*, 2(1), 112–149.
- Busemeyer, J.R. & Bruza, P.D. (2012). *Quantum Models of Cognition and Decision*. Cambridge University Press.
- Watson, D., Clark, L.A. & Tellegen, A. (1988). Development and validation of brief measures of positive and negative affect: the PANAS scales. *JPSP*, 54(6), 1063–1070.
- Nielsen, M.A. & Chuang, I.L. (2000). *Quantum Computation and Quantum Information*. Cambridge University Press.

## Licence

GPL v3
