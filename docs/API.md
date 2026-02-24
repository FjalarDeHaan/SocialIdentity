# SocialIdentity.jl — Full API Documentation

## Overview

SocialIdentity.jl implements the mathematical formalism for the Social Identity
Approach (de Haan, 2025). The package is organised around three core types and
a set of operations on them.

### Core types

| Type | Represents | Mathematical object |
|------|------------|---------------------|
| `Identity{N,T}` | Individual social identity | Unit vector ψ ∈ Cⁿ |
| `GroupIdentity{N,T}` | Group social identity | Density matrix ρ (N×N) |
| `SocialIdentityNetwork{L,N,T}` | Social network of identifications | Weighted digraph with stochastic adjacency matrix W |

### Aliases

| Alias | Expands to | Notes |
|-------|------------|-------|
| `ψ` | `Identity` | Also callable as `ψ(sin, label)` |
| `SIN` | `SocialIdentityNetwork` | |
| `φ` | `angle_between` | |

---

## `identities.jl` — Identity type

### Type

```julia
struct Identity{N, T <: Number} <: StaticVector{N, T}
```

An individual social identity as a unit vector in Cⁿ. The squared modulus
|ψᵢ|² of each coordinate gives the salience probability of normative dimension
`i`. Relative phases between components encode the integration vs.
compartmentalisation of identity dimensions (synthesis document §3.2).

Type parameters:
- `N::Int` — dimensionality of identity space
- `T <: Number` — element type; `ComplexF64` is the default and recommended

### Constructors

---

#### `Identity(values::NTuple{N,T}) -> Identity{N,T}`

Construct from a tuple. No normalisation applied.

```julia
ψ = Identity((0.6+0im, 0.8+0im))
```

---

#### `Identity(values::AbstractVector{T}; norm::Real=1.0) -> Identity`

Construct from a vector, scaling to the target norm (default: unit norm).

```julia
ψ = Identity([0.6, 0.8, 0.0])          # normalised to unit length
ψ = Identity([1.0, 2.0]; norm=2.0)     # scaled to norm 2
```

---

#### `Identity(n::Int; norm::Real=1.0) -> Identity{n, ComplexF64}`
#### `Identity(rng::AbstractRNG, n::Int; norm::Real=1.0) -> Identity{n, ComplexF64}`

Random complex Identity, Haar-uniform on the unit sphere in Cⁿ. Achieved by
drawing from a standard complex normal distribution and normalising.

```julia
ψ = Identity(4)                          # random, unseeded
ψ = Identity(Xoshiro(42), 4)            # seeded
```

---

#### `Identity(n::Int, ::Type{<:Real}; norm::Real=1.0) -> Identity{n, Float64}`
#### `Identity(rng::AbstractRNG, n::Int, ::Type{<:Real}; norm::Real=1.0)`

Random real Identity, uniform on the unit sphere in Rⁿ.

```julia
ψ = Identity(4, Real)
ψ = Identity(Xoshiro(42), 4, Real)
```

---

#### `Identity(n, biasdir::Identity, biasangle::Real; norm=1.0) -> Identity`
#### `Identity(rng, n, biasdir::Identity, biasangle::Real; norm=1.0) -> Identity`

Random Identity within a bias cone: rejection-sample from the Haar-uniform
distribution, accepting only vectors within `biasangle` radians of `biasdir`.
Corresponds to the shared-bias group construction (Theorem 1, de Haan 2025).

**Warning:** slow for tight cones in high dimensions. Acceptance probability
scales as `biasangle^(n-1)`.

```julia
biasdir = e(1, 4)
ψ = Identity(4, biasdir, π/6)   # within π/6 of biasdir
```

---

### Operations

#### `normalise(ψ::Identity) -> Identity`

Return a unit-norm copy of `ψ`.

---

## `groups.jl` — GroupIdentity type

### Type

```julia
struct GroupIdentity{N, T <: Number, L} <: StaticMatrix{N, N, T}
```

A group social identity as an N×N density matrix over Cⁿ. Valid density
matrices are Hermitian, positive semidefinite, and have unit trace. The `L`
parameter equals N² and is required by StaticArrays; it is handled internally.

### Constructors

---

#### `GroupIdentity(ψ::Identity) -> GroupIdentity`

Pure state: ρ = ψψ†. `purity(ρ) == 1`, `von_neumann_entropy(ρ) == 0`.

```julia
ρ = GroupIdentity(ψ)
```

---

#### `GroupIdentity(weights, identities) -> GroupIdentity`

Mixed state: ρ = Σᵢ wᵢ ψᵢψᵢ†. Weights are normalised automatically and need
only be non-negative.

```julia
ρ = GroupIdentity([0.7, 0.3], [ψ₁, ψ₂])
```

---

#### `GroupIdentity(m::AbstractMatrix; check::Bool=true) -> GroupIdentity`

From a raw matrix. By default checks Hermiticity, unit trace, and positive
semidefiniteness. Pass `check=false` to skip in hot loops.

```julia
ρ = GroupIdentity(some_matrix)             # with checks
ρ = GroupIdentity(some_matrix; check=false) # without checks
```

---

### Invariants enforced on construction (`check=true`)

| Invariant | Tolerance |
|-----------|-----------|
| Hermitian: `ρ ≈ ρ†` | `atol=1e-10` |
| Unit trace: `tr(ρ) ≈ 1` | `atol=1e-10` |
| Positive semidefinite: all eigenvalues ≥ 0 | `atol=1e-10` |

---

## `bases.jl` — Basis vectors and transformations

### Functions

---

#### `e(i::Int, n::Int) -> Identity{n, Float64}`
#### `e(i::Int, ψ::Identity) -> Identity`

The `i`-th standard basis vector in Rⁿ. Represents a pure normative position
along a single dimension.

```julia
e(1, 4)     # [1,0,0,0] as Identity{4, Float64}
e(2, ψ)     # infers dimension from ψ
```

---

#### `standard_basis(n::Int) -> Vector{Identity{n, Float64}}`

All `n` standard basis vectors.

```julia
basis = standard_basis(4)
```

---

#### `orthonormal_basis(vecs::Vector{<:Identity}) -> Vector{<:Identity}`

Gram-Schmidt orthonormalisation. Input vectors must be linearly independent.

```julia
onb = orthonormal_basis([ψ₁, ψ₂, ψ₃])
```

---

#### `represent(ψ::Identity, basis::Vector{<:Identity}) -> Vector{ComplexF64}`

Coordinates of `ψ` in the given orthonormal basis. Implements the
questionnaire-change operation: coordinates against questionnaire A can be
converted to questionnaire B coordinates.

```julia
coords = represent(ψ, standard_basis(4))
```

---

#### `change_basis(ψ, from_basis, to_basis) -> Identity`

Express `ψ` (in `from_basis` coordinates) in `to_basis` coordinates. Both
bases must be orthonormal.

```julia
ψ_new = change_basis(ψ, basis_A, basis_B)
```

---

## `observables.jl` — Pure scalar functions

All functions here are pure (no mutation, no type construction).

### On Identity vectors

---

#### `inner(ψ₁::Identity, ψ₂::Identity) -> Complex`

The Hermitian inner product ⟨ψ₁|ψ₂⟩. The primitive from which all other
observables are derived.

---

#### `overlap(ψ₁::Identity, ψ₂::Identity) -> Float64`

Squared modulus |⟨ψ₁|ψ₂⟩|² ∈ [0, 1]. Identity similarity: 1 = identical
(up to global phase), 0 = orthogonal.

---

#### `salience(ψ::Identity, c::Identity) -> Float64`

Probability that identity `ψ` expresses along context direction `c`:
|⟨c|ψ⟩|² ∈ [0, 1]. Formally identical to `overlap`; distinct in semantics.

**References:** de Haan (2025) §2.2; synthesis document §3.2.

---

#### `angle_between(ψ₁::Identity, ψ₂::Identity) -> Float64`
#### Alias: `φ(ψ₁, ψ₂)`

The angle arccos(|⟨ψ₁|ψ₂⟩|) ∈ [0, π/2]. The fundamental identity comparison.

The absolute value ensures global-phase invariance: `ψ` and `e^{iθ}ψ` have
angle 0. Consequently, antipodal real vectors have angle 0 — consistent with
Framework Choice 4 (opposing identities are distinct dimensions, not sign flips).

**References:** de Haan (2025) §2.2; synthesis document §4.1.

---

### On GroupIdentity

---

#### `purity(ρ::GroupIdentity) -> Float64`

Tr(ρ²) ∈ [1/N, 1]. Pure states: 1. Maximally mixed: 1/N.

---

#### `von_neumann_entropy(ρ::GroupIdentity) -> Float64`

-Tr(ρ log ρ) in nats. Pure states: 0. Maximally mixed: log(N).

**References:** Nielsen & Chuang (2000) §11.3; synthesis document §1.2.

---

#### `alignment(ρ::GroupIdentity) -> Float64`

Largest eigenvalue of ρ. Ranges from 1/N (maximally mixed) to 1 (pure).
Measures how strongly one identity direction dominates the group.

---

## `contexts.jl` — Projection and collapse

A context is a subspace of identity space, represented as a matrix `A` whose
columns span the subspace. Columns of `A` need not be orthonormal; any
full-column-rank matrix is valid. The projection formula P = A(AᵀA)⁻¹Aᵀ
handles the general case.

**References:** de Haan (2025) Definition 5 (collapse), Theorem 3
(polarisation); synthesis document §3.3.

---

#### `project(ψ::Identity, A::AbstractMatrix) -> Vector`
#### `project(ψ::Identity, dims::Union{Int, Vector{Int}}) -> Vector`

Orthogonal projection of `ψ` onto col(A). Returns a plain vector — not
normalised, not an Identity. Use `collapse` to obtain a normalised Identity.

```julia
z = project(ψ, [1, 2])           # project onto dims 1 and 2
z = project(ψ, A)                 # project onto col(A)
```

---

#### `collapse(ψ::Identity, A::AbstractMatrix) -> Identity`
#### `collapse(ψ::Identity, dims::Union{Int, Vector{Int}}) -> Identity`

Orthogonal projection of `ψ` onto col(A) followed by renormalisation. The
projection postulate: the result is the unit vector in col(A) closest to `ψ`.

Raises an error if `ψ` is orthogonal to col(A) (collapse undefined).

**Theorem 3 (de Haan 2025):** collapse onto a proper subspace amplifies all
remaining coordinates — this is the mechanism of polarisation.

```julia
ψ_c = collapse(ψ, [1, 2])   # collapse onto dimensions 1 and 2
ψ_c = collapse(ψ, A)         # collapse onto col(A)
```

---

#### `collapse(ρ::GroupIdentity, A::AbstractMatrix) -> GroupIdentity`

Collapse a mixed state onto col(A). Applies the projection postulate to each
eigenvector, reweights by squared projection norms, renormalises to unit trace.

---

#### `context_sequence(ψ::Identity, contexts::Vector{<:AbstractMatrix}) -> Identity`

Apply a sequence of context collapses in order. Demonstrates order dependence:
`context_sequence(ψ, [A, B]) ≠ context_sequence(ψ, [B, A])` in general.
This is the social analogue of non-commutativity of quantum measurements.

**References:** synthesis document §3.3; Moore (2002) on order effects.

---

## `networks.jl` — SocialIdentityNetwork

### Auxiliary type: `RecurrentClass{L}`

```julia
struct RecurrentClass{L}
    group_identity :: GroupIdentity
    members        :: Vector{L}
    weights        :: Vector{Float64}
end
```

The stable subgroup produced by Markov chain decomposition of a SIN. Fields:
- `group_identity` — the group's density matrix ρ = Σᵢ πᵢ ψᵢψᵢ†
- `members` — vertex labels belonging to this recurrent class
- `weights` — stationary distribution πᵢ (influence of each member)

### SocialIdentityNetwork type

```julia
struct SocialIdentityNetwork{L, N, T <: Number, G}
```

Type parameters:
- `L` — vertex label type (any hashable type)
- `N` — identity space dimension
- `T` — element type (`ComplexF64` by default)
- `G` — internal MetaGraph type (inferred at construction)

### Constructors

---

#### `SocialIdentityNetwork{L,N,T}() -> SocialIdentityNetwork`

Empty SIN with given type parameters.

```julia
sin = SocialIdentityNetwork{Symbol, 4, ComplexF64}()
```

---

#### `SocialIdentityNetwork(W, identities) -> SocialIdentityNetwork{Int,N,T}`
#### `SocialIdentityNetwork(W, identities, labels) -> SocialIdentityNetwork{L,N,T}`

From a stochastic adjacency matrix and Identity vectors. `W[i,j]` is the
weight agent `i` places on agent `j`. `W` must be non-negative and
row-stochastic (rows sum to 1, `atol=1e-8`).

```julia
sin = SocialIdentityNetwork(W, identities)
sin = SocialIdentityNetwork(W, identities, [:alice, :bob, :carol])
```

---

#### `hierarchy(rng, n, teamsize; identity_dim=4) -> SocialIdentityNetwork`

A hierarchical SIN: `n` manager-nodes in a random tree, each with a clique of
`teamsize` team members. Identification flows more strongly up the hierarchy.
Corresponds to Example 1 and Theorem 2 of de Haan (2025).

---

#### `strongly_er(rng, n, m; identity_dim=4) -> SocialIdentityNetwork`

A random strongly-connected SIN via Erdős–Rényi sampling.

---

### Core graph operations

---

#### `add_identity!(sin, label, ψ::Identity)`

Add a new agent. Raises an error if the label already exists.

---

#### `remove_identity!(sin, label)`

Remove an agent and all incident edges.

---

#### `set_identifications!(sin, from_label, to_labels, weights)`

Set all outgoing identification weights from `from_label`. Weights are
normalised automatically. Any existing outgoing edges not in `to_labels` are
removed. This is the only public API for setting edge weights, ensuring
row-stochasticity is always maintained.

```julia
set_identifications!(sin, :alice, [:alice, :bob], [0.3, 0.7])
```

---

#### `normalise!(sin, label)`

Renormalise outgoing edges of `label` to sum to 1. Use after constructing a
SIN externally where weights are valid but may not sum exactly to 1.

---

#### `identity_of(sin, label) -> Identity`

Retrieve the Identity vector of an agent. Also callable as `ψ(sin, label)`.

---

#### `weight(sin, from_label, to_label) -> Float64`

Return the identification weight from `from_label` to `to_label`, or 0.0 if
no edge exists.

---

#### `labels(sin) -> Vector`

All vertex labels in the SIN.

---

#### `n_agents(sin) -> Int`

Number of agents in the SIN.

---

### DeGroot dynamics

---

#### `stochastic_matrix(sin) -> Matrix{Float64}`

The adjacency matrix W. Entry `W[i,j]` is the weight agent `i` places on
agent `j`. Rows sum to 1.

**References:** de Haan (2025) §2.3.

---

#### `degroot(sin, t::Int) -> Matrix{Float64}`

The t-step update matrix Wᵗ. Entry (i,j) gives agent i's weight on agent j
after `t` rounds of social learning.

**References:** DeGroot (1974); Golub & Jackson (2010).

---

#### `stationary(sin) -> Vector{RecurrentClass{L}}`

Decompose the SIN into recurrent classes and compute stationary distributions.
Returns one `RecurrentClass` per recurrent class. Transient agents do not
appear in any class.

**Algorithm:** strongly connected components → condensation DAG → recurrent
SCCs → left eigenvector of sub-matrix with eigenvalue 1 → GroupIdentity.

**References:** de Haan (2025) §3.3, Theorem 2; Bertsekas & Tsitsiklis (2002) §4.

---

#### `group_identity(sin) -> Vector{GroupIdentity}`

Convenience wrapper: returns just the `GroupIdentity` per recurrent class.

---

#### `influence(sin, label) -> Float64`

Stationary distribution weight of `label` in its recurrent class. Transient
agents return 0.0. Measures the fraction of group identity determined by this
agent.

**References:** de Haan (2025) §3.3, Theorem 2 (influence declines
geometrically with depth in a hierarchy).

---

## Design notes

### Dependency order

Files are included in the following order in `SocialIdentity.jl`:

```
identities.jl → groups.jl → bases.jl → observables.jl → contexts.jl → networks.jl
```

`observables.jl` and `bases.jl` have no dependencies on other package files.
`contexts.jl` depends on `identities.jl` and `groups.jl`. `networks.jl`
depends on all preceding files.

### Row-stochasticity invariant

The SIN maintains row-stochasticity through a restricted mutation API:
`set_identifications!` (normalises automatically) and `normalise!` (explicit
renormalisation). There is no single-edge setter — this eliminates the
possibility of leaving the matrix in a non-stochastic state through the public
API.

### Complex vs. real identities

Complex identities (`T = ComplexF64`) are the default and recommended choice.
The relative phase between components is an empirically testable quantity
(synthesis document §3.4): the unprimed vs. primed condition in an oblique
context experiment distinguishes the complex model from the real model.
Real identities (`T = Float64`) are available as a special case by passing
`Real` to the random constructors.

### `purity` duplication

`purity` and `von_neumann_entropy` appear in both `groups.jl` (as methods on
`GroupIdentity`) and `observables.jl`. The `observables.jl` versions are the
exported API; the `groups.jl` versions are used internally by the display code.

### Dynamic extension

`dynamics.jl` is reserved for future continuous-time and non-linear dynamics:
the Lohe sphere model (Lohe 2009), Kuramoto generalisation (Markdahl et al.
2018), and Fréchet mean iteration (Thunberg et al. 2018). DeGroot iteration
and W∞ live in `networks.jl` as they are intrinsic to the SIN type.
