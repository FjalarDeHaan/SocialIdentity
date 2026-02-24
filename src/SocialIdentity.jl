"""
    SocialIdentity

A Julia package implementing the mathematical formalism for the Social Identity
Approach (de Haan, 2025). Individual identities are unit vectors in Cⁿ; group
identities are density matrices; social identity networks are weighted directed
graphs with row-stochastic adjacency matrices.

# Main Types
- `Identity{N,T}` — individual social identity as a unit vector in Cⁿ
- `GroupIdentity{N,T}` — group identity as a density matrix (N×N)
- `SocialIdentityNetwork{L,N,T}` — weighted directed graph of identifications
- `RecurrentClass{L}` — stable subgroup from Markov decomposition of a SIN

# Aliases
- `ψ` — alias for `Identity` and for `identity_of(sin, label)`
- `SIN` — alias for `SocialIdentityNetwork`
- `φ` — alias for `angle_between`

# References
- de Haan, F.J. (2025). Towards a Formalism for the Social Identity Approach.
  In: Advances in Social Simulation, Springer Proceedings in Complexity.
- Synthesis document: Identity, Density Matrices, and Quantum Cognition (2026).
"""
module SocialIdentity

using StaticArrays
using Graphs
using MetaGraphsNext
using LinearAlgebra
using Random
using Printf

# ── Source files ───────────────────────────────────────────────────────────────

include("identities.jl")
include("groups.jl")
include("bases.jl")
include("observables.jl")
include("contexts.jl")
include("networks.jl")

# ── Aliases ────────────────────────────────────────────────────────────────────

"""
    ψ

Alias for `Identity`. Also callable as `ψ(sin, label)` to retrieve the
Identity vector of a labelled agent in a SocialIdentityNetwork.
"""
const ψ = Identity

# Overload ψ as an accessor for identity_of
(::Type{Identity})(sin::SocialIdentityNetwork, label) = identity_of(sin, label)

"""
    SIN

Alias for `SocialIdentityNetwork`.
"""
const SIN = SocialIdentityNetwork

"""
    φ

Alias for `angle_between`. The angle between two identity vectors in [0, π/2].
"""
const φ = angle_between

# ── Exports ────────────────────────────────────────────────────────────────────

# Types
export Identity, GroupIdentity, SocialIdentityNetwork, RecurrentClass

# Aliases
export ψ, SIN, φ

# identities.jl
export normalise

# groups.jl
export purity, von_neumann_entropy, alignment

# bases.jl
export e, standard_basis, orthonormal_basis, represent, change_basis

# observables.jl
export inner, overlap, salience, angle_between

# contexts.jl
export project, collapse, context_sequence

# networks.jl
export add_identity!, remove_identity!, set_identifications!, normalise!
export identity_of, weight, labels, n_agents
export stochastic_matrix, degroot, stationary, group_identity, influence
export hierarchy, strongly_er

end # module SocialIdentity
