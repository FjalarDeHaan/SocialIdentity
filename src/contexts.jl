# contexts.jl
#
# Projection and collapse operations on Identity vectors and GroupIdentity
# matrices. A context is a subspace of identity space; placing an identity
# in a context collapses it onto that subspace via orthogonal projection
# followed by renormalisation (the projection postulate).
#
# References:
#   de Haan (2025), Definition 5 (collapse) and Theorem 3 (polarisation).
#   Synthesis document §3.3: The Projection Postulate and Social Contexts.
#   Moore (2002): order effects in survey contexts.

# ── Projection ─────────────────────────────────────────────────────────────────

"""
    project(ψ::Identity, A::AbstractMatrix) -> Vector

Orthogonally project the identity vector `ψ` onto the subspace spanned by
the columns of `A`. Returns a plain vector in the original space — not
normalised, and not an Identity.

The columns of `A` need not be orthonormal; any full-column-rank matrix
spanning the desired subspace is valid. The projection is computed using the
standard formula:

    P = A (AᵀA)⁻¹ Aᵀ,    z = P ψ

Use `collapse` to obtain a normalised Identity after projection.

# Errors
Raises an error if `A` is not full column rank (columns are linearly
dependent).
"""
function project(ψ::Identity, A::AbstractMatrix)
    size(A, 1) == length(ψ) ||
        throw(DimensionMismatch(
            "A has $(size(A,1)) rows but ψ has dimension $(length(ψ))"))
    # Projection matrix: P = A (AᵀA)⁻¹ Aᵀ
    # Applied directly: z = A (AᵀA)⁻¹ Aᵀ ψ
    AᵀA = A' * A
    rank_deficient = abs(det(AᵀA)) < 1e-12
    rank_deficient && throw(ArgumentError(
        "columns of A are linearly dependent (det(AᵀA) ≈ 0)"))
    return A * (AᵀA \ (A' * Vector{ComplexF64}(ψ)))
end

"""
    project(ψ::Identity, dims::Union{Int, Vector{Int}}) -> Vector

Project `ψ` onto the coordinate subspace spanned by the standard basis
vectors at the listed dimension indices. Convenience wrapper around
`project(ψ, A)` where `A` is constructed from standard basis vectors.

# Examples
```julia
project(ψ, 1)        # project onto dimension 1
project(ψ, [1, 3])   # project onto the subspace of dimensions 1 and 3
```
"""
function project(ψ::Identity{N}, dims::Union{Int, Vector{Int}}) where N
    ds = dims isa Int ? [dims] : dims
    A  = hcat([Vector{Float64}(e(i, N)) for i in ds]...)
    return project(ψ, A)
end

# ── Collapse ───────────────────────────────────────────────────────────────────

"""
    collapse(ψ::Identity, A::AbstractMatrix) -> Identity

Collapse the identity vector `ψ` onto the subspace spanned by the columns
of `A`: orthogonal projection followed by renormalisation. This is the
projection postulate of quantum mechanics applied to social identity
(Definition 5, de Haan 2025).

The resulting Identity is the unit vector in col(A) closest to ψ in the
sense of minimising the angle between them.

The columns of `A` need not be orthonormal — any full-column-rank matrix
spanning the desired subspace is valid.

!!! note
    If `ψ` is orthogonal to col(A), the projection is the zero vector and
    renormalisation is undefined. An error is raised in this case.

# References
- de Haan (2025), Definition 5 and Theorem 3 (polarisation follows from collapse).
- Synthesis document §3.3.
"""
function collapse(ψ::Identity, A::AbstractMatrix)
    z = project(ψ, A)
    n = LinearAlgebra.norm(z)
    n < 1e-12 && throw(ArgumentError(
        "ψ is orthogonal to the context subspace; collapse is undefined"))
    return Identity(z; norm=1.0)
end

"""
    collapse(ψ::Identity, dims::Union{Int, Vector{Int}}) -> Identity

Collapse `ψ` onto the coordinate subspace spanned by the standard basis
vectors at the listed dimension indices, then renormalise.

This is the simplest form of contextualisation: restricting identity to a
subset of normative dimensions and renormalising to a unit vector. Theorem 3
of de Haan (2025) shows this produces polarisation — the remaining coordinates
are amplified.

# Examples
```julia
collapse(ψ, [1, 3])   # collapse onto dimensions 1 and 3
collapse(ψ, 2)        # collapse onto dimension 2 alone
```

# References
- de Haan (2025), Theorem 3.
- Synthesis document §3.3, Example 2 (cat-jazz polarisation).
"""
function collapse(ψ::Identity{N}, dims::Union{Int, Vector{Int}}) where N
    ds = dims isa Int ? [dims] : dims
    A  = hcat([Vector{Float64}(e(i, N)) for i in ds]...)
    return collapse(ψ, A)
end

"""
    collapse(ρ::GroupIdentity, A::AbstractMatrix) -> GroupIdentity

Collapse a mixed-state group identity onto the subspace spanned by the
columns of `A`. Applies the projection postulate to each eigenvector of ρ,
reweights by the squared norm of each projection (which differs across
eigenvectors), and renormalises to unit trace.

!!! note
    Collapsing a mixed state is not simply collapsing each eigenvector
    independently with equal weights. The projection changes the effective
    mixture weights in proportion to |⟨projected|original⟩|², consistent
    with the Born rule for mixed states.
"""
function collapse(ρ::GroupIdentity{N,T}, A::AbstractMatrix) where {N,T}
    h      = Hermitian(Matrix(ρ))
    λs     = real.(eigvals(h))
    vs     = eigvecs(h)
    active = findall(>(1e-15), λs)

    new_weights = Float64[]
    new_idents  = Identity[]

    for idx in active
        ψᵢ = Identity(vs[:, idx]; norm=1.0)
        z  = project(ψᵢ, A)
        n  = LinearAlgebra.norm(z)
        n < 1e-12 && continue   # this eigenvector is orthogonal to context
        push!(new_weights, λs[idx] * n^2)
        push!(new_idents,  Identity(z; norm=1.0))
    end

    isempty(new_weights) && throw(ArgumentError(
        "all eigenvectors are orthogonal to the context subspace"))

    # Call the weighted mixture constructor explicitly by converting to
    # concrete typed vectors, bypassing StaticArrays constructor interception.
    return GroupIdentity(
        convert(Vector{Float64}, new_weights),
        convert(Vector{Identity{N,T}}, new_idents),
    )
end

# ── Context sequences ──────────────────────────────────────────────────────────

"""
    context_sequence(ψ::Identity, contexts::Vector{<:AbstractMatrix}) -> Identity

Apply a sequence of context collapses to `ψ` in order. Each element of
`contexts` is a matrix whose columns span a context subspace.

Demonstrates order dependence: `context_sequence(ψ, [A, B])` differs in
general from `context_sequence(ψ, [B, A])` because collapse is
non-commutative. This is the social analogue of the non-commutativity of
quantum measurements and the mechanism behind order effects observed in
survey contexts (Moore 2002).

# References
- Synthesis document §3.3.
- Moore, D.W. (2002). Measuring new types of question-order effects.
  Public Opinion Quarterly, 66(1), 80–91.
"""
function context_sequence(ψ::Identity, contexts::Vector{<:AbstractMatrix})
    result = ψ
    for A in contexts
        result = collapse(result, A)
    end
    return result
end
