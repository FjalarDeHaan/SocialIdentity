# groups.jl
#
# Defines the GroupIdentity type: an N×N density matrix over Cⁿ representing
# a group social identity as a probability distribution over pure states.
#
# References:
#   Synthesis document §1, §2.2 (Framework Choice 2).
#   de Haan (2025), §3: group identity as W∞ applied to identity vectors.

# ── Type definition ────────────────────────────────────────────────────────────

"""
    GroupIdentity{N, T <: Number, L} <: StaticMatrix{N, N, T}

A group social identity represented as an N×N density matrix over Cⁿ.
A valid density matrix is Hermitian, positive semidefinite, and has unit trace.

Pure states (a single coherent group identity) have `purity == 1` and
correspond to a single Identity vector via `ρ = ψψ†`. Mixed states represent
a classical probability distribution over pure states — an ensemble of identity
vectors with associated weights — arising naturally as the group identity from
DeGroot social learning on a SocialIdentityNetwork.

The `L = N²` parameter is required by StaticArrays for fixed-size matrix
storage and should not normally be set by the caller; use the provided
constructors.

# References
- Synthesis document §1: The Geometry of Density Matrices.
- Synthesis document §2.2: Framework Choice 2 (density matrices preferred).
- de Haan (2025), §3: Emergence of Group Identities.
"""
struct GroupIdentity{N, T <: Number, L} <: StaticMatrix{N, N, T}
    values::NTuple{L, T}

    function GroupIdentity{N,T,L}(values::NTuple{L,T}) where {N, T <: Number, L}
        L == N * N || throw(ArgumentError("L must equal N²=$(N*N), got $L"))
        new{N,T,L}(values)
    end
end

# Hide the L parameter from callers
GroupIdentity{N,T}(values::NTuple{L,T}) where {N, T <: Number, L} =
    GroupIdentity{N,T,L}(values)

# ── StaticArrays interface ─────────────────────────────────────────────────────

StaticArrays.Size(::Type{GroupIdentity{N,T,L}}) where {N,T,L} = Size(N, N)

@inline function Base.getindex(ρ::GroupIdentity, i::Int)
    @boundscheck checkbounds(ρ, i)
    ρ.values[i]
end

StaticArrays.similar_type(::Type{<:GroupIdentity}, ::Type{T}, ::Size{S}) where {T,S} =
    GroupIdentity{S[1], T, S[1]*S[2]}

# ── Invariant checking ─────────────────────────────────────────────────────────

"""
    _check_density_matrix(m::AbstractMatrix)

Internal. Check that `m` satisfies the three density matrix invariants:
Hermiticity, unit trace, and positive semidefiniteness. Raises an
`ArgumentError` with a descriptive message if any invariant is violated.

The tolerance `atol=1e-10` is tight enough to catch real errors while
tolerating accumulated floating-point noise from typical operations.
"""
function _check_density_matrix(m::AbstractMatrix)
    n, k = size(m)
    n == k ||
        throw(ArgumentError("density matrix must be square, got $(n)×$(k)"))
    isapprox(tr(m), 1.0, atol=1e-10) ||
        throw(ArgumentError("trace is $(tr(m)), expected 1"))
    isapprox(m, m', atol=1e-10) ||
        throw(ArgumentError("matrix is not Hermitian"))
    all(≥(-1e-10), eigvals(Hermitian(Matrix(m)))) ||
        throw(ArgumentError("matrix is not positive semidefinite"))
end

# ── Constructors ───────────────────────────────────────────────────────────────

"""
    GroupIdentity(ψ::Identity) -> GroupIdentity

Construct the pure-state density matrix ρ = ψψ†. This represents a perfectly
coherent group: every group member shares the same identity vector. The
resulting matrix has `purity == 1` and `von_neumann_entropy == 0`.
"""
function GroupIdentity(ψ::Identity{N,T}) where {N, T <: Number}
    m = ψ * ψ'
    return GroupIdentity{N,T,N*N}(ntuple(i -> m[i], Val(N*N)))
end

"""
    GroupIdentity( weights::AbstractVector{<:Real}
                 , identities::AbstractVector{<:Identity{N,T}}
                 ) -> GroupIdentity{N,T}

Construct a mixed-state density matrix as the weighted mixture:

    ρ = Σᵢ wᵢ ψᵢψᵢ†

Weights are normalised to sum to 1 automatically and need only be
non-negative. This represents a group whose members hold diverse identities;
the resulting purity is less than 1 unless all identities are identical.
"""
function GroupIdentity( weights::AbstractVector{<:Real}
                      , identities::AbstractVector{<:Identity{N,T}}
                      ) where {N, T <: Number}
    length(weights) == length(identities) ||
        throw(ArgumentError("weights and identities must have equal length"))
    all(≥(0), weights) ||
        throw(ArgumentError("weights must be non-negative"))
    w = weights ./ sum(weights)
    m = sum(wᵢ * (ψᵢ * ψᵢ') for (wᵢ, ψᵢ) in zip(w, identities))
    return GroupIdentity{N,T,N*N}(ntuple(i -> m[i], Val(N*N)))
end

"""
    GroupIdentity(m::AbstractMatrix{T}; check::Bool=true) -> GroupIdentity

Construct a GroupIdentity from a raw matrix. By default, checks Hermiticity,
unit trace, and positive semidefiniteness. Pass `check=false` to skip
invariant checking — useful in hot loops where the matrix is known to be
valid by construction (e.g. output of a dynamics step that provably preserves
the invariants).
"""
function GroupIdentity(m::AbstractMatrix{T}; check::Bool=true) where {T <: Number}
    n = size(m, 1)
    check && _check_density_matrix(m)
    return GroupIdentity{n,T,n*n}(ntuple(i -> m[i], Val(n*n)))
end

# ── Internal helpers ───────────────────────────────────────────────────────────
# purity, von_neumann_entropy, and alignment are defined in observables.jl.
# _is_pure is internal to the display code.

_is_pure(ρ::GroupIdentity) = real(tr(ρ * ρ)) ≈ 1.0

# ── Display ────────────────────────────────────────────────────────────────────

const _SUBSCRIPTS = ['₀','₁','₂','₃','₄','₅','₆','₇','₈','₉']

function _subscript(n::Int)
    n == 0 && return "₀"
    digits = Int[]
    m = n
    while m > 0
        pushfirst!(digits, m % 10)
        m ÷= 10
    end
    return join(_SUBSCRIPTS[d+1] for d in digits)
end

function _matrix_block(io::IO, ρ::GroupIdentity{N}) where N
    entries    = [_format_entry(ρ[i,j]) for i in 1:N, j in 1:N]
    col_widths = [maximum(length(entries[i,j]) for i in 1:N) for j in 1:N]
    for i in 1:N
        print(io, "  ")
        for j in 1:N
            print(io, lpad(entries[i,j], col_widths[j]))
            j < N && print(io, "  ")
        end
        println(io)
    end
end

function Base.show(io::IO, ρ::GroupIdentity{N,T}) where {N,T}
    S         = von_neumann_entropy(ρ)
    state_str = _is_pure(ρ) ? "pure" : "mixed"

    println(io, "$(N)×$(N) group identity ($(state_str)):")
    println(io)

    # ── Matrix block
    println(io, " matrix:")
    _matrix_block(io, ρ)
    println(io)

    # ── Eigendecomposition
    println(io, " decomposition:")
    h     = Hermitian(Matrix(ρ))
    λs    = eigvals(h)
    vs    = eigvecs(h)
    order = sortperm(λs, rev=true)
    λs    = λs[order]
    vs    = vs[:, order]

    # Non-negligible components only
    active = findall(>(1e-15), λs)
    kets   = ["|ψ" * _subscript(i) * "⟩" for i in active]

    # Summary: weight  |ψᵢ⟩
    for (i, idx) in enumerate(active)
        println(io, "  ", _fmt5_pos(λs[idx]), "  ", kets[i])
    end
    println(io)

    # Eigenvector listings
    for (i, idx) in enumerate(active)
        println(io, " ", kets[i], ":")
        ψ   = Identity(vs[:, idx]; norm=1.0)
        buf = IOBuffer()
        show(buf, ψ)
        for line in split(String(take!(buf)), '\n')
            isempty(line) || println(io, "  ", line)
        end
        println(io)
    end

    # ── Entropy summary
    println(io, @sprintf(" S = %.4f nats", S))
end

Base.show(io::IO, ::MIME"text/plain", ρ::GroupIdentity) = show(io, ρ)
