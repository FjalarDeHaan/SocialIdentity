# identities.jl
#
# Defines the Identity type: a unit vector in Cⁿ representing an individual
# social identity. Identities live on the unit sphere in identity space.
#
# References:
#   de Haan (2025), §2.2: Identity Vector Spaces.
#   Synthesis document §3.1: Real vs. Complex Identity Space.

# ── Type definition ────────────────────────────────────────────────────────────

"""
    Identity{N, T <: Number} <: StaticVector{N, T}

An individual social identity represented as a unit vector in Cⁿ (or Rⁿ as a
special case). Identities live on the unit sphere: `norm(ψ) == 1` is the
expected invariant, enforced by the provided constructors.

The type parameter `N` is the dimensionality of the identity space — the number
of independent normative dimensions used to characterise identity. The parameter
`T` is the element type; `ComplexF64` is the default and recommended choice,
as the framework is developed in Cⁿ (Framework Choice 3, synthesis document §3).

Each coordinate ψᵢ is a complex amplitude. The squared modulus |ψᵢ|² gives
the probability (or intensity) of identification along normative dimension i.
The relative phase between components encodes the integration vs.
compartmentalisation of identity dimensions (synthesis document §3.2).

# Aliases
    const ψ = Identity    # defined in SocialIdentity.jl

# References
- de Haan, F.J. (2025). Towards a Formalism for the Social Identity Approach.
- Synthesis document §1.1, §3.1.
"""
struct Identity{N, T <: Number} <: StaticVector{N, T}
    values::NTuple{N, T}
end

# ── StaticArrays interface ─────────────────────────────────────────────────────

StaticArrays.Size(::Type{Identity{N, T}}) where {N, T} = Size(N)

@inline function Base.getindex(ψ::Identity, i::Int)
    @boundscheck checkbounds(ψ, i)
    ψ.values[i]
end

StaticArrays.similar_type(::Type{<:Identity}, ::Type{T}, ::Size{S}) where {T, S} =
    length(S) == 1 ? Identity{S[1], T} : SArray{Tuple{S...}, T, length(S), prod(S)}

# ── Constructors ───────────────────────────────────────────────────────────────

# The default struct constructor Identity{N,T}(::NTuple{N,T}) is provided
# automatically by Julia. No explicit outer NTuple constructor is needed —
# defining one would overwrite the inner constructor and cause a precompilation
# error. Callers can use Identity{N,T}(tuple) directly.

"""
    Identity(values::AbstractVector{T}; norm::Real=1.0) -> Identity

Construct an Identity from a vector, scaling to the given target norm
(default 1.0, i.e. normalised to unit length). Use `norm != 1.0` only when
constructing unnormalised intermediate results.
"""
function Identity(values::AbstractVector{T}; norm::Real=1.0) where {T <: Number}
    v = values .* (norm / LinearAlgebra.norm(values))
    return Identity(Tuple(v))
end

"""
    Identity(n::Int; norm::Real=1.0) -> Identity{n, ComplexF64}

Construct a random complex Identity of dimension `n`, uniformly distributed
on the unit sphere in Cⁿ under the Haar measure. This is achieved by drawing
each component from a standard complex normal distribution and normalising —
the unique rotationally-isotropic construction.

This is the default random constructor. Complex is the default because the
framework is developed in Cⁿ (Framework Choice 3, synthesis document §3).

The `norm` keyword scales the result to any target L2 norm (default 1.0).
"""
Identity(n::Int; norm::Real=1.0) = Identity(Random.default_rng(), n; norm=norm)

"""
    Identity(rng::AbstractRNG, n::Int; norm::Real=1.0) -> Identity{n, ComplexF64}

Seeded version of the random complex constructor. Draws from a standard
complex normal distribution and normalises, giving a Haar-uniform sample
on the unit sphere in Cⁿ.
"""
function Identity(rng::AbstractRNG, n::Int; norm::Real=1.0)
    v = randn(rng, ComplexF64, n)
    return Identity(v; norm=norm)
end

"""
    Identity(n::Int, ::Type{<:Real}; norm::Real=1.0) -> Identity{n, Float64}

Construct a random real Identity of dimension `n`, uniformly distributed on
the unit sphere in Rⁿ. Pass `Real` as the second argument to opt in to the
real-valued case.
"""
Identity(n::Int, ::Type{<:Real}; norm::Real=1.0) =
    Identity(Random.default_rng(), n, Real; norm=norm)

"""
    Identity(rng::AbstractRNG, n::Int, ::Type{<:Real}; norm::Real=1.0)

Seeded version of the random real constructor.
"""
function Identity(rng::AbstractRNG, n::Int, ::Type{<:Real}; norm::Real=1.0)
    v = randn(rng, Float64, n)
    return Identity(v; norm=norm)
end

"""
    Identity( rng::AbstractRNG, n::Int
            , biasdir::Identity
            , biasangle::Real
            ; norm::Real=1.0 ) -> Identity{n, ComplexF64}

Construct a random Identity within a bias cone: rejection-sample from the
Haar-uniform distribution on the unit sphere in Cⁿ, accepting only vectors
whose angle from `biasdir` is less than `biasangle` (in radians).

Corresponds to the shared-bias group construction in Theorem 1 of de Haan
(2025). The expected group identity of a set of such identities is parallel
to `biasdir`.

!!! warning
    Rejection sampling can be slow for tight cones (small `biasangle`) in
    high dimensions. The acceptance probability scales as `biasangle^(n-1)`.
"""
function Identity( rng::AbstractRNG, n::Int
                 , biasdir::Identity
                 , biasangle::Real
                 ; norm::Real=1.0 )
    while true
        ψ = Identity(rng, n; norm=norm)
        if _angle_between_raw(ψ, biasdir) < biasangle
            return ψ
        end
    end
end

# Unseeded bias-cone constructor
function Identity(n::Int, biasdir::Identity, biasangle::Real; norm::Real=1.0)
    return Identity(Random.default_rng(), n, biasdir, biasangle; norm=norm)
end

# ── Operations ─────────────────────────────────────────────────────────────────

"""
    normalise(ψ::Identity) -> Identity

Return a unit-norm copy of `ψ`. Equivalent to `ψ / norm(ψ)`.
"""
normalise(ψ::Identity) = ψ / LinearAlgebra.norm(ψ)

# Internal: angle without absolute value, used in bias-cone sampling
_angle_between_raw(ψ₁::Identity, ψ₂::Identity) =
    acos(clamp(real(dot(ψ₁, ψ₂)) / (LinearAlgebra.norm(ψ₁) * LinearAlgebra.norm(ψ₂)), -1.0, 1.0))

# ── Display helpers ────────────────────────────────────────────────────────────
#
# Fixed 5-column field: [sign][units][.][d1][d2]
#
#   sign  : '-' or ' '
#   units : digit or ' '   (blank for pure fractions; no leading zero ever)
#   '.'
#   d1 d2 : two decimal digits, or spaces for exact integers
#
# Examples:
#    .34   →  "  .34"
#   -.57   →  "- .57"
#    1     →  " 1   "
#   -1     →  "-1   "
#    0     →  " 0   "

function _fmt5(x::Real)
    x ==  0.0 && return " 0   "
    x ==  1.0 && return " 1   "
    x == -1.0 && return "-1   "
    sgn  = x < 0 ? '-' : ' '
    frac = lstrip(@sprintf("%.2f", abs(x)), '0')   # "0.34" → ".34"
    return string(sgn, ' ', frac)
end

function _fmt5_pos(r::Real)
    r == 0.0 && return " 0   "
    r == 1.0 && return " 1   "
    frac = lstrip(@sprintf("%.2f", r), '0')
    return "  " * frac
end

# Angle string: exact fractions of π shown symbolically, fallback to decimal×π
function _angle_str(θ::Real)
    ratio = θ / π
    for q in 1:12
        for p in (-q):q
            p == 0 && continue
            if abs(ratio - p / q) < 1e-9
                g  = gcd(abs(p), q)
                pn = p ÷ g
                qn = q ÷ g
                qn == 1  && return pn ==  1 ? "π"  : "-π"
                pn ==  1 && return "π/$qn"
                pn == -1 && return "-π/$qn"
                return "($pn/$qn)π"
            end
        end
    end
    # Fallback: decimal coefficient of π, no leading zero.
    # If the coefficient rounds to ±1 at 2dp, the angle is near but not equal
    # to ±π — flag with ≈ rather than printing the misleading "1.00π".
    sgn = ratio < 0 ? '-' : ' '
    ar  = abs(ratio)
    if round(ar, digits=2) == 1.0
        return ratio < 0 ? "≈-π" : "≈π"
    end
    frac = ar < 1 ? lstrip(@sprintf("%.2f", ar), '0') : @sprintf("%.2f", ar)
    return string(sgn, frac, 'π')
end

function _format_entry(x::Real)
    return _fmt5(x)
end

function _format_entry(z::Complex)
    r = abs(z)
    r == 0.0                  && return " 0   "
    θ = angle(z)               # range (-π, π]
    abs(θ)     < 1e-10        && return _fmt5(r)    # nearly positive real
    abs(θ - π) < 1e-10        && return _fmt5(-r)   # nearly negative real
    abs(θ + π) < 1e-10        && return _fmt5(-r)   # -π (same point)
    round(abs(θ / π), digits=2) == 0.0 && return _fmt5_pos(r) * " φ≈0"
    angle_s = _angle_str(θ)
    sep = startswith(angle_s, '≈') ? " φ" : " φ="
    return _fmt5_pos(r) * sep * angle_s
end

function Base.show(io::IO, ψ::Identity)
    n     = length(ψ.values)
    T     = eltype(ψ)
    tstr  = T <: Complex ? "complex " : ""
    n_val = LinearAlgebra.norm(ψ)
    if n_val ≈ 1
        println(io, n, "-dimensional $(tstr)identity (normalised):")
    else
        println(io, n, "-dimensional $(tstr)identity ",
                "(norm ≈ ", @sprintf("%.2f", n_val), "):")
    end
    for v in ψ.values
        println(io, _format_entry(v))
    end
end

Base.show(io::IO, ::MIME"text/plain", ψ::Identity) = show(io, ψ)
