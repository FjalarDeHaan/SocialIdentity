# bases.jl
#
# Standard basis vectors and basis transformations for identity space.
#
# In the formalism, the standard basis corresponds to an exhaustive, mutually
# independent list of normative topics. Any orthonormal set of such topics
# forms a valid basis, and identity vectors against one basis (questionnaire A)
# can be converted to another (questionnaire B) by orthogonal transformation.
#
# References:
#   de Haan (2025), §2.2: the identity space basis and questionnaire interpretation.

"""
    e(i::Int, n::Int) -> Identity{n, Float64}

Return the `i`-th standard basis vector in Rⁿ as an Identity. Represents a
'pure' normative position along the `i`-th identity dimension — an individual
whose identity is entirely defined by a single normative aspect.

# Examples
```julia
e(1, 3)  # [1.0, 0.0, 0.0] as an Identity{3, Float64}
e(2, 4)  # [0.0, 1.0, 0.0, 0.0] as an Identity{4, Float64}
```
"""
function e(i::Int, n::Int)
    i in 1:n || throw(BoundsError("index $i out of range for dimension $n"))
    return Identity(NTuple{n, Float64}(j == i ? 1.0 : 0.0 for j in 1:n))
end

"""
    e(i::Int, ψ::Identity) -> Identity

Return the `i`-th standard basis vector in the same space as `ψ`, inferring
the dimension from `ψ`.
"""
e(i::Int, ψ::Identity{N}) where N = e(i, N)

"""
    standard_basis(n::Int) -> Vector{Identity{n, Float64}}

Return all `n` standard basis vectors in Rⁿ as a `Vector` of Identity
objects. These correspond to the `n` normative dimensions of identity space,
each represented as a pure identity along one axis.
"""
standard_basis(n::Int) = [e(i, n) for i in 1:n]

"""
    orthonormal_basis(vecs::Vector{<:Identity}) -> Vector{<:Identity}

Compute an orthonormal basis for the span of `vecs` using the Gram-Schmidt
process. Input vectors must be linearly independent; an error is raised if
near-linear-dependence is detected (norm of residual < 1e-10).

This corresponds to the basis-change operation in de Haan (2025), §2.2: any
complete orthonormal set of normative directions is a valid basis for identity
space, and identities expressed against one basis can be converted to another
by unitary transformation.

# References
- de Haan (2025), §2.2.
"""
function orthonormal_basis(vecs::Vector{<:Identity})
    isempty(vecs) && throw(ArgumentError("input vector list is empty"))
    basis = Identity[]
    for v in vecs
        # Subtract projections onto all existing basis vectors
        r = Vector{eltype(v)}(v)
        for b in basis
            r = r - (dot(b, v)) * Vector{eltype(b)}(b)
        end
        n = LinearAlgebra.norm(r)
        n < 1e-10 && throw(ArgumentError(
            "input vectors are linearly dependent (residual norm = $n)"))
        push!(basis, Identity(r; norm=1.0))
    end
    return basis
end

"""
    represent(ψ::Identity, basis::Vector{<:Identity}) -> Vector{ComplexF64}

Return the coordinates of `ψ` in the given orthonormal basis. This is the
basis-change operation: if `basis` corresponds to questionnaire B and `ψ`
was constructed against questionnaire A, `represent` gives the questionnaire-B
coordinates.

The `basis` must be orthonormal (use `orthonormal_basis` to construct one if
needed). No normalisation is applied to the output — coordinates of a unit
vector in an orthonormal basis have squared moduli summing to 1.

# References
- de Haan (2025), §2.2: orthogonal transformations preserve angles and norms.
"""
function represent(ψ::Identity, basis::Vector{<:Identity})
    return [dot(b, ψ) for b in basis]
end

"""
    change_basis( ψ::Identity
                , from_basis::Vector{<:Identity}
                , to_basis::Vector{<:Identity}
                ) -> Identity

Express `ψ` (given in `from_basis` coordinates) in `to_basis` coordinates.
Both bases must be orthonormal and span the same space.

The transformation preserves the angle between any two identities and their
norms — it is a unitary transformation of the identity space.

# References
- de Haan (2025), §2.2: orthogonal transformations as changes of questionnaire.
"""
function change_basis( ψ::Identity
                     , from_basis::Vector{<:Identity}
                     , to_basis::Vector{<:Identity} )
    length(from_basis) == length(to_basis) ||
        throw(ArgumentError("bases must have the same length"))
    # Coordinates in the from_basis
    coords = represent(ψ, from_basis)
    # Reconstruct in the ambient space using from_basis, then re-express in to_basis
    ambient = sum(c * Vector{ComplexF64}(b) for (c, b) in zip(coords, from_basis))
    return Identity(ambient; norm=1.0)
end
