# observables.jl
#
# Pure functions mapping package types to scalars or plain collections.
# No mutation, no type construction. All functions here take Identity or
# GroupIdentity values and return plain Julia numbers.
#
# Network-level observables (e.g. influence) live in networks.jl as they
# require access to the SocialIdentityNetwork type.

# ── Identity observables ───────────────────────────────────────────────────────

"""
    inner(ψ₁::Identity, ψ₂::Identity) -> Complex

The Hermitian inner product ⟨ψ₁|ψ₂⟩. This is the primitive operation from
which all other scalar observables are derived. Returns a complex number in
general; real when both identities are real-valued.

Note: in Dirac notation this is ⟨ψ₁|ψ₂⟩, not ⟨ψ₂|ψ₁⟩.
"""
inner(ψ₁::Identity, ψ₂::Identity) = dot(ψ₁, ψ₂)

"""
    overlap(ψ₁::Identity, ψ₂::Identity) -> Float64

The squared modulus of the inner product: |⟨ψ₁|ψ₂⟩|² ∈ [0, 1].

Measures identity similarity: 1 means identical identities (up to global
phase), 0 means orthogonal identities (no shared normative content). In the
real case this equals cos²(θ) where θ is the angle between ψ₁ and ψ₂.

This is identical in computation to `salience` but differs in semantics: in
`overlap` both arguments are identity vectors on equal footing, while in
`salience` one argument is a context direction.
"""
overlap(ψ₁::Identity, ψ₂::Identity) = abs2(inner(ψ₁, ψ₂))

"""
    salience(ψ::Identity, c::Identity) -> Float64

The probability that identity `ψ` expresses along context direction `c`:
|⟨c|ψ⟩|² ∈ [0, 1].

Formally identical to `overlap`, but with distinct semantics: `ψ` is the
identity being measured and `c` is the context direction (e.g. a normative
axis or another individual's identity). A salience of 1 means ψ is fully
aligned with the context; 0 means ψ is orthogonal to it.

# References
- de Haan (2025), §2.2: coordinates as salience probabilities.
- Synthesis document §3.2: the interference term in oblique contexts.
"""
salience(ψ::Identity, c::Identity) = abs2(inner(c, ψ))

"""
    angle_between(ψ₁::Identity, ψ₂::Identity) -> Float64

The angle between two identity vectors: arccos(|⟨ψ₁|ψ₂⟩|) ∈ [0, π/2].

The absolute value of the inner product is used in the complex case because
global phase is unobservable: `ψ` and `e^{iθ}ψ` represent the same identity.
Consequently, the range is [0, π/2] rather than [0, π], and antipodal real
vectors (ψ and -ψ) have angle 0.

This is the fundamental comparison operation in the formalism: two identities
are compared by the angle they subtend in identity space.

# Alias
    const φ = angle_between    # defined in SocialIdentity.jl

# References
- de Haan (2025), §2.2.
- Synthesis document §1.1, §4.1 (anti-identity problem and its resolution).
"""
function angle_between(ψ₁::Identity, ψ₂::Identity)
    o = clamp(sqrt(overlap(ψ₁, ψ₂)), 0.0, 1.0)
    return acos(o)
end

# ── GroupIdentity observables ──────────────────────────────────────────────────

"""
    purity(ρ::GroupIdentity) -> Float64

Compute the purity Tr(ρ²) ∈ [1/N, 1]. Equal to 1 for pure states and 1/N
for the maximally mixed state (uniform mixture over N orthogonal identities).
Measures how concentrated the group identity is around a single pure state.

See also `alignment`, `von_neumann_entropy`.
"""
purity(ρ::GroupIdentity) = real(tr(ρ * ρ))

"""
    von_neumann_entropy(ρ::GroupIdentity) -> Float64

Compute the von Neumann entropy S = -Tr(ρ log ρ) in nats (natural logarithm).
Equal to 0 for pure states and log(N) for the maximally mixed state.
Measures identity diversity within the group.

Eigenvalues below 1e-15 are treated as zero and omitted from the sum (they
arise from pure or near-pure states where numerical noise may give tiny
negative eigenvalues).

# References
- Nielsen & Chuang (2000), §11.3.
- Synthesis document §1.2.
"""
function von_neumann_entropy(ρ::GroupIdentity)
    λs = eigvals(Hermitian(Matrix(ρ)))
    return -sum(λ * log(λ) for λ in λs if λ > 1e-15)
end

"""
    alignment(ρ::GroupIdentity) -> Float64

Return the largest eigenvalue of ρ: the weight of the dominant pure state
in the spectral decomposition. Ranges from 1/N (maximally mixed) to 1 (pure).
A high alignment means one identity direction strongly dominates the group.

See also `purity`, `von_neumann_entropy`.
"""
alignment(ρ::GroupIdentity) = maximum(real.(eigvals(Hermitian(Matrix(ρ)))))
