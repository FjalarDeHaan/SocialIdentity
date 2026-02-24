@testset "Observables" begin

    @testset "inner product" begin
        ψ1 = Identity([1.0+0im, 0.0+0im])
        ψ2 = Identity([0.0+0im, 1.0+0im])
        @test inner(ψ1, ψ1) ≈ 1.0 + 0im atol=1e-12
        @test inner(ψ1, ψ2) ≈ 0.0 + 0im atol=1e-12
    end

    @testset "overlap" begin
        ψ1 = Identity([1.0+0im, 0.0+0im])
        ψ2 = Identity([0.0+0im, 1.0+0im])
        @test overlap(ψ1, ψ1) ≈ 1.0 atol=1e-12
        @test overlap(ψ1, ψ2) ≈ 0.0 atol=1e-12
        # overlap ∈ [0,1] always
        for _ in 1:20
            a, b = Identity(4), Identity(4)
            @test 0 ≤ overlap(a, b) ≤ 1
        end
    end

    @testset "salience" begin
        ψ = Identity([1.0+0im, 0.0+0im])
        c = Identity([1.0+0im, 0.0+0im])
        @test salience(ψ, c) ≈ 1.0 atol=1e-12
        c2 = Identity([0.0+0im, 1.0+0im])
        @test salience(ψ, c2) ≈ 0.0 atol=1e-12
    end

    @testset "angle_between — basic" begin
        ψ1 = Identity([1.0+0im, 0.0+0im])
        ψ2 = Identity([0.0+0im, 1.0+0im])
        @test angle_between(ψ1, ψ1) ≈ 0.0 atol=1e-12
        @test angle_between(ψ1, ψ2) ≈ π/2 atol=1e-10
    end

    @testset "angle_between — global phase invariance" begin
        ψ = Identity(4)
        ψ_phased = Identity(exp(im * π/3) .* Vector{ComplexF64}(ψ); norm=1.0)
        @test angle_between(ψ, ψ_phased) ≈ 0.0 atol=1e-10
    end

    @testset "angle_between — real antipodal vectors have angle 0" begin
        ψ = Identity([1.0, 0.0, 0.0])
        # Framework Choice 4: -ψ is not the same as ψ's opposite in this model,
        # but geometrically they have angle 0 because |⟨ψ|-ψ⟩| = 1.
        neg_ψ = Identity([-1.0, 0.0, 0.0])
        @test angle_between(ψ, neg_ψ) ≈ 0.0 atol=1e-10
    end

    @testset "angle_between ∈ [0, π/2]" begin
        for _ in 1:50
            a, b = Identity(4), Identity(4)
            θ = angle_between(a, b)
            @test 0 ≤ θ ≤ π/2 + 1e-10
        end
    end

    @testset "φ alias for angle_between" begin
        ψ1, ψ2 = Identity(4), Identity(4)
        @test φ(ψ1, ψ2) == angle_between(ψ1, ψ2)
    end

    @testset "purity" begin
        ψ = Identity([1.0+0im, 0.0+0im])
        ρ_pure = GroupIdentity(ψ)
        @test purity(ρ_pure) ≈ 1.0 atol=1e-10
        ψ2 = Identity([0.0+0im, 1.0+0im])
        ρ_mixed = GroupIdentity([0.5, 0.5], [ψ, ψ2])
        @test purity(ρ_mixed) ≈ 0.5 atol=1e-10
    end

    @testset "von_neumann_entropy" begin
        ψ = Identity([1.0+0im, 0.0+0im])
        ρ_pure = GroupIdentity(ψ)
        @test von_neumann_entropy(ρ_pure) ≈ 0.0 atol=1e-10
        ψ2 = Identity([0.0+0im, 1.0+0im])
        ρ_max = GroupIdentity([0.5, 0.5], [ψ, ψ2])
        @test von_neumann_entropy(ρ_max) ≈ log(2) atol=1e-10
    end

    @testset "alignment" begin
        ψ = Identity([1.0+0im, 0.0+0im])
        ρ = GroupIdentity(ψ)
        @test alignment(ρ) ≈ 1.0 atol=1e-10
    end

end
