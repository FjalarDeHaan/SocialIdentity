@testset "GroupIdentity" begin

    @testset "Pure state from Identity" begin
        ψ = Identity([1.0+0im, 0.0+0im])
        ρ = GroupIdentity(ψ)
        @test size(ρ) == (2, 2)
        @test tr(Matrix(ρ)) ≈ 1.0 atol=1e-12
        @test isapprox(Matrix(ρ), Matrix(ρ)', atol=1e-12)
        @test purity(ρ) ≈ 1.0 atol=1e-10
        @test von_neumann_entropy(ρ) ≈ 0.0 atol=1e-10
    end

    @testset "Mixed state from weights and identities" begin
        ψ1 = Identity([1.0+0im, 0.0+0im])
        ψ2 = Identity([0.0+0im, 1.0+0im])
        ρ  = GroupIdentity([0.5, 0.5], [ψ1, ψ2])
        @test tr(Matrix(ρ)) ≈ 1.0 atol=1e-12
        @test purity(ρ) ≈ 0.5 atol=1e-10
        # Maximally mixed 2×2: entropy = log(2)
        @test von_neumann_entropy(ρ) ≈ log(2) atol=1e-10
    end

    @testset "Weight normalisation" begin
        ψ1 = Identity([1.0+0im, 0.0+0im])
        ψ2 = Identity([0.0+0im, 1.0+0im])
        # Weights [2, 2] should give same result as [1, 1]
        ρ1 = GroupIdentity([1.0, 1.0], [ψ1, ψ2])
        ρ2 = GroupIdentity([2.0, 2.0], [ψ1, ψ2])
        @test isapprox(Matrix(ρ1), Matrix(ρ2), atol=1e-12)
    end

    @testset "Invariant checking — bad trace" begin
        m = [0.6 0.0; 0.0 0.6]  # trace = 1.2
        @test_throws ArgumentError GroupIdentity(m; check=true)
    end

    @testset "Invariant checking — not Hermitian" begin
        m = [0.5+0im 0.1+0.1im; 0.0+0im 0.5+0im]
        @test_throws ArgumentError GroupIdentity(m; check=true)
    end

    @testset "check=false bypasses invariant check" begin
        # Non-stochastic matrix, but check=false should not error
        m = [0.6+0im 0.0+0im; 0.0+0im 0.6+0im]
        @test_nowarn GroupIdentity(m; check=false)
    end

    @testset "alignment" begin
        ψ = Identity([1.0+0im, 0.0+0im])
        ρ = GroupIdentity(ψ)
        @test alignment(ρ) ≈ 1.0 atol=1e-10
    end

end
