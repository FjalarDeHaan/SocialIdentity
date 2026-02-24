@testset "Identity" begin

    @testset "Construction from tuple" begin
        ψ = Identity((1.0+0im, 0.0+0im))
        @test length(ψ) == 2
        @test ψ[1] ≈ 1.0
        @test ψ[2] ≈ 0.0
    end

    @testset "Construction from vector — normalisation" begin
        v = [3.0, 4.0]
        ψ = Identity(v)
        @test norm(ψ) ≈ 1.0
        @test ψ[1] ≈ 0.6
        @test ψ[2] ≈ 0.8
    end

    @testset "Construction from vector — target norm" begin
        v = [1.0, 0.0]
        ψ = Identity(v; norm=2.0)
        @test norm(ψ) ≈ 2.0
    end

    @testset "Random complex constructor — unit norm" begin
        for _ in 1:20
            ψ = Identity(4)
            @test norm(ψ) ≈ 1.0 atol=1e-12
            @test eltype(ψ) == ComplexF64
        end
    end

    @testset "Random real constructor — unit norm" begin
        for _ in 1:20
            ψ = Identity(4, Real)
            @test norm(ψ) ≈ 1.0 atol=1e-12
            @test eltype(ψ) == Float64
        end
    end

    @testset "Seeded constructor is deterministic" begin
        using Random
        rng1 = Xoshiro(42)
        rng2 = Xoshiro(42)
        @test Identity(rng1, 4) == Identity(rng2, 4)
    end

    @testset "Bias-cone constructor" begin
        biasdir = Identity([1.0, 0.0, 0.0, 0.0])
        for _ in 1:10
            ψ = Identity(4, biasdir, π/6)
            @test angle_between(ψ, biasdir) < π/6 + 1e-10
            @test norm(ψ) ≈ 1.0 atol=1e-12
        end
    end

    @testset "normalise" begin
        ψ = Identity([3.0+0im, 4.0+0im]; norm=5.0)
        ψn = normalise(ψ)
        @test norm(ψn) ≈ 1.0 atol=1e-12
    end

    @testset "StaticArrays interface" begin
        ψ = Identity(3)
        @test size(ψ) == (3,)
        @test length(ψ) == 3
    end

    @testset "ψ alias" begin
        # ψ is Identity
        @test ψ === Identity
    end

end
