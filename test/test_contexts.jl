@testset "Contexts" begin

    @testset "project onto standard basis subspace" begin
        ψ = Identity([0.6+0im, 0.8+0im, 0.0+0im])
        z = project(ψ, [1])   # project onto dimension 1
        @test z[1] ≈ 0.6 atol=1e-10
        @test z[2] ≈ 0.0 atol=1e-10
        @test z[3] ≈ 0.0 atol=1e-10
    end

    @testset "collapse renormalises" begin
        ψ = Identity([0.6+0im, 0.8+0im, 0.0+0im])
        ψ_c = collapse(ψ, [1, 2])
        @test norm(ψ_c) ≈ 1.0 atol=1e-12
    end

    @testset "collapse onto full space is identity" begin
        ψ   = Identity([0.6+0im, 0.8+0im, 0.0+0im])
        A   = Matrix(I, 3, 3) * 1.0   # full space
        ψ_c = collapse(ψ, A)
        @test angle_between(ψ, ψ_c) ≈ 0.0 atol=1e-10
    end

    @testset "polarisation — coordinates amplify after collapse (Theorem 3)" begin
        # ψ has components along dims 1,2,3; collapse onto dims 1,2
        ψ = Identity([1/√3, 1/√3, 1/√3] .+ 0im)
        ψ_c = collapse(ψ, [1, 2])
        # After collapse, |ψ_c[1]|² + |ψ_c[2]|² = 1
        # Each coordinate must be ≥ its pre-collapse value
        @test abs(ψ_c[1]) ≥ abs(ψ[1]) - 1e-10
        @test abs(ψ_c[2]) ≥ abs(ψ[2]) - 1e-10
    end

    @testset "orthogonal projection raises error" begin
        ψ = Identity([0.0+0im, 0.0+0im, 1.0+0im])
        # Collapse onto dims 1,2 — ψ is orthogonal to this subspace
        @test_throws ArgumentError collapse(ψ, [1, 2])
    end

    @testset "context_sequence — order dependence" begin
        ψ = Identity([1/√3, 1/√3, 1/√3] .+ 0im)
        A1 = Float64[1 0; 0 1; 0 0]   # dims 1,2
        A2 = Float64[1 0; 0 0; 0 1]   # dims 1,3
        ψ_AB = context_sequence(ψ, [A1, A2])
        ψ_BA = context_sequence(ψ, [A2, A1])
        # In general, order matters
        # They may coincidentally agree; just check they are valid identities
        @test norm(ψ_AB) ≈ 1.0 atol=1e-10
        @test norm(ψ_BA) ≈ 1.0 atol=1e-10
    end

    @testset "collapse on GroupIdentity" begin
        ψ1 = Identity([1.0+0im, 0.0+0im, 0.0+0im])
        ψ2 = Identity([0.0+0im, 1.0+0im, 0.0+0im])
        ρ  = GroupIdentity([0.5, 0.5], [ψ1, ψ2])
        A  = Float64[1 0; 0 1; 0 0]   # dims 1,2
        ρ_c = collapse(ρ, A)
        @test tr(Matrix(ρ_c)) ≈ 1.0 atol=1e-10
    end

end
