@testset "Bases" begin

    @testset "Standard basis vector e(i, n)" begin
        v = e(1, 3)
        @test v[1] ≈ 1.0
        @test v[2] ≈ 0.0
        @test v[3] ≈ 0.0
        @test norm(v) ≈ 1.0
    end

    @testset "e(i, ψ) infers dimension" begin
        ψ = Identity(4)
        v = e(2, ψ)
        @test length(v) == 4
        @test v[2] ≈ 1.0
    end

    @testset "standard_basis is orthonormal" begin
        basis = standard_basis(3)
        @test length(basis) == 3
        for i in 1:3, j in 1:3
            @test inner(basis[i], basis[j]) ≈ (i == j ? 1.0 : 0.0) atol=1e-12
        end
    end

    @testset "orthonormal_basis — already orthonormal input" begin
        vecs  = standard_basis(3)
        basis = orthonormal_basis(vecs)
        for i in 1:3, j in 1:3
            @test abs(inner(basis[i], basis[j])) ≈ (i == j ? 1.0 : 0.0) atol=1e-10
        end
    end

    @testset "orthonormal_basis — general vectors" begin
        v1 = Identity([1.0+0im, 1.0+0im, 0.0+0im]; norm=1.0)
        v2 = Identity([0.0+0im, 1.0+0im, 1.0+0im]; norm=1.0)
        v3 = Identity([1.0+0im, 0.0+0im, 1.0+0im]; norm=1.0)
        basis = orthonormal_basis([v1, v2, v3])
        @test length(basis) == 3
        for i in 1:3, j in 1:3
            @test abs(inner(basis[i], basis[j])) ≈ (i == j ? 1.0 : 0.0) atol=1e-10
        end
    end

    @testset "represent recovers coordinates" begin
        basis = standard_basis(3)
        ψ     = Identity([0.6+0im, 0.8+0im, 0.0+0im])
        coords = represent(ψ, basis)
        @test coords[1] ≈ 0.6 atol=1e-10
        @test coords[2] ≈ 0.8 atol=1e-10
        @test coords[3] ≈ 0.0 atol=1e-10
    end

    @testset "change_basis is invertible" begin
        from_basis = standard_basis(3)
        v1 = Identity([1.0+0im, 1.0+0im, 0.0+0im]; norm=1.0)
        v2 = Identity([0.0+0im, 1.0+0im, 1.0+0im]; norm=1.0)
        v3 = Identity([1.0+0im, 0.0+0im, 1.0+0im]; norm=1.0)
        to_basis = orthonormal_basis([v1, v2, v3])
        ψ = Identity(3, Real)
        ψ2 = change_basis(ψ, from_basis, to_basis)
        ψ3 = change_basis(ψ2, to_basis, from_basis)
        # Round-trip should recover ψ up to numerical error
        @test norm(ψ) ≈ norm(ψ3) atol=1e-10
        @test angle_between(ψ, ψ3) ≈ 0.0 atol=1e-10
    end

end
