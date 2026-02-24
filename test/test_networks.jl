@testset "Networks" begin

    function simple_sin()
        # Two agents: a identifies with b, b identifies with itself
        ψ_a = Identity([1.0+0im, 0.0+0im])
        ψ_b = Identity([0.0+0im, 1.0+0im])
        W   = [0.0 1.0; 0.0 1.0]
        return SocialIdentityNetwork(W, [ψ_a, ψ_b])
    end

    @testset "Construction from W and identities" begin
        sin = simple_sin()
        @test n_agents(sin) == 2
    end

    @testset "Construction with labels" begin
        ψ_a = Identity([1.0+0im, 0.0+0im])
        ψ_b = Identity([0.0+0im, 1.0+0im])
        W   = [0.0 1.0; 0.0 1.0]
        sin = SocialIdentityNetwork(W, [ψ_a, ψ_b], [:alice, :bob])
        @test n_agents(sin) == 2
        @test identity_of(sin, :alice) == ψ_a
        @test identity_of(sin, :bob)   == ψ_b
    end

    @testset "add_identity! and identity_of" begin
        sin = SocialIdentityNetwork{Symbol, 2, ComplexF64}()
        ψ   = Identity([1.0+0im, 0.0+0im])
        add_identity!(sin, :alice, ψ)
        @test n_agents(sin) == 1
        @test identity_of(sin, :alice) == ψ
    end

    @testset "set_identifications! maintains stochasticity" begin
        sin = SocialIdentityNetwork{Symbol, 2, ComplexF64}()
        ψ_a = Identity([1.0+0im, 0.0+0im])
        ψ_b = Identity([0.0+0im, 1.0+0im])
        add_identity!(sin, :alice, ψ_a)
        add_identity!(sin, :bob,   ψ_b)
        set_identifications!(sin, :alice, [:alice, :bob], [1.0, 3.0])
        @test weight(sin, :alice, :alice) ≈ 0.25 atol=1e-10
        @test weight(sin, :alice, :bob)   ≈ 0.75 atol=1e-10
    end

    @testset "remove_identity!" begin
        sin = SocialIdentityNetwork{Symbol, 2, ComplexF64}()
        ψ   = Identity([1.0+0im, 0.0+0im])
        add_identity!(sin, :alice, ψ)
        remove_identity!(sin, :alice)
        @test n_agents(sin) == 0
    end

    @testset "stochastic_matrix" begin
        sin = simple_sin()
        W   = stochastic_matrix(sin)
        @test size(W) == (2, 2)
        for i in 1:2
            @test sum(W[i,:]) ≈ 1.0 atol=1e-10
        end
    end

    @testset "degroot — t=0 is identity matrix" begin
        sin = simple_sin()
        W0  = degroot(sin, 0)
        @test W0 ≈ I atol=1e-12
    end

    @testset "stationary — absorbing state" begin
        # Agent 2 is absorbing (self-loop only); agent 1 identifies only with agent 2.
        # So agent 2 is the sole recurrent class.
        sin = simple_sin()
        rcs = stationary(sin)
        @test length(rcs) == 1
        # The recurrent class has one member: agent 2 (index 2)
        @test length(rcs[1].members) == 1
        @test rcs[1].weights[1] ≈ 1.0 atol=1e-10
    end

    @testset "influence — transient agent has influence 0" begin
        sin = simple_sin()
        # Agent 1 (index 1) is transient
        @test influence(sin, 1) ≈ 0.0 atol=1e-10
        # Agent 2 (index 2) is recurrent
        @test influence(sin, 2) ≈ 1.0 atol=1e-10
    end

    @testset "SIN alias" begin
        @test SIN === SocialIdentityNetwork
    end

end
