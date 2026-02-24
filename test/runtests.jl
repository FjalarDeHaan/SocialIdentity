using Test
using SocialIdentity
using LinearAlgebra

@testset "SocialIdentity.jl" begin
    include("test_identities.jl")
    include("test_groups.jl")
    include("test_bases.jl")
    include("test_observables.jl")
    include("test_contexts.jl")
    include("test_networks.jl")
end
