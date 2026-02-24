# networks.jl
#
# Defines the SocialIdentityNetwork (SIN) type: a weighted directed graph
# in which vertices carry Identity vectors and edges carry identification
# weights, forming a row-stochastic adjacency matrix. Implements DeGroot
# social learning and Markov chain decomposition for computing group identities.
#
# References:
#   de Haan (2025), §2.1: Social Identity Networks.
#   de Haan (2025), §3: Emergence of Group Identities (DeGroot, W∞).
#   Synthesis document §2.1–2.2.

# ── Auxiliary types ────────────────────────────────────────────────────────────

"""
    RecurrentClass{L}

The result of decomposing a SocialIdentityNetwork into one of its recurrent
(absorbing) classes under the Markov chain induced by the stochastic adjacency
matrix W.

Only recurrent classes contribute to stable group identities. Transient agents
are eventually absorbed into recurrent classes and do not appear here.

# Fields
- `group_identity :: GroupIdentity` — the shared group identity of this class,
  computed as the stationary-distribution-weighted mixture of member pure states:
  ρ = Σᵢ πᵢ ψᵢψᵢ†
- `members :: Vector{L}` — vertex labels of agents in this recurrent class
- `weights :: Vector{Float64}` — stationary distribution πᵢ over members,
  measuring each agent's influence on the group identity (Theorem 2,
  de Haan 2025)

# References
- de Haan (2025), §3.3: Stable Group Identities Through Social Structure.
- Bertsekas & Tsitsiklis (2002), §4: recurrent and transient classes.
"""
struct RecurrentClass{L}
    group_identity :: GroupIdentity
    members        :: Vector{L}
    weights        :: Vector{Float64}
end

# ── SocialIdentityNetwork type ─────────────────────────────────────────────────

"""
    SocialIdentityNetwork{L, N, T}

A social identity network (SIN): a weighted directed graph in which each
vertex represents an individual carrying an Identity vector, and each directed
edge i → j carries a weight Wᵢⱼ ∈ [0,1] representing how much individual i
identifies with individual j.

Row sums equal 1 (stochastic matrix): each individual's total identification
weight is 1, distributed across those they identify with. Self-loops are
allowed and represent self-identification (sui generis identity).

Wraps a MetaGraphsNext.MetaGraph with:
  - Vertex label type L (any hashable type)
  - Vertex data type Identity{N,T}
  - Edge data type Float64 (identification weight)

Row-stochasticity is enforced by `set_identifications!` and `normalise!`.

# Alias
    const SIN = SocialIdentityNetwork    # defined in SocialIdentity.jl

# References
- de Haan (2025), §2.1: Social Identity Networks.
- de Haan (2025), §2.3: the adjacency matrix W is a stochastic matrix.
"""
struct SocialIdentityNetwork{L, N, T <: Number, G}
    _graph::G

    function SocialIdentityNetwork{L,N,T}() where {L, N, T <: Number}
        g = MetaGraph(
                SimpleDiGraph{Int}(),
                label_type       = L,
                vertex_data_type = Identity{N,T},
                edge_data_type   = Float64,
                graph_data       = nothing,
            )
        new{L,N,T,typeof(g)}(g)
    end
end

# ── Constructors ───────────────────────────────────────────────────────────────

"""
    SocialIdentityNetwork{L,N,T}() -> SocialIdentityNetwork{L,N,T}

Construct an empty SIN with vertex label type `L`, identity dimension `N`,
and element type `T`.

# Examples
```julia
sin = SocialIdentityNetwork{Symbol, 4, ComplexF64}()
```
"""
# (defined via inner constructor above)

"""
    SocialIdentityNetwork( W::AbstractMatrix{Float64}
                         , identities::Vector{<:Identity{N,T}}
                         ) -> SocialIdentityNetwork{Int,N,T}

Construct a SIN from a stochastic adjacency matrix `W` and a corresponding
vector of Identity vectors. Vertex labels are integer indices `1:n`. `W` must
be square, non-negative, and row-stochastic (rows sum to 1 within `atol=1e-8`).

Entry `W[i,j]` is the weight that agent `i` places on agent `j`.
"""
function SocialIdentityNetwork( W::AbstractMatrix{Float64}
                               , identities::Vector{<:Identity{N,T}}
                               ) where {N, T <: Number}
    n = size(W, 1)
    n == size(W, 2) || throw(ArgumentError("W must be square"))
    n == length(identities) ||
        throw(ArgumentError("W and identities must have the same length"))
    _check_stochastic(W)
    sin = SocialIdentityNetwork{Int,N,T}()
    for i in 1:n
        add_identity!(sin, i, identities[i])
    end
    for i in 1:n, j in 1:n
        W[i,j] > 0 && _set_edge!(sin, i, j, W[i,j])
    end
    return sin
end

"""
    SocialIdentityNetwork( W::AbstractMatrix{Float64}
                         , identities::Vector{<:Identity{N,T}}
                         , labels::Vector{L}
                         ) -> SocialIdentityNetwork{L,N,T}

As above, with explicit vertex labels. `labels[i]` is the label for the
`i`-th row/column of `W`.
"""
function SocialIdentityNetwork( W::AbstractMatrix{Float64}
                               , identities::Vector{<:Identity{N,T}}
                               , labels::Vector{L}
                               ) where {N, T <: Number, L}
    n = size(W, 1)
    n == length(labels) || throw(ArgumentError("W and labels must have the same length"))
    n == length(identities) || throw(ArgumentError("W and identities must have the same length"))
    _check_stochastic(W)
    sin = SocialIdentityNetwork{L,N,T}()
    for i in 1:n
        add_identity!(sin, labels[i], identities[i])
    end
    for i in 1:n, j in 1:n
        W[i,j] > 0 && _set_edge!(sin, labels[i], labels[j], W[i,j])
    end
    return sin
end

# ── Random / structured constructors ──────────────────────────────────────────

"""
    hierarchy( rng::AbstractRNG, n::Int, teamsize::Int
             ; identity_dim::Int=4
             ) -> SocialIdentityNetwork{Int, identity_dim, ComplexF64}

Construct a hierarchical SIN: a random tree of `n` manager-nodes, each with
a clique of `teamsize` team members attached as leaves. Edge weights are
assigned so that identification flows more strongly up the hierarchy (weight 5)
than across peers (weight 1) or down (weight 0.5). Rows are normalised to
maintain stochasticity.

Corresponds to Example 1 (de Haan 2025, §3.3) and Theorem 2 (hierarchy
implies stability of group identity through social structure).

# References
- de Haan (2025), §3.3, Definition 3 and Theorem 2.
"""
function hierarchy( rng::AbstractRNG=Random.default_rng()
                  , n::Int=5
                  , teamsize::Int=5
                  ; identity_dim::Int=4 )
    N   = identity_dim
    T   = ComplexF64
    # Build a random tree and convert to directed graph
    g   = uniform_tree(n; rng=rng) |> DiGraph
    root   = argmax(degree(g))
    d      = gdistances(g, root)
    leaves = findall(v -> degree(g, v) == 1 && v != root, 1:nv(g))

    # Build adjacency matrix with hierarchy weights
    W = zeros(Float64, n, n)
    for edge in edges(g)
        Δd = d[edge.dst] - d[edge.src]
        w  = Δd > 0 ? 0.5 : Δd < 0 ? 5.0 : 1.0
        W[edge.src, edge.dst] = w
        # Undirected tree: also add reverse
        W[edge.dst, edge.src] = Δd > 0 ? 5.0 : Δd < 0 ? 0.5 : 1.0
    end
    # Self-identification for root
    W[root, root] = 2.0

    # Add team cliques to leaves
    n_total = n + length(leaves) * teamsize
    W_full  = zeros(Float64, n_total, n_total)
    W_full[1:n, 1:n] = W

    offset = n
    for leaf in leaves
        # Fully connected team clique
        for i in 1:teamsize, j in 1:teamsize
            if i != j
                W_full[offset+i, offset+j] = 1.0
            end
        end
        # Team identifies with leaf manager
        for i in 1:teamsize
            W_full[offset+i, leaf] = 2.0
        end
        # Leaf manager identifies with first team member (to make it recurrent)
        W_full[leaf, offset+1] = 1.0
        offset += teamsize
    end

    # Normalise rows
    for i in 1:n_total
        s = sum(W_full[i, :])
        s > 0 && (W_full[i, :] ./= s)
        s == 0 && (W_full[i, i] = 1.0)  # isolated vertex: self-identify
    end

    identities = [Identity(rng, N) for _ in 1:n_total]
    return SocialIdentityNetwork(W_full, identities)
end

"""
    strongly_er( rng::AbstractRNG, n::Int, m::Int
               ; identity_dim::Int=4
               ) -> SocialIdentityNetwork{Int, identity_dim, ComplexF64}

Construct a random strongly-connected SIN via Erdős–Rényi sampling with `n`
vertices and `m` edges, rejecting graphs that are not strongly connected.
Edge weights are uniform and row-normalised.

!!! warning
    For sparse `m` (near `n`), rejection may be slow. Ensure `m > n` for
    reasonable acceptance rates.
"""
function strongly_er( rng::AbstractRNG=Random.default_rng()
                    , n::Int=5
                    , m::Int=10
                    ; identity_dim::Int=4 )
    N = identity_dim
    T = ComplexF64
    local g
    while true
        g = erdos_renyi(n, m; rng=rng) |> DiGraph
        is_strongly_connected(g) && break
    end
    W = Float64.(collect(adjacency_matrix(g)))
    for i in 1:n
        s = sum(W[i, :])
        s > 0 && (W[i, :] ./= s)
        s == 0 && (W[i, i] = 1.0)
    end
    identities = [Identity(rng, N) for _ in 1:n]
    return SocialIdentityNetwork(W, identities)
end

# ── Internal helpers ───────────────────────────────────────────────────────────

function _check_stochastic(W::AbstractMatrix)
    all(≥(0), W) ||
        throw(ArgumentError("W has negative entries"))
    for i in 1:size(W,1)
        s = sum(W[i,:])
        isapprox(s, 1.0, atol=1e-8) ||
            throw(ArgumentError("row $i of W sums to $s, expected 1"))
    end
end

# Internal edge setter — bypasses stochastic checks, for use in constructors
function _set_edge!(sin::SocialIdentityNetwork, from_label, to_label, w::Float64)
    sin._graph[from_label, to_label] = w
end

# ── Core graph operations ──────────────────────────────────────────────────────

"""
    add_identity!(sin::SocialIdentityNetwork, label, ψ::Identity)

Add a new agent with the given label and Identity vector as a vertex in the
SIN. The agent initially has no identification edges; use `set_identifications!`
to assign them.

Raises an error if the label already exists.
"""
function add_identity!(sin::SocialIdentityNetwork, label, ψ::Identity)
    haskey(sin._graph, label) &&
        throw(ArgumentError("label $label already exists in the network"))
    sin._graph[label] = ψ
end

"""
    remove_identity!(sin::SocialIdentityNetwork, label)

Remove the agent with the given label from the SIN, along with all incident
edges.

!!! note
    This invalidates any cached stochastic matrices or stationary distributions.
    MetaGraphsNext uses swap-and-pop internally, so integer codes are reassigned
    after removal; always access vertices by label, not by internal code.
"""
function remove_identity!(sin::SocialIdentityNetwork, label)
    haskey(sin._graph, label) ||
        throw(ArgumentError("label $label not found in the network"))
    delete!(sin._graph, label)
end

"""
    set_identifications!( sin::SocialIdentityNetwork, from_label
                        , to_labels::Vector
                        , weights::Vector{Float64} )

Set all outgoing identification weights from `from_label` to the agents in
`to_labels` with the corresponding `weights`. Normalises the row automatically
since the full set of identifications for this agent is being specified at once.
Weights need only be non-negative; they are normalised to sum to 1.

Any existing outgoing edges from `from_label` not in `to_labels` are removed.

This is the only public API for setting edge weights. Providing all outgoing
identifications at once ensures row-stochasticity is always maintained.
"""
function set_identifications!( sin::SocialIdentityNetwork, from_label
                              , to_labels::Vector
                              , weights::Vector{Float64} )
    length(to_labels) == length(weights) ||
        throw(ArgumentError("to_labels and weights must have equal length"))
    all(≥(0), weights) ||
        throw(ArgumentError("weights must be non-negative"))
    sum(weights) > 0 ||
        throw(ArgumentError("at least one weight must be positive"))

    # Remove existing outgoing edges from from_label
    g    = sin._graph
    code = code_for(g, from_label)
    for nb in copy(outneighbors(g.graph, code))
        rem_edge!(g.graph, code, nb)
    end

    w_norm = weights ./ sum(weights)
    for (to_label, w) in zip(to_labels, w_norm)
        _set_edge!(sin, from_label, to_label, w)
    end
end

"""
    normalise!(sin::SocialIdentityNetwork, label)

Renormalise the outgoing edge weights of `label` so that they sum to 1,
restoring row-stochasticity. Use this after externally constructing a SIN
whose edge weights are known to be valid but may not sum exactly to 1.

Raises an error if all outgoing weights are zero (undefined normalisation).
"""
function normalise!(sin::SocialIdentityNetwork, label)
    g    = sin._graph
    code = code_for(g, label)
    nbs  = outneighbors(g.graph, code)
    isempty(nbs) && throw(ArgumentError(
        "agent $label has no outgoing edges to normalise"))
    ws   = [g[label_for(g, code), label_for(g, nb)] for nb in nbs]
    s    = sum(ws)
    s ≈ 0 && throw(ArgumentError(
        "all outgoing weights for $label are zero; cannot normalise"))
    for (nb, w) in zip(nbs, ws)
        g[label, label_for(g, nb)] = w / s
    end
end

"""
    identity_of(sin::SocialIdentityNetwork, label) -> Identity

Return the Identity vector of the agent with the given label.

This function is overloaded by the `ψ` alias (see `SocialIdentity.jl`), so
`ψ(sin, label)` is equivalent.
"""
identity_of(sin::SocialIdentityNetwork, label) = sin._graph[label]

"""
    weight(sin::SocialIdentityNetwork, from_label, to_label) -> Float64

Return the identification weight Wᵢⱼ from `from_label` to `to_label`,
or 0.0 if no edge exists between them.
"""
function weight(sin::SocialIdentityNetwork, from_label, to_label)
    g = sin._graph
    haskey(g, from_label, to_label) ? g[from_label, to_label] : 0.0
end

"""
    labels(sin::SocialIdentityNetwork) -> Vector

Return all vertex labels in the SIN.
"""
labels(sin::SocialIdentityNetwork) = collect(labels(sin._graph))

"""
    n_agents(sin::SocialIdentityNetwork) -> Int

Return the number of agents (vertices) in the SIN.
"""
n_agents(sin::SocialIdentityNetwork) = nv(sin._graph)

# ── DeGroot dynamics ───────────────────────────────────────────────────────────

"""
    stochastic_matrix(sin::SocialIdentityNetwork) -> Matrix{Float64}

Return the adjacency matrix W of the SIN as a plain Julia matrix. Entry Wᵢⱼ
is the identification weight of agent i towards agent j. Each row sums to 1.

Vertices are ordered by internal code (insertion order for integer-labelled
SINs). Use `labels(sin)` to recover the label for each row/column index.

# References
- de Haan (2025), §2.3: W is a stochastic matrix.
"""
function stochastic_matrix(sin::SocialIdentityNetwork)
    g = sin._graph
    n = nv(g.graph)
    W = zeros(Float64, n, n)
    for e in edge_labels(g)
        i = code_for(g, e[1])
        j = code_for(g, e[2])
        W[i,j] = e[3]
    end
    return W
end

"""
    degroot(sin::SocialIdentityNetwork, t::Int) -> Matrix{Float64}

Return the t-step DeGroot update matrix Wᵗ. Entry (i,j) of Wᵗ gives the
weight that agent i places on agent j's identity after `t` rounds of
social learning.

As t → ∞, Wᵗ converges to W∞ where it exists. Use `stationary` for the
limit directly.

# References
- DeGroot, M.H. (1974). Reaching a consensus. JASA, 69(345), 118–121.
- Golub & Jackson (2010). Naïve learning in social networks. AEJ Micro, 2(1).
"""
function degroot(sin::SocialIdentityNetwork, t::Int)
    t ≥ 0 || throw(ArgumentError("t must be non-negative"))
    W = stochastic_matrix(sin)
    return W^t
end

"""
    stationary(sin::SocialIdentityNetwork) -> Vector{RecurrentClass{L}}

Decompose the SIN into its recurrent classes and compute the stationary
distribution within each. Returns one `RecurrentClass` per recurrent class,
containing the group identity, member labels, and influence weights.

Transient agents do not appear in any `RecurrentClass` — their identities
do not contribute to any stable group identity (standard Markov chain result;
Bertsekas & Tsitsiklis 2002, §4).

If the SIN is strongly connected and each agent self-identifies to a finite
degree, there is exactly one recurrent class.

# Algorithm
1. Find strongly connected components (SCCs) of the graph.
2. Build the condensation DAG; identify recurrent SCCs (no outgoing edges).
3. For each recurrent SCC, compute the stationary distribution via the
   left eigenvector of the sub-matrix of W with eigenvalue 1.
4. Construct the GroupIdentity as ρ = Σᵢ πᵢ ψᵢψᵢ†.

# References
- de Haan (2025), §3.3, Theorem 2.
- Synthesis document §2.1.
- Bertsekas & Tsitsiklis (2002). Introduction to Probability, §4.
"""
function stationary(sin::SocialIdentityNetwork)
    g      = sin._graph
    W      = stochastic_matrix(sin)
    graph  = g.graph
    n      = nv(graph)
    n == 0 && return RecurrentClass[]

    # Step 1: SCCs
    sccs = strongly_connected_components(graph)

    # Step 2: identify recurrent SCCs (no outgoing edges to other SCCs)
    scc_id = zeros(Int, n)
    for (k, scc) in enumerate(sccs)
        for v in scc
            scc_id[v] = k
        end
    end

    recurrent_sccs = Int[]
    for (k, scc) in enumerate(sccs)
        is_recurrent = true
        for v in scc
            for nb in outneighbors(graph, v)
                if scc_id[nb] != k
                    is_recurrent = false
                    break
                end
            end
            is_recurrent || break
        end
        is_recurrent && push!(recurrent_sccs, k)
    end

    # Step 3 & 4: for each recurrent SCC, find stationary distribution
    result = RecurrentClass[]
    for k in recurrent_sccs
        scc      = sccs[k]
        m        = length(scc)
        # Sub-matrix of W for this SCC
        W_sub    = W[scc, scc]
        # Left eigenvector with eigenvalue ≈ 1: solve (W_sub' - I)π = 0
        # via eigendecomposition of W_sub'
        F        = eigen(W_sub')
        idx      = argmin(abs.(F.values .- 1.0))
        π_raw    = real.(F.vectors[:, idx])
        π_raw    = abs.(π_raw)          # ensure non-negative
        π        = π_raw ./ sum(π_raw)  # normalise

        # Retrieve labels and identities
        member_labels = [label_for(g, v) for v in scc]
        member_ids    = [identity_of(sin, l) for l in member_labels]

        ρ = GroupIdentity(π, member_ids)
        push!(result, RecurrentClass(ρ, member_labels, π))
    end

    return result
end

"""
    group_identity(sin::SocialIdentityNetwork) -> Vector{GroupIdentity}

Convenience wrapper around `stationary`. Returns just the `GroupIdentity`
for each recurrent class, discarding membership and weight information.
"""
group_identity(sin::SocialIdentityNetwork) =
    [rc.group_identity for rc in stationary(sin)]

"""
    influence(sin::SocialIdentityNetwork, label) -> Float64

The stationary distribution weight of agent `label` in its recurrent class:
the fraction of the group identity determined by this agent's identity vector.

Returns 0.0 for transient agents (their identity does not contribute to any
stable group identity).

Corresponds to w∞ᵢ in de Haan (2025), Theorem 2. Agents higher in a
hierarchy have larger influence; lower agents have geometrically smaller
influence in proportion to the power differentials on the path to the top.

# References
- de Haan (2025), §3.3, Theorem 2.
"""
function influence(sin::SocialIdentityNetwork, label)
    for rc in stationary(sin)
        idx = findfirst(==(label), rc.members)
        isnothing(idx) || return rc.weights[idx]
    end
    return 0.0
end

# ── Display ────────────────────────────────────────────────────────────────────

function Base.show(io::IO, sin::SocialIdentityNetwork{L,N,T}) where {L,N,T}
    n = n_agents(sin)
    println(io, "SocialIdentityNetwork{$L} with $n agents, $N-dimensional $(T) identities")
    println(io, " Labels: ", join(labels(sin), ", "))
end

Base.show(io::IO, ::MIME"text/plain", sin::SocialIdentityNetwork) = show(io, sin)
