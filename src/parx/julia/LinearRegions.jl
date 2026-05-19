module LinearRegions

using LinearAlgebra
using JuMP
using HiGHS

# ── Internal helpers ──────────────────────────────────────────────────────────

# Numpy arrays passed via juliacall implement AbstractArray; constructing
# Matrix{Float64}/Vector{Float64} copies into column-major Julia storage
# while preserving the (out_features, in_features) indexing convention.
function _to_weights(py_weights, py_biases)
    weights = [Matrix{Float64}(w) for w in py_weights]
    biases  = [Vector{Float64}(b) for b in py_biases]
    return weights, biases
end

# ── Phase 1: smoke tests & bridge verification ────────────────────────────────

function hello()
    return "LinearRegions.jl is loaded and running on $(Threads.nthreads()) thread(s)."
end

"""
    network_info(py_weights, py_biases) -> (input_dim, n_layers)

Accepts weight/bias lists from Python and returns basic architecture info.
Serves as the Phase 1 smoke test for the Python↔Julia data bridge.
"""
function network_info(py_weights, py_biases)
    weights, biases = _to_weights(py_weights, py_biases)
    input_dim = size(weights[1], 2)
    n_layers  = length(weights)
    return (input_dim, n_layers)
end

# ── Phase 2: sparse region finder ────────────────────────────────────────────

"""
    find_regions_sparse(py_weights, py_biases, py_points)
        -> (patterns, offsets, centroids)

Find all linear regions observed in `py_points` via parallel forward passes.

Each unique activation path corresponds to one linear region.

Returns
-------
patterns  : (n_regions, total_bits) Int8 matrix — flattened activation bits,
            all layers concatenated.
offsets   : (n_layers+1,) Int64 vector — 0-indexed column boundaries such that
            `patterns[i, offsets[l]+1 : offsets[l+1]]` (Julia 1-indexed) or
            `patterns[i, offsets[l] : offsets[l+1]]` (Python 0-indexed) gives
            the activation at layer l for region i.
centroids : (n_regions, input_dim) Float64 matrix — the first data point routed
            to each region, used as a feasible interior point.
"""
function find_regions_sparse(py_weights, py_biases, py_points)
    weights, biases = _to_weights(py_weights, py_biases)
    points = Matrix{Float64}(py_points)   # (N, input_dim)

    N         = size(points, 1)
    n_layers  = length(weights)
    layer_sizes = [size(w, 1) for w in weights]   # neurons at each layer
    total_bits  = sum(layer_sizes)

    # 0-indexed column boundaries (Python-compatible)
    offsets = Vector{Int64}(undef, n_layers + 1)
    offsets[1] = 0
    for l in 1:n_layers
        offsets[l + 1] = offsets[l] + layer_sizes[l]
    end

    # ── Parallel forward pass ─────────────────────────────────────────────
    all_paths = Vector{Vector{Int8}}(undef, N)

    Threads.@threads for i in 1:N
        x    = Vector{Float64}(points[i, :])
        path = Vector{Int8}(undef, total_bits)
        a    = x
        for l in 1:n_layers
            z = weights[l] * a + biases[l]
            q = Int8.(z .> 0)
            # offsets are 0-indexed; Julia slices are 1-indexed and inclusive
            path[offsets[l] + 1 : offsets[l + 1]] = q
            a = q .* z   # ReLU: active neurons pass z through, others → 0
        end
        all_paths[i] = path
    end

    # ── Deduplication (sequential) ────────────────────────────────────────
    # Map each unique path to the index of the first point that produced it.
    seen          = Dict{Vector{Int8}, Int}()
    unique_indices = Int[]

    for i in 1:N
        if !haskey(seen, all_paths[i])
            seen[all_paths[i]] = i
            push!(unique_indices, i)
        end
    end

    n_unique = length(unique_indices)

    # ── Pack outputs ──────────────────────────────────────────────────────
    input_dim = size(points, 2)
    patterns  = Matrix{Int8}(undef,    n_unique, total_bits)
    centroids = Matrix{Float64}(undef, n_unique, input_dim)

    for (i, idx) in enumerate(unique_indices)
        patterns[i, :]  = all_paths[idx]
        centroids[i, :] = points[idx, :]
    end

    return (patterns, offsets, centroids)
end

# ── Phase 3: exact region finder (DFS + facet-flipping) ──────────────────────

# Find the Chebyshev centre of {x : D*x ≤ g}.
# Returns (x_interior, radius).  Returns (nothing, 0.0) when empty or degenerate.
#
# Zero-norm rows arise when upstream neurons are inactive, making W_hat[i,:] = 0.
# Such rows encode a fixed constraint 0*x ≤ g[i]:
#   g[i] < 0  →  infeasible   →  return nothing
#   g[i] ≥ 0  →  trivially satisfied  →  skip (would make LP unbounded otherwise)
function _chebyshev_center(D::Matrix{Float64}, g::Vector{Float64})
    m = size(D, 1)
    n = size(D, 2)
    m == 0 && return zeros(n), Inf

    valid = [norm(@view D[i, :]) >= 1e-10 for i in 1:m]
    for i in 1:m
        !valid[i] && g[i] < -1e-10 && return nothing, 0.0
    end
    valid_idx = findall(valid)
    isempty(valid_idx) && return zeros(n), Inf

    D_v = D[valid_idx, :]
    g_v = g[valid_idx]
    mv  = length(valid_idx)

    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x[1:n])
    # Cap r so the LP stays bounded even when the polytope is unbounded.
    @variable(model, 0.0 <= r <= 1e3)
    row_norms = [norm(D_v[i, :]) for i in 1:mv]
    for i in 1:mv
        @constraint(model, dot(D_v[i, :], x) + r * row_norms[i] <= g_v[i])
    end
    @objective(model, Max, r)
    optimize!(model)

    st = termination_status(model)
    (st == MOI.OPTIMAL || st == MOI.ALMOST_OPTIMAL) || return nothing, 0.0
    value(r) > 1e-8                                  || return nothing, 0.0
    return value.(x), value(r)
end

# Indices (1-based within D_local rows) of non-redundant local constraints.
# D_full = [D_prev ; D_local], n_prev = number of rows belonging to D_prev.
# A local constraint is non-redundant if its removal strictly enlarges the polytope,
# i.e. there exists a feasible point (without that constraint) that violates it.
function _active_local_indices(
    D_full::Matrix{Float64},
    g_full::Vector{Float64},
    n_prev::Int,
)
    n_full  = size(D_full, 1)
    n_local = n_full - n_prev
    n_vars  = size(D_full, 2)
    active  = Int32[]

    for i in 1:n_local
        full_i = n_prev + i
        # Zero-norm rows are degenerate (fixed-value neuron) — not a real facet.
        norm(@view D_full[full_i, :]) < 1e-10 && continue
        model  = Model(HiGHS.Optimizer)
        set_silent(model)
        @variable(model, x[1:n_vars])
        for j in 1:n_full
            j == full_i && continue
            @constraint(model, dot(D_full[j, :], x) <= g_full[j])
        end
        @objective(model, Max, dot(D_full[full_i, :], x))
        optimize!(model)

        st = termination_status(model)
        if st == MOI.DUAL_INFEASIBLE    # unbounded ↔ constraint is active
            push!(active, Int32(i))
        elseif (st == MOI.OPTIMAL || st == MOI.ALMOST_OPTIMAL) &&
               objective_value(model) > g_full[full_i] - 1e-8
            push!(active, Int32(i))
        end
    end
    return active
end

# Recursive DFS.  Appends (flat_activation_path, interior_point) pairs to `results`.
function _dfs_exact!(
    weights ::Vector{Matrix{Float64}},
    biases  ::Vector{Vector{Float64}},
    layer   ::Int,
    A_prev  ::Matrix{Float64},   # accumulated linear map from input (input_dim → n_out_{layer-1})
    c_prev  ::Vector{Float64},   # accumulated bias
    D_prev  ::Matrix{Float64},   # stacked constraint rows from ancestor layers
    g_prev  ::Vector{Float64},
    q_path  ::Vector{BitVector}, # activation patterns for layers 1 … layer-1
    x_parent::Vector{Float64},   # interior point of parent region
    results ::Vector{Tuple{Vector{Int8}, Vector{Float64}}},
)
    if layer > length(weights)
        # Leaf: complete activation path collected → store it.
        path_flat = reduce(vcat, [Int8.(q) for q in q_path])
        push!(results, (path_flat, copy(x_parent)))
        return
    end

    W, b  = weights[layer], biases[layer]
    W_hat = W * A_prev        # (n_out, input_dim)
    b_hat = W * c_prev + b    # (n_out,)

    # Starting activation pattern at this layer given the parent's interior point.
    q_start = BitVector(W_hat * x_parent + b_hat .> 0)
    queue   = [q_start]
    visited = Set{BitVector}([q_start])

    while !isempty(queue)
        q_curr = popfirst!(queue)

        # Halfspace constraints induced by q_curr at this layer.
        s       = -2.0 .* Float64.(q_curr) .+ 1.0  # -1 if active, +1 if inactive
        D_local = s .* W_hat     # (n_out, input_dim)
        g_local = -(s .* b_hat)  # (n_out,)

        D_full = vcat(D_prev, D_local)
        g_full = vcat(g_prev, g_local)

        # Check feasibility and obtain a Chebyshev centre.
        x_int, r = _chebyshev_center(D_full, g_full)
        isnothing(x_int) && continue   # empty or degenerate region — skip

        # Propagate affine map for the next layer (active neurons only).
        A_next = Float64.(q_curr) .* W_hat   # (n_out, input_dim)
        c_next = Float64.(q_curr) .* b_hat   # (n_out,)

        # Recurse deeper.
        _dfs_exact!(
            weights, biases,
            layer + 1,
            A_next, c_next,
            D_full, g_full,
            [q_path; [q_curr]],
            x_int,
            results,
        )

        # Facet-flipping: find active (non-redundant) local constraints and
        # enqueue the neighbouring region obtained by flipping each one.
        active = _active_local_indices(D_full, g_full, size(D_prev, 1))
        for i in active
            q_n = copy(q_curr)
            q_n[i] = !q_n[i]
            if q_n ∉ visited
                push!(visited, q_n)
                push!(queue, q_n)
            end
        end
    end
end

"""
    find_regions_exact(py_weights, py_biases, py_x0)
        -> (patterns, offsets, centroids)

Exactly enumerate all feasible linear regions via DFS + facet-flipping.

`py_x0` is a starting point in input space, shape `(input_dim,)`.  The DFS
begins from the region containing `x0` and traverses all reachable neighbours
by flipping active halfspace constraints at each layer.

Return format is identical to `find_regions_sparse`.  Scales with the total
number of regions, not with dataset size; each LP is tiny but many are solved.
"""
function find_regions_exact(py_weights, py_biases, py_x0)
    weights, biases = _to_weights(py_weights, py_biases)
    x0 = Vector{Float64}(py_x0)

    n_layers    = length(weights)
    layer_sizes = [size(w, 1) for w in weights]
    total_bits  = sum(layer_sizes)
    input_dim   = size(weights[1], 2)

    offsets = Vector{Int64}(undef, n_layers + 1)
    offsets[1] = 0
    for l in 1:n_layers
        offsets[l + 1] = offsets[l] + layer_sizes[l]
    end

    results = Vector{Tuple{Vector{Int8}, Vector{Float64}}}()
    _dfs_exact!(
        weights, biases,
        1,
        Matrix{Float64}(I, input_dim, input_dim),
        zeros(Float64, input_dim),
        Matrix{Float64}(undef, 0, input_dim),
        Float64[],
        BitVector[],
        x0,
        results,
    )

    n_regions = length(results)
    patterns  = Matrix{Int8}(undef,    n_regions, total_bits)
    centroids = Matrix{Float64}(undef, n_regions, input_dim)

    for (i, (path, centroid)) in enumerate(results)
        patterns[i, :]  = path
        centroids[i, :] = centroid
    end

    return (patterns, offsets, centroids)
end

end # module
