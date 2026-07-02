# LP helpers shared by all exact-enumeration backends.
#
# Both functions accept a `make_model` argument — a zero-argument callable that
# returns a fresh JuMP model — so the caller controls solver construction:
#
#   _make_model_standard() = Model(HiGHS.Optimizer)        # full JuMP bridge
#   _make_model_direct()   = direct_model(HiGHS.Optimizer()) # skip bridge (~6× faster)

_make_model_standard() = Model(HiGHS.Optimizer)
_make_model_direct()   = direct_model(HiGHS.Optimizer())

# Find the Chebyshev centre of {x : D*x ≤ g}.
# Returns (x_interior, radius).  Returns (nothing, 0.0) when empty or degenerate.
#
# Zero-norm rows arise when upstream neurons are inactive (W_hat[i,:] = 0).
# Such rows encode a fixed constraint 0*x ≤ g[i]:
#   g[i] < 0  →  infeasible  →  return nothing
#   g[i] ≥ 0  →  trivially satisfied  →  skip (would make LP unbounded otherwise)
function _chebyshev_center(D::Matrix{Float64}, g::Vector{Float64}, make_model)
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

    model = make_model()
    set_silent(model)
    @variable(model, x[1:n])
    @variable(model, 0.0 <= r <= 1e3)
    row_norms = [norm(D_v[i, :]) for i in 1:mv]
    for i in 1:mv
        @constraint(model, dot(D_v[i, :], x) + r * row_norms[i] <= g_v[i])
    end
    @objective(model, Max, 1.0 * r)   # affine form works for both Model and direct_model
    optimize!(model)

    st = termination_status(model)
    (st == MOI.OPTIMAL || st == MOI.ALMOST_OPTIMAL) || return nothing, 0.0
    value(r) > 1e-8                                  || return nothing, 0.0
    return value.(x), value(r)
end

# Indices (1-based within D_local rows) of non-redundant local constraints.
# D_full = [D_prev ; D_local], n_prev = number of rows belonging to D_prev.
# A local constraint is non-redundant if its removal strictly enlarges the
# polytope, i.e. there exists a feasible point that violates it.
function _active_local_indices(
    D_full::Matrix{Float64},
    g_full::Vector{Float64},
    n_prev::Int,
    make_model,
)
    n_full  = size(D_full, 1)
    n_local = n_full - n_prev
    n_vars  = size(D_full, 2)
    active_flags = Vector{Bool}(undef, n_local)

    Threads.@threads for i in 1:n_local
        full_i = n_prev + i
        # Zero-norm rows are degenerate (fixed-value neuron) — not a real facet.
        if norm(@view D_full[full_i, :]) < 1e-10
            active_flags[i] = false
            continue
        end
        model = make_model()
        set_silent(model)
        @variable(model, x[1:n_vars])
        for j in 1:n_full
            j == full_i && continue
            @constraint(model, dot(D_full[j, :], x) <= g_full[j])
        end
        @objective(model, Max, dot(D_full[full_i, :], x))
        optimize!(model)

        st = termination_status(model)
        active_flags[i] = (st == MOI.DUAL_INFEASIBLE) ||
            ((st == MOI.OPTIMAL || st == MOI.ALMOST_OPTIMAL) &&
             objective_value(model) > g_full[full_i] - 1e-8)
    end

    return Int32.(findall(active_flags))
end
