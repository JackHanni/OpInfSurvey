# Shared pieces for the OpInf analyses in docs/project_plan.html (questions 1–5).
#
# Everything here works on the Burgers' setup in ../burgers/burgers_fd.jl. The FOM is
#   du/dt = A u + B(u, u),   A = ν D2 (interior rows),  B(u, u) = -u ⊙ (D1 u),
# with zero Dirichlet boundaries, so the reduced model has no constant term.

using LinearAlgebra, Printf, Random, Plots
include(joinpath(@__DIR__, "..", "burgers", "burgers_fd.jl"))   # solve_burgers, burgers_rhs!, rk4_step!

const FIGDIR = joinpath(@__DIR__, "output")
mkpath(FIGDIR)

# ------------------------------------------------------------------ data
"""
    burgers_data(; t_train = 2.0, t_end = 4.0)

Solve the FOM to `t_end` with default_params and split at `t_train`.
Returns (; xs, ts, U, dUdt, itr, dx, ν) where `itr` indexes the training snapshots.
"""
function burgers_data(; t_train = 2.0, t_end = 4.0)
    p = merge(default_params, (t_max = t_end, make_plot = false, make_gif = false))
    xs, ts, U, dUdt, _ = solve_burgers(p)
    itr = findall(t -> t <= t_train + 1e-12, ts)
    return (; xs, ts, U, dUdt, itr, dx = xs[2] - xs[1], ν = p.ν, t_train)
end

pod_basis(Utr, r) = svd(Utr).U[:, 1:r]

# ------------------------------------------------------------------ OpInf pieces
compact_kron(x) = [x[i] * x[j] for i in eachindex(x) for j in 1:i]   # same order as ../burgers/opInf.jl
nquad(r) = r * (r + 1) ÷ 2
ncols(r) = r + nquad(r)

"Data matrix D = [Q̂ᵀ  c(Q̂)ᵀ], one row per snapshot."
data_matrix(Qh) = hcat(Qh', reduce(vcat, [compact_kron(q)' for q in eachcol(Qh)]))

split_ops(O, r) = (O[:, 1:r], O[:, r+1:end])

"Second-order centered time derivative of the columns of X (one-sided at the ends)."
function fd_time_derivative(X, dt)
    Xd = similar(X)
    Xd[:, 2:end-1] = (X[:, 3:end] - X[:, 1:end-2]) / (2dt)
    Xd[:, 1]   = (-3X[:, 1] + 4X[:, 2] - X[:, 3]) / (2dt)
    Xd[:, end] = (3X[:, end] - 4X[:, end-1] + X[:, end-2]) / (2dt)
    return Xd
end

# ------------------------------------------------------------------ intrusive operators
"Centered first-difference applied to interior nodes (boundary rows zero)."
function d1(a, dx)
    out = zero(a)
    @inbounds for i in 2:length(a)-1
        out[i] = (a[i+1] - a[i-1]) / (2dx)
    end
    return out
end
"Symmetric bilinear form with B(u, u) = -u ⊙ D1 u (interior rows)."
bilinear(a, b, dx) = -(a .* d1(b, dx) .+ b .* d1(a, dx)) ./ 2

"""
    galerkin_ops(V, ν, dx) -> (Â, Ĥ)

Intrusive POD-Galerkin operators of the FD semi-discretization, with Ĥ in the
same compact ordering as `compact_kron`.
"""
function galerkin_ops(V, ν, dx)
    n, r = size(V)
    AV = similar(V)
    for j in 1:r                      # A v = burgers_rhs minus its quadratic part, done directly:
        v = V[:, j]; out = zero(v)
        @inbounds for i in 2:n-1
            out[i] = ν * (v[i+1] - 2v[i] + v[i-1]) / dx^2
        end
        AV[:, j] = out
    end
    Â = V' * AV
    Ĥ = zeros(r, nquad(r))
    col = 0
    for i in 1:r, j in 1:i
        col += 1
        Ĥ[:, col] = (i == j ? 1.0 : 2.0) * (V' * bilinear(V[:, i], V[:, j], dx))
    end
    return Â, Ĥ
end

# ------------------------------------------------------------------ least-squares solvers
# All solve  min ‖D Oᵀ − R‖²_F + λ‖O‖²_F  and return O (r × ncols).
solve_normal(D, R, λ = 0.0) = (cholesky(Symmetric(D' * D + λ * I)) \ (D' * R))'
function solve_qr(D, R, λ = 0.0)
    n = size(D, 2)
    A = λ > 0 ? [D; Matrix(sqrt(λ) * I, n, n)] : D
    B = λ > 0 ? [R; zeros(n, size(R, 2))] : R
    F = qr(A)                                        # Householder, no pivoting
    return (UpperTriangular(F.R) \ (F.Q' * B)[1:n, :])'
end

"""
    select_lambda_stability(D, R, Qh, ts, itr, λs; S = svd(D), growth = 1.5)

Stability-based choice (in the spirit of McQuarrie, Huang & Willcox 2021): among
λ whose ROM stays finite and within `growth`·max|Q̂_train| over the whole time
span `ts` (training and beyond), pick the one with the smallest reduced-state
error on the training window. Uses training data only.
Returns (λ, O, train_err); λ = NaN if no candidate is stable.
"""
function select_lambda_stability(D, R, Qh, ts, itr, λs; S = svd(D), growth = 1.5)
    r = size(Qh, 1)
    bound = growth * maximum(abs, Qh)
    best = (NaN, fill(NaN, r, ncols(r)), Inf)
    for λ in λs
        O = solve_svd(D, R, λ; S)
        Â, Ĥ = split_ops(O, r)
        Q = integrate_rom(Â, Ĥ, Qh[:, 1], ts; bound)
        any(isnan, Q) && continue
        e = norm(Q[:, itr] - Qh) / norm(Qh)
        e < best[3] && (best = (λ, O, e))
    end
    return best
end
"Tikhonov via the SVD (filter factors s/(s²+λ)); λ = 0 gives the pseudoinverse solution."
function solve_svd(D, R, λ = 0.0; S = svd(D))
    f = λ == 0 ? [s > eps() * S.S[1] * length(S.S) ? 1 / s : 0.0 for s in S.S] : S.S ./ (S.S .^ 2 .+ λ)
    return (S.V * (f .* (S.U' * R)))'
end

relerr(A, B) = norm(A - B) / norm(B)

# ------------------------------------------------------------------ ROM integration
"""
    integrate_rom(Â, Ĥ, q0, ts; sub = 10, bound = Inf) -> Q

RK4 with `sub` steps per output interval. Stops (rest NaN) if the state is
non-finite or exceeds `bound` in max-norm.
"""
function integrate_rom(Â, Ĥ, q0, ts; sub = 10, bound = Inf)
    r = length(q0)
    Q = fill(NaN, r, length(ts))
    q = copy(q0); Q[:, 1] .= q
    f!(dq, x) = (dq .= Â * x .+ Ĥ * compact_kron(x))
    k1, k2, k3, k4, tmp = (similar(q) for _ in 1:5)
    for n in 1:length(ts)-1
        h = (ts[n+1] - ts[n]) / sub
        for _ in 1:sub
            rk4_step!(q, h, f!, k1, k2, k3, k4, tmp)
        end
        (all(isfinite, q) && maximum(abs, q) <= bound) || return Q
        Q[:, n+1] .= q
    end
    return Q
end

"Relative Frobenius error of the lifted ROM over the columns `idx` (Inf if it blew up)."
function rom_error(V, Q, U, idx)
    any(isnan, view(Q, :, idx)) && return Inf
    return norm(V * Q[:, idx] - U[:, idx]) / norm(U[:, idx])
end

# ------------------------------------------------------------------ plotting helpers
default(fontfamily = "sans-serif", linewidth = 2, framestyle = :box, grid = true,
        legendfontsize = 8, guidefontsize = 10, titlefontsize = 10, size = (640, 420), dpi = 150)
logsafe(v; floor = 1e-17) = [isfinite(x) && x > 0 ? max(x, floor) : NaN for x in v]
savefig_here(plt, name) = (f = joinpath(FIGDIR, name); savefig(plt, f); println("  figure → ", relpath(f, dirname(@__DIR__))); f)
