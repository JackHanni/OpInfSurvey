# Shared pieces for the OpInf analyses in docs/project_plan.html (questions 1–6).
#
# Everything here works on the Burgers' setup in ../burgers/burgers_fd.jl. The FOM is
#   du/dt = A u + B(u, u),   A = ν D2 (interior rows),  B(u, u) = -u ⊙ (D1 u),
# with zero Dirichlet boundaries, so the reduced model has no constant term.
#
# Notation follows analysis/README.md: V is the POD basis, Q̂ = VᵀU the reduced snapshots,
# D the data matrix, R the reduced time derivatives, O = [Â Ĥ] the operators, λ the Tikhonov
# parameter, r the ROM dimension.

using LinearAlgebra, Printf, Random, Plots
include(joinpath(@__DIR__, "..", "burgers", "burgers_fd.jl"))   # solve_burgers, burgers_rhs!, rk4_step!

const FIGDIR = joinpath(@__DIR__, "output")
mkpath(FIGDIR)

# ------------------------------------------------------------------ data
"""
    burgers_data(; t_train = 2.0, t_end = 4.0)

Solve the FOM to `t_end` with default_params and split at `t_train`.
Returns (; xs, ts, U, dUdt, train_idx, dx, ν, t_train) where `train_idx` indexes the
training snapshots.
"""
function burgers_data(; t_train = 2.0, t_end = 4.0)
    p = merge(default_params, (t_max = t_end, make_plot = false, make_gif = false))
    xs, ts, U, dUdt, _ = solve_burgers(p)
    train_idx = findall(t -> t <= t_train + 1e-12, ts)
    return (; xs, ts, U, dUdt, train_idx, dx = xs[2] - xs[1], ν = p.ν, t_train)
end

"The first r left singular vectors of the snapshot matrix."
pod_basis(snapshots, r) = svd(snapshots).U[:, 1:r]

# ------------------------------------------------------------------ OpInf pieces
"The r(r+1)/2 distinct products q_i q_j (j ≤ i): q ⊗ q without the repeated entries."
compact_kron(q) = [q[i] * q[j] for i in eachindex(q) for j in 1:i]   # same order as ../burgers/opInf.jl

n_quadratic_terms(r) = r * (r + 1) ÷ 2
"Number of unknowns per row of O = number of columns of D."
n_unknowns(r) = r + n_quadratic_terms(r)

"Data matrix D = [Q̂ᵀ  c(Q̂)ᵀ], one row per snapshot (`Q̂` has one snapshot per column)."
data_matrix(Q̂) = hcat(Q̂', reduce(vcat, [compact_kron(q̂)' for q̂ in eachcol(Q̂)]))

"Split O = [Â Ĥ] into (Â, Ĥ)."
split_operators(O, r) = (O[:, 1:r], O[:, r+1:end])

"Second-order centered time derivative of the columns of X (one-sided at the ends)."
function fd_time_derivative(X, dt)
    dXdt = similar(X)
    dXdt[:, 2:end-1] = (X[:, 3:end] - X[:, 1:end-2]) / (2dt)
    dXdt[:, 1]   = (-3X[:, 1] + 4X[:, 2] - X[:, 3]) / (2dt)
    dXdt[:, end] = (3X[:, end] - 4X[:, end-1] + X[:, end-2]) / (2dt)
    return dXdt
end

# ------------------------------------------------------------------ intrusive operators
"Centered first difference at interior nodes (boundary entries zero)."
function centered_difference(a, dx)
    result = zero(a)
    @inbounds for i in 2:length(a)-1
        result[i] = (a[i+1] - a[i-1]) / (2dx)
    end
    return result
end

"Symmetric bilinear form with B(u, u) = -u ⊙ D1 u (interior rows)."
burgers_bilinear(a, b, dx) = -(a .* centered_difference(b, dx) .+ b .* centered_difference(a, dx)) ./ 2

"""
    galerkin_operators(V, ν, dx) -> (Â, Ĥ)

Intrusive POD-Galerkin operators of the FD semi-discretization, with Ĥ in the
same compact ordering as `compact_kron`. Used only as the reference answer that
OpInf should recover.
"""
function galerkin_operators(V, ν, dx)
    n_points, r = size(V)
    A_times_V = similar(V)
    for j in 1:r          # A v is the diffusion part of burgers_rhs!, done directly
        v = V[:, j]
        Av = zero(v)
        @inbounds for i in 2:n_points-1
            Av[i] = ν * (v[i+1] - 2v[i] + v[i-1]) / dx^2
        end
        A_times_V[:, j] = Av
    end
    Â = V' * A_times_V
    Ĥ = zeros(r, n_quadratic_terms(r))
    column = 0
    for i in 1:r, j in 1:i
        column += 1
        symmetry_factor = i == j ? 1.0 : 2.0     # q_i q_j appears twice in q ⊗ q when i ≠ j
        Ĥ[:, column] = symmetry_factor * (V' * burgers_bilinear(V[:, i], V[:, j], dx))
    end
    return Â, Ĥ
end

# ------------------------------------------------------------------ least-squares solvers
# All solve  min ‖D Oᵀ − R‖²_F + λ‖O‖²_F  and return O (r × n_unknowns).
"Normal equations, solved by Cholesky. Throws if DᵀD + λI is not numerically positive definite."
solve_normal(D, R, λ = 0.0) = (cholesky(Symmetric(D' * D + λ * I)) \ (D' * R))'

"Householder QR (no pivoting) of D, or of [D; √λ I] when λ > 0."
function solve_qr(D, R, λ = 0.0)
    n_cols = size(D, 2)
    if λ > 0
        D_stacked = [D; Matrix(sqrt(λ) * I, n_cols, n_cols)]
        R_stacked = [R; zeros(n_cols, size(R, 2))]
    else
        D_stacked, R_stacked = D, R
    end
    qr_D = qr(D_stacked)
    return (UpperTriangular(qr_D.R) \ (qr_D.Q' * R_stacked)[1:n_cols, :])'
end

"""
Tikhonov via the SVD of D (filter factors σ/(σ² + λ)). λ = 0 gives the pseudoinverse
solution, dropping singular values below the usual rank tolerance. Pass `svd_D` to reuse
one factorization across many λ.
"""
function solve_svd(D, R, λ = 0.0; svd_D = svd(D))
    σ = svd_D.S
    if λ == 0
        rank_tolerance = eps() * σ[1] * length(σ)
        inverse_factors = [s > rank_tolerance ? 1 / s : 0.0 for s in σ]
    else
        inverse_factors = σ ./ (σ .^ 2 .+ λ)
    end
    return (svd_D.V * (inverse_factors .* (svd_D.U' * R)))'
end

"""
    select_lambda_stability(D, R, Q̂, ts, train_idx, λ_grid; svd_D = svd(D), growth = 1.5)

Stability-based choice (in the spirit of McQuarrie, Huang & Willcox 2021): among
λ whose ROM stays finite and within `growth`·max|Q̂_train| over the whole time
span `ts` (training and beyond), pick the one with the smallest reduced-state
error on the training window. Uses training data only.
Returns (λ, O, train_err); λ = NaN if no candidate is stable.
"""
function select_lambda_stability(D, R, Q̂, ts, train_idx, λ_grid; svd_D = svd(D), growth = 1.5)
    r = size(Q̂, 1)
    max_abs = growth * maximum(abs, Q̂)
    best_λ, best_O, best_err = NaN, fill(NaN, r, n_unknowns(r)), Inf
    for λ in λ_grid
        O = solve_svd(D, R, λ; svd_D)
        Â, Ĥ = split_operators(O, r)
        Q = integrate_rom(Â, Ĥ, Q̂[:, 1], ts; max_abs)
        if any(isnan, Q)          # blew up or left the bound: not a candidate
            continue
        end
        train_err = norm(Q[:, train_idx] - Q̂) / norm(Q̂)
        if train_err < best_err
            best_λ, best_O, best_err = λ, O, train_err
        end
    end
    return best_λ, best_O, best_err
end

relative_error(A, B) = norm(A - B) / norm(B)

# ------------------------------------------------------------------ ROM integration
"""
    integrate_rom(Â, Ĥ, q0, ts; substeps = 10, max_abs = Inf) -> Q

RK4 with `substeps` steps per output interval. Stops (rest NaN) if the state is
non-finite or exceeds `max_abs` in max-norm.
"""
function integrate_rom(Â, Ĥ, q0, ts; substeps = 10, max_abs = Inf)
    r = length(q0)
    Q = fill(NaN, r, length(ts))
    q = copy(q0)
    Q[:, 1] .= q
    rom_rhs!(dq, x) = (dq .= Â * x .+ Ĥ * compact_kron(x))
    k1, k2, k3, k4, stage = (similar(q) for _ in 1:5)
    for n in 1:length(ts)-1
        dt = (ts[n+1] - ts[n]) / substeps
        for _ in 1:substeps
            rk4_step!(q, dt, rom_rhs!, k1, k2, k3, k4, stage)
        end
        blew_up = !all(isfinite, q) || maximum(abs, q) > max_abs
        if blew_up
            return Q
        end
        Q[:, n+1] .= q
    end
    return Q
end

"""
Trajectory of the ROM chosen by `select_lambda_stability`: integrate O from q0, or all NaN
if no λ was stable (λ = NaN).
"""
function integrate_selected_rom(λ, O, q0, ts)
    r = length(q0)
    if isnan(λ)
        return fill(NaN, r, length(ts))
    end
    return integrate_rom(split_operators(O, r)..., q0, ts)
end

"Relative Frobenius error of the lifted ROM over the columns `idx` (Inf if it blew up)."
function rom_error(V, Q, U, idx)
    if any(isnan, view(Q, :, idx))
        return Inf
    end
    return norm(V * Q[:, idx] - U[:, idx]) / norm(U[:, idx])
end

"Relative error of the POD projection V Vᵀ U over the columns `idx` (best possible in this basis)."
pod_projection_error(V, U, idx) = norm(V * (V' * U[:, idx]) - U[:, idx]) / norm(U[:, idx])

# ------------------------------------------------------------------ plotting helpers
default(fontfamily = "sans-serif", linewidth = 2, framestyle = :box, grid = true,
        legendfontsize = 8, guidefontsize = 10, titlefontsize = 10, size = (640, 420), dpi = 150)

"Values ready for a log-scale axis: non-positive or non-finite entries become NaN (gaps)."
for_log_plot(values; floor = 1e-17) = [isfinite(x) && x > 0 ? max(x, floor) : NaN for x in values]

"Save a figure to analysis/output/ and print where it went."
function savefig_here(plt, name)
    path = joinpath(FIGDIR, name)
    savefig(plt, path)
    println("  figure → ", relpath(path, dirname(@__DIR__)))
    return path
end
