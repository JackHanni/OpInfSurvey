# Runs the analyses from docs/project_plan.html on the Burgers' problem and
# writes figures and a text summary (results.txt) to analysis/output/.
#
#   julia analysis/run_all.jl          (≈ 3 min, mostly Plots compiling)
#
# Training window t ∈ [0, 2] (the same data as ../burgers/output/burgers_data.jld2); the FOM is
# also run to t = 4 so predictions beyond the training window can be scored.

include(joinpath(@__DIR__, "common.jl"))

const RESULTS_FILE = open(joinpath(FIGDIR, "results.txt"), "w")
"Print to the terminal and to results.txt."
report(args...) = (println(args...); println(RESULTS_FILE, args...); flush(RESULTS_FILE))

const R_VALUES = 2:18                        # r + r(r+1)/2 ≤ 201 snapshots needs r ≤ 18
const LAMBDA_GRID = 10.0 .^ range(-12, 2, length = 57)
const R_REGULARIZATION = 15                  # r used for the regularization study (as in ../burgers/opInf.jl)

data = burgers_data()
(; xs, ts, U, dUdt, train_idx, dx, ν) = data
predict_idx = (train_idx[end]+1):length(ts)
U_train = U[:, train_idx]
dt = ts[2] - ts[1]
n_points = length(xs)
snapshot_singular_values = svd(U_train).S
report("Burgers' ν = $ν, N = $n_points, $(length(train_idx)) training snapshots (t ≤ $(data.t_train)), ",
       "$(length(predict_idx)) prediction snapshots (t ≤ $(ts[end]))\n")

"Burgers' right-hand side evaluated at the lifted reduced state V q̂, projected back: Vᵀ f(V q̂)."
reprojected_rhs(V, q̂) = V' * burgers_rhs!(zeros(n_points), V * q̂, ν, dx)

"""
Everything the analyses need for one r: POD basis, reduced snapshots, data matrix, the
Galerkin operators (the reference answer) and three right-hand sides: projected exact FOM
derivatives, finite differences of Q̂, and re-projected data.
"""
function reduced_problem(r)
    V = pod_basis(U_train, r)
    Q̂ = V' * U_train
    D = data_matrix(Q̂)
    Â_galerkin, Ĥ_galerkin = galerkin_operators(V, ν, dx)
    R_exact  = Matrix((V' * dUdt[:, train_idx])')
    R_fd     = Matrix(fd_time_derivative(Q̂, dt)')
    R_reproj = Matrix(reduce(hcat, [reprojected_rhs(V, q̂) for q̂ in eachcol(Q̂)])')
    return (; r, V, Q̂, D, O_galerkin = [Â_galerkin Ĥ_galerkin], R_exact, R_fd, R_reproj)
end
problems = Dict(r => reduced_problem(r) for r in R_VALUES)

# ================================================================ Q1 recovery
report("== Q1  Operator recovery: ‖Ô − Ô_Galerkin‖/‖Ô_Galerkin‖ (QR, λ = 0) ==")
report(@sprintf("%4s %12s %12s %12s", "r", "re-projected", "exact dU/dt", "FD dQ̂/dt"))
recovery_err = Dict(k => Float64[] for k in (:reproj, :exact, :fd))
for r in R_VALUES
    problem = problems[r]
    for (key, R) in ((:reproj, problem.R_reproj), (:exact, problem.R_exact), (:fd, problem.R_fd))
        push!(recovery_err[key], relative_error(solve_qr(problem.D, R), problem.O_galerkin))
    end
    report(@sprintf("%4d %12.2e %12.2e %12.2e", r, recovery_err[:reproj][end], recovery_err[:exact][end], recovery_err[:fd][end]))
end
plt = plot(R_VALUES, for_log_plot(recovery_err[:reproj]), yscale = :log10, marker = :circle, label = "re-projected data",
           xlabel = "r", ylabel = "‖Ô − Ô_Gal‖ / ‖Ô_Gal‖", title = "Q1: does OpInf recover the intrusive operators?",
           legend = :right)
plot!(plt, R_VALUES, for_log_plot(recovery_err[:exact]), marker = :utriangle, label = "projected exact dU/dt")
plot!(plt, R_VALUES, for_log_plot(recovery_err[:fd]), marker = :square, label = "FD of snapshots (Δt = $dt)")
savefig_here(plt, "q1_operator_recovery.png")
report("""
Re-projected data satisfy the reduced Galerkin equations exactly, so OpInf
recovers Â, Ĥ up to rounding amplified by κ(D). Projected FOM derivatives
include the closure error Vᵀf(u) − Vᵀf(VVᵀu), so the operators differ from
Galerkin by an amount that shrinks only as the projection error shrinks.
""")

# ================================================================ Q2 conditioning
report("== Q2  Conditioning of D ==")
report(@sprintf("%4s %6s %12s %12s %12s %14s", "r", "cols", "κ(D)", "σ1/σr", "(σ1/σr)²", "κ(D col-scaled)"))
r_all = 1:last(R_VALUES)
κ_D = Float64[]           # condition number of the data matrix
κ_linear = Float64[]      # σ₁/σᵣ of the snapshots = κ of the linear block of D
κ_scaled = Float64[]      # κ(D) after scaling every column to unit norm
for r in r_all
    if r in R_VALUES
        D = problems[r].D
    else
        D = data_matrix(pod_basis(U_train, r)' * U_train)
    end
    singular_values_D = svdvals(D)
    D_unit_columns = D ./ norm.(eachcol(D))'
    push!(κ_D, singular_values_D[1] / singular_values_D[end])
    push!(κ_linear, snapshot_singular_values[1] / snapshot_singular_values[r])
    push!(κ_scaled, cond(D_unit_columns))
    report(@sprintf("%4d %6d %12.2e %12.2e %12.2e %14.2e", r, n_unknowns(r), κ_D[end], κ_linear[end], κ_linear[end]^2, κ_scaled[end]))
end
plt = plot(r_all, κ_D, yscale = :log10, marker = :circle, label = "κ(D)", xlabel = "r", ylabel = "condition number",
           title = "Q2: κ(D) vs r", legend = :topleft)
plot!(plt, r_all, κ_linear .^ 2, ls = :dash, label = "(σ₁/σᵣ)²  (snapshot singular values)")
plot!(plt, r_all, κ_linear, ls = :dot, label = "σ₁/σᵣ = κ(linear block)")
plot!(plt, r_all, κ_scaled, marker = :diamond, label = "κ(D), columns scaled to unit norm")
hline!(plt, [1 / eps()], color = :gray, ls = :dashdot, label = "1/ε_mach")
savefig_here(plt, "q2_conditioning.png")
plt = plot(xlabel = "index i", ylabel = "σᵢ(D) / σ₁(D)", yscale = :log10, title = "Singular values of D",
           legend = :bottomleft)
for r in (6, 12, 18)
    singular_values_D = svdvals(problems[r].D)
    plot!(plt, singular_values_D ./ singular_values_D[1], marker = :circle, ms = 2, label = "r = $r ($(n_unknowns(r)) cols)")
end
savefig_here(plt, "q2_singular_values_D.png")
report("""
κ(D) grows much faster than (σ₁/σᵣ)², and scaling the columns to unit norm
only helps for small r. The rest is genuine near-dependence: with a single
trajectory every column of D is a smooth function of the one parameter t,
so the r + r(r+1)/2 columns are numerically dependent (like a Vandermonde
matrix). See Q6 for what more trajectories do.
""")

# ================================================================ Q3 solvers
report("== Q3  Solvers on a consistent problem with known answer (re-projected data, λ = 0) ==")
report(@sprintf("%4s %10s %12s %12s %12s %10s %10s %10s", "r", "κ(D)", "err NE", "err QR", "err SVD", "t NE", "t QR", "t SVD"))
solvers = ((:ne, solve_normal), (:qr, solve_qr), (:svd, solve_svd))
solver_err = Dict(key => Float64[] for (key, _) in solvers)
solver_seconds = Dict(key => Float64[] for (key, _) in solvers)
κ_solver = Float64[]
for r in R_VALUES
    problem = problems[r]
    push!(κ_solver, cond(problem.D))
    for (key, solver) in solvers
        O = try
            solver(problem.D, problem.R_reproj)
        catch                          # Cholesky fails once DᵀD is not numerically positive definite
            nothing
        end
        if O === nothing
            push!(solver_err[key], NaN)
            push!(solver_seconds[key], NaN)
        else
            push!(solver_err[key], relative_error(O, problem.O_galerkin))
            push!(solver_seconds[key], minimum(@elapsed(solver(problem.D, problem.R_reproj)) for _ in 1:20))
        end
    end
    report(@sprintf("%4d %10.2e %12.2e %12.2e %12.2e %10.1e %10.1e %10.1e", r, κ_solver[end],
                    solver_err[:ne][end], solver_err[:qr][end], solver_err[:svd][end],
                    solver_seconds[:ne][end], solver_seconds[:qr][end], solver_seconds[:svd][end]))
end
n_cholesky_failures = count(isnan, solver_err[:ne])
if n_cholesky_failures > 0
    report("normal equations: Cholesky failed (DᵀD not numerically positive definite) for $n_cholesky_failures value(s) of r")
end
plt = plot(κ_solver, for_log_plot(solver_err[:ne]), xscale = :log10, yscale = :log10, seriestype = :scatter,
           label = "normal equations", xlabel = "κ(D)", ylabel = "relative operator error",
           title = "Q3: solver accuracy vs conditioning", legend = :topleft)
scatter!(plt, κ_solver, for_log_plot(solver_err[:qr]), marker = :utriangle, label = "Householder QR")
scatter!(plt, κ_solver, for_log_plot(solver_err[:svd]), marker = :diamond, label = "SVD")
κ_line = 10 .^ range(log10(minimum(κ_solver)), log10(maximum(κ_solver)), length = 50)
plot!(plt, κ_line, eps() .* κ_line, color = :gray, ls = :dash, label = "ε·κ")
plot!(plt, κ_line, min.(eps() .* κ_line .^ 2, 10.0), color = :black, ls = :dot, label = "ε·κ²")
savefig_here(plt, "q3_solver_accuracy.png")
report("""
Backward-stable QR and SVD give errors ~ ε·κ(D); forming DᵀD squares the
condition number, so the normal equations lose about twice as many digits
and break down entirely once κ(D) ≳ 1/√ε ≈ 7e7. All three are cheap here
(D is at most 201 × 189), so accuracy, not cost, decides.
""")

# ================================================================ Q4 regularization
report("== Q4  Choosing λ  (r = $R_REGULARIZATION, projected exact derivatives) ==")
regularization_problem = problems[R_REGULARIZATION]
svd_D = svd(regularization_problem.D)
n_snapshots = size(regularization_problem.D, 1)
residual_norm = Float64[]        # ‖D Oᵀ − R‖, x-axis of the L-curve
solution_norm = Float64[]        # ‖O‖, y-axis of the L-curve
gcv = Float64[]                  # generalized cross-validation function
reduced_train_err = Float64[]    # ROM error in reduced coordinates on the training window
stable = Bool[]                  # ROM stays bounded to the end of the prediction window
spectral_abscissa = Float64[]    # max Re λ(Â)
train_err = Float64[]
predict_err = Float64[]
max_abs = 1.5 * maximum(abs, regularization_problem.Q̂)
for λ in LAMBDA_GRID
    O = solve_svd(regularization_problem.D, regularization_problem.R_exact, λ; svd_D)
    filter_factors = svd_D.S .^ 2 ./ (svd_D.S .^ 2 .+ λ)
    residual = regularization_problem.D * O' - regularization_problem.R_exact
    push!(residual_norm, norm(residual))
    push!(solution_norm, norm(O))
    push!(gcv, norm(residual)^2 / (n_snapshots - sum(filter_factors))^2)
    Â, Ĥ = split_operators(O, R_REGULARIZATION)
    push!(spectral_abscissa, maximum(real, eigvals(Â)))
    Q = integrate_rom(Â, Ĥ, regularization_problem.Q̂[:, 1], ts; max_abs)
    is_stable = !any(isnan, Q)
    push!(stable, is_stable)
    push!(reduced_train_err, is_stable ? norm(Q[:, train_idx] - regularization_problem.Q̂) / norm(regularization_problem.Q̂) : NaN)
    push!(train_err, rom_error(regularization_problem.V, Q, U, train_idx))
    push!(predict_err, rom_error(regularization_problem.V, Q, U, predict_idx))
end
# L-curve corner = point of maximum curvature of (log ρ, log η) as a function of log λ
log_residual, log_solution, log_λ = log.(residual_norm), log.(solution_norm), log.(LAMBDA_GRID)
"Centered finite-difference derivative with respect to log λ (one-sided at the ends)."
d_dlogλ(y) = [(y[min(i+1, end)] - y[max(i-1, 1)]) / (log_λ[min(i+1, end)] - log_λ[max(i-1, 1)]) for i in eachindex(y)]
dρ, dη = d_dlogλ(log_residual), d_dlogλ(log_solution)
d2ρ, d2η = d_dlogλ(dρ), d_dlogλ(dη)
curvature = (dρ .* d2η .- d2ρ .* dη) ./ (dρ .^ 2 .+ dη .^ 2) .^ 1.5
i_lcurve    = argmax(curvature[2:end-1]) + 1
i_gcv       = argmin(gcv)
i_stability = argmin([is_stable ? err : Inf for (is_stable, err) in zip(stable, reduced_train_err)])
report(@sprintf("%-22s %10s %12s %12s %12s", "method", "λ", "train err", "predict err", "max Re λ(Â)"))
for (name, i) in (("L-curve corner", i_lcurve), ("GCV", i_gcv), ("stability-based", i_stability))
    report(@sprintf("%-22s %10.1e %12.2e %12.2e %12.2e", name, LAMBDA_GRID[i], train_err[i], predict_err[i], spectral_abscissa[i]))
end
report(@sprintf("%-22s %10s %12.2e %12.2e", "POD projection floor", "—",
                pod_projection_error(regularization_problem.V, U, train_idx), pod_projection_error(regularization_problem.V, U, predict_idx)))
Â_galerkin = regularization_problem.O_galerkin[:, 1:R_REGULARIZATION]
report(@sprintf("Galerkin Â: max Re eigenvalue %.2e;  %d of %d λ values give a ROM that stays bounded to t = %g",
                maximum(real, eigvals(Â_galerkin)), count(stable), length(LAMBDA_GRID), ts[end]))

plt = plot(residual_norm, solution_norm, xscale = :log10, yscale = :log10, marker = :circle, ms = 2,
           label = "λ from 1e-12 to 1e2", xlabel = "residual ‖D Ôᵀ − Rᵀ‖", ylabel = "‖Ô‖",
           title = "Q4: L-curve (r = $R_REGULARIZATION)", legend = :topright)
for (name, i, marker) in (("L-curve corner", i_lcurve, :star5), ("GCV", i_gcv, :diamond), ("stability", i_stability, :square))
    scatter!(plt, [residual_norm[i]], [solution_norm[i]], marker = marker, ms = 8,
             label = @sprintf("%s  λ = %.0e", name, LAMBDA_GRID[i]))
end
savefig_here(plt, "q4_lcurve.png")
plt = plot(LAMBDA_GRID, for_log_plot(train_err), xscale = :log10, yscale = :log10, label = "ROM error, training window",
           xlabel = "λ", ylabel = "relative error", title = "Q4: ROM error vs λ (r = $R_REGULARIZATION); gaps = unstable ROM",
           legend = :topleft)
plot!(plt, LAMBDA_GRID, for_log_plot(predict_err), ls = :dash, label = "ROM error, prediction t ∈ (2, 4]")
plot!(plt, LAMBDA_GRID, gcv ./ maximum(gcv), ls = :dot, label = "GCV(λ), normalized")
vline!(plt, [LAMBDA_GRID[i_lcurve]], color = :gray, ls = :dash, label = "L-curve")
vline!(plt, [LAMBDA_GRID[i_gcv]], color = :purple, ls = :dashdot, label = "GCV")
vline!(plt, [LAMBDA_GRID[i_stability]], color = :green, label = "stability")
savefig_here(plt, "q4_error_vs_lambda.png")
plt = plot(LAMBDA_GRID, spectral_abscissa, xscale = :log10, label = "OpInf", xlabel = "λ", ylabel = "max Re eigenvalue of Â",
           title = "Q4: stability of the learned linear operator (r = $R_REGULARIZATION)", legend = :topright)
hline!(plt, [maximum(real, eigvals(Â_galerkin))], ls = :dash, label = "Galerkin")
hline!(plt, [0.0], color = :black, lw = 1, label = "")
savefig_here(plt, "q4_spectral_abscissa.png")
plt = scatter(real(eigvals(Â_galerkin)), imag(eigvals(Â_galerkin)), marker = :x, ms = 6, label = "Galerkin",
              xlabel = "Re", ylabel = "Im", title = "Eigenvalues of Â (r = $R_REGULARIZATION)", legend = :bottomleft)
for (label, i, marker) in (("OpInf λ = $(@sprintf("%.0e", LAMBDA_GRID[1]))", 1, :circle), ("OpInf, stability λ", i_stability, :utriangle))
    Â = split_operators(solve_svd(regularization_problem.D, regularization_problem.R_exact, LAMBDA_GRID[i]; svd_D), R_REGULARIZATION)[1]
    eigenvalues = eigvals(Â)
    scatter!(plt, real(eigenvalues), imag(eigenvalues), marker = marker, label = label)
end
savefig_here(plt, "q4_eigenvalues.png")
if i_gcv == 1
    report("GCV is minimized at the smallest λ on the grid, i.e. it fails to find an interior minimum.")
end
report("""
GCV and the L-curve only look at the regression; the stability criterion
looks at the ROM it produces (using training data only), which is what
actually matters for prediction.
""")

# ================================================================ Q5 robustness + accuracy vs r
report("== Q5a  ROM accuracy vs r (no noise; λ by stability criterion) ==")
report(@sprintf("%4s %11s %11s %11s %11s %11s %11s %11s %11s", "r", "POD tr", "POD pr", "Gal tr", "Gal pr",
                "OpI-ex tr", "OpI-ex pr", "OpI-FD tr", "OpI-FD pr"))
# keys: <model>_tr (training window) and <model>_pr (prediction window)
accuracy_columns = (:pod_tr, :pod_pr, :gal_tr, :gal_pr, :ex_tr, :ex_pr, :fd_tr, :fd_pr)
accuracy = Dict(key => Float64[] for key in accuracy_columns)
for r in R_VALUES
    problem = problems[r]
    push!(accuracy[:pod_tr], pod_projection_error(problem.V, U, train_idx))
    push!(accuracy[:pod_pr], pod_projection_error(problem.V, U, predict_idx))
    Q_galerkin = integrate_rom(split_operators(problem.O_galerkin, r)..., problem.Q̂[:, 1], ts;
                               max_abs = 1e3 * maximum(abs, problem.Q̂))
    push!(accuracy[:gal_tr], rom_error(problem.V, Q_galerkin, U, train_idx))
    push!(accuracy[:gal_pr], rom_error(problem.V, Q_galerkin, U, predict_idx))
    for (model, R) in ((:ex, problem.R_exact), (:fd, problem.R_fd))
        λ, O, _ = select_lambda_stability(problem.D, R, problem.Q̂, ts, train_idx, LAMBDA_GRID)
        Q = integrate_selected_rom(λ, O, problem.Q̂[:, 1], ts)
        push!(accuracy[Symbol(model, :_tr)], rom_error(problem.V, Q, U, train_idx))
        push!(accuracy[Symbol(model, :_pr)], rom_error(problem.V, Q, U, predict_idx))
    end
    report(@sprintf("%4d", r), join(@sprintf(" %11.2e", accuracy[key][end]) for key in accuracy_columns))
end
plt = plot(xlabel = "r", ylabel = "relative error", yscale = :log10, legend = :bottomleft,
           title = "Q5: ROM error vs r  (solid: t ≤ 2 training,  dashed: 2 < t ≤ 4 prediction)")
for (model, label, color) in ((:pod, "POD projection", :gray), (:gal, "Galerkin", :black),
                              (:ex, "OpInf, exact dU/dt", :blue), (:fd, "OpInf, FD derivatives", :orange))
    plot!(plt, R_VALUES, for_log_plot(accuracy[Symbol(model, :_tr)]), color = color, marker = :circle, ms = 3, label = label)
    plot!(plt, R_VALUES, for_log_plot(accuracy[Symbol(model, :_pr)]), color = color, ls = :dash, label = "")
end
savefig_here(plt, "q5_error_vs_r.png")

report("\n== Q5b  Noise robustness (Gaussian noise, std = η·max|u|; FD derivatives; λ by stability; median of 3 seeds) ==")
noise_levels = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
r_noise = (6, 10, 15)
report(@sprintf("%9s %4s %11s %11s %10s", "η", "r", "train err", "pred err", "λ chosen"))
noise_err = Dict((r, window) => Float64[] for r in r_noise, window in (:tr, :pr))
median_of_three(values) = sort(values)[2]
for noise_level in noise_levels, r in r_noise
    seed_train_err = Float64[]
    seed_predict_err = Float64[]
    seed_λ = Float64[]
    for seed in 1:3
        U_noisy = U_train .+ noise_level * maximum(abs, U_train) .* randn(MersenneTwister(seed), size(U_train))
        U_noisy[1, :] .= 0                     # boundary values are known exactly
        U_noisy[end, :] .= 0
        V = pod_basis(U_noisy, r)
        Q̂ = V' * U_noisy
        D = data_matrix(Q̂)
        R = Matrix(fd_time_derivative(Q̂, dt)')
        λ, O, _ = select_lambda_stability(D, R, Q̂, ts, train_idx, LAMBDA_GRID)
        Q = integrate_selected_rom(λ, O, Q̂[:, 1], ts)
        push!(seed_train_err, rom_error(V, Q, U, train_idx))
        push!(seed_predict_err, rom_error(V, Q, U, predict_idx))
        push!(seed_λ, λ)
    end
    push!(noise_err[(r, :tr)], median_of_three(seed_train_err))
    push!(noise_err[(r, :pr)], median_of_three(seed_predict_err))
    report(@sprintf("%9.0e %4d %11.2e %11.2e %10.0e", noise_level, r,
                    median_of_three(seed_train_err), median_of_three(seed_predict_err), median_of_three(seed_λ)))
end
noise_levels_plotted = [max(noise_level, 1e-6) for noise_level in noise_levels]   # η = 0 can't go on a log axis
plt = plot(xlabel = "noise level η  (η = 0 plotted at 1e-6)", ylabel = "relative error vs clean FOM",
           xscale = :log10, yscale = :log10, legend = :topleft,
           title = "Q5: noise robustness (solid: training,  dashed: prediction)")
for (r, color) in zip(r_noise, (:blue, :orange, :green))
    plot!(plt, noise_levels_plotted, for_log_plot(noise_err[(r, :tr)]), color = color, marker = :circle, label = "r = $r")
    plot!(plt, noise_levels_plotted, for_log_plot(noise_err[(r, :pr)]), color = color, ls = :dash, marker = :utriangle, label = "")
end
plot!(plt, noise_levels_plotted, noise_levels_plotted, color = :gray, ls = :dot, label = "error = η")
savefig_here(plt, "q5_noise.png")

# ================================================================ Q6 more trajectories
report("\n== Q6  Remedy: train on several trajectories  u0 = sin(πx) + c·sin(2πx), c ∈ {0, 0.5, −0.5, 0.25} ==")
sin2_amplitudes = (0.0, 0.5, -0.5, 0.25)        # c in the initial condition above; c = 0 is the original
trajectories = [solve_burgers(merge(default_params, (t_max = data.t_train, u0 = x -> sin(π * x) + c * sin(2π * x),
                                                     make_plot = false, make_gif = false))) for c in sin2_amplitudes]
U_all_trajectories    = reduce(hcat, [trajectory.U for trajectory in trajectories])
dUdt_all_trajectories = reduce(hcat, [trajectory.dUdt for trajectory in trajectories])
@assert trajectories[1].U ≈ U_train
report(@sprintf("%4s %11s %11s | %11s %11s | %11s %11s | %11s %11s", "r", "κ(D) 1tr", "κ(D) 4tr",
                "rec-rp 1tr", "rec-rp 4tr", "rec-ex 1tr", "rec-ex 4tr", "pred 1tr", "pred 4tr"))
# keys end in 1 (one trajectory) or 4 (four): κ = cond(D), rp/ex = recovery error with re-projected /
# exact-derivative data, tr/pr = ROM error on the training / prediction window
q6 = Dict(key => Float64[] for key in (:κ1, :κ4, :rp1, :rp4, :ex1, :ex4, :tr1, :tr4, :pr1, :pr4))
for r in R_VALUES
    one_trajectory = problems[r]
    V = pod_basis(U_all_trajectories, r)
    Q̂_per_trajectory = [V' * trajectory.U for trajectory in trajectories]
    D = reduce(vcat, [data_matrix(Q̂) for Q̂ in Q̂_per_trajectory])
    R_exact = Matrix((V' * dUdt_all_trajectories)')
    R_reproj = Matrix(reduce(hcat, [reprojected_rhs(V, q̂) for Q̂ in Q̂_per_trajectory for q̂ in eachcol(Q̂)])')
    O_galerkin = hcat(galerkin_operators(V, ν, dx)...)
    push!(q6[:κ1], cond(one_trajectory.D))
    push!(q6[:κ4], cond(D))
    push!(q6[:rp1], relative_error(solve_qr(one_trajectory.D, one_trajectory.R_reproj), one_trajectory.O_galerkin))
    push!(q6[:rp4], relative_error(solve_qr(D, R_reproj), O_galerkin))
    push!(q6[:ex1], relative_error(solve_qr(one_trajectory.D, one_trajectory.R_exact), one_trajectory.O_galerkin))
    push!(q6[:ex4], relative_error(solve_qr(D, R_exact), O_galerkin))
    # predict the c = 0 trajectory with ROMs trained on one or on four trajectories
    for (n_traj, D_fit, R_fit, V_fit, Q̂_start) in ((1, one_trajectory.D, one_trajectory.R_exact, one_trajectory.V, one_trajectory.Q̂),
                                                   (4, D, R_exact, V, Q̂_per_trajectory[1]))
        λ, O, _ = select_lambda_stability(D_fit, R_fit, Q̂_start, ts, train_idx, LAMBDA_GRID)
        Q = integrate_selected_rom(λ, O, Q̂_start[:, 1], ts)
        push!(q6[Symbol(:tr, n_traj)], rom_error(V_fit, Q, U, train_idx))
        push!(q6[Symbol(:pr, n_traj)], rom_error(V_fit, Q, U, predict_idx))
    end
    report(@sprintf("%4d %11.2e %11.2e | %11.2e %11.2e | %11.2e %11.2e | %11.2e %11.2e", r, q6[:κ1][end], q6[:κ4][end],
                    q6[:rp1][end], q6[:rp4][end], q6[:ex1][end], q6[:ex4][end], q6[:pr1][end], q6[:pr4][end]))
end
plt = plot(R_VALUES, q6[:κ1], yscale = :log10, marker = :circle, label = "1 trajectory", xlabel = "r", ylabel = "κ(D)",
           title = "Q6: conditioning with more training trajectories", legend = :topleft)
plot!(plt, R_VALUES, q6[:κ4], marker = :utriangle, label = "4 trajectories")
hline!(plt, [1 / eps()], color = :gray, ls = :dashdot, label = "1/ε_mach")
savefig_here(plt, "q6_conditioning_trajectories.png")
plt = plot(R_VALUES, for_log_plot(q6[:rp1]), yscale = :log10, marker = :circle, color = :blue, label = "1 traj, re-projected",
           xlabel = "r", ylabel = "‖Ô − Ô_Gal‖ / ‖Ô_Gal‖", title = "Q6: operator recovery with more trajectories",
           legend = :topleft)
plot!(plt, R_VALUES, for_log_plot(q6[:rp4]), marker = :utriangle, color = :blue, ls = :dash, label = "4 traj, re-projected")
plot!(plt, R_VALUES, for_log_plot(q6[:ex1]), marker = :circle, color = :orange, label = "1 traj, exact dU/dt")
plot!(plt, R_VALUES, for_log_plot(q6[:ex4]), marker = :utriangle, color = :orange, ls = :dash, label = "4 traj, exact dU/dt")
savefig_here(plt, "q6_recovery_trajectories.png")
plt = plot(R_VALUES, for_log_plot(accuracy[:gal_pr]), yscale = :log10, color = :black, label = "Galerkin", xlabel = "r",
           ylabel = "prediction error, 2 < t ≤ 4", title = "Q6: prediction error of the c = 0 trajectory",
           legend = :bottomleft)
plot!(plt, R_VALUES, for_log_plot(q6[:pr1]), marker = :circle, color = :blue, label = "OpInf, 1 trajectory")
plot!(plt, R_VALUES, for_log_plot(q6[:pr4]), marker = :utriangle, color = :green, label = "OpInf, 4 trajectories")
savefig_here(plt, "q6_prediction_trajectories.png")

close(RESULTS_FILE)
println("\nsummary → analysis/output/results.txt")
