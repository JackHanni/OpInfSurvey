# OpInf on a published binary-black-hole solution of the Einstein equations (SXS:BBH:0305).
#
# Data: run fetch_sxs.py first. It writes data/sxs_bbh_0305.h5 with the gravitational-wave
# strain h(t, θ, φ) = Σ h_lm(t) ₋₂Y_lm(θ, φ) as 77 complex modes (ℓ = 2..8) and their time
# derivatives. The state is x(t) = [Re h_lm; Im h_lm]. The spin-weighted harmonics are
# orthonormal, so 2-norms of x equal L² norms of h on the sphere. The m = 0 modes are left out
# by default (see load_sxs).
#
# Experiments (train on one window, predict forward, compare with the true solution):
#   ringdown: after merger, where the remnant black hole rings down. Also checks that the learned
#             Â contains the Kerr quasinormal-mode frequency, and when the signal becomes linear.
#   inspiral: the last orbits before merger, where the frequency chirps upward.
# Models: linear  dx̂/dt = Â x̂, and quadratic  dx̂/dt = Â x̂ + Ĥ (x̂ ⊗ x̂).
#
# Notation follows the README: V is the POD basis, X the snapshot matrix, D the OpInf data
# matrix, R the reduced time derivatives, O = [Â Ĥ] the learned operators, λ the Tikhonov
# parameter, r the ROM dimension.
#
# Run:  julia einstein/opinf_sxs.jl

using JLD2, LinearAlgebra, Printf, Plots

const HERE = @__DIR__
const OUTDIR = mkpath(joinpath(HERE, "output"))
const LOG_FILE = Ref{IO}(devnull)   # set to output/sxs_log.txt when this file is run as a script

"Print to the terminal and to the log file."
function logln(parts...)
    println(parts...)
    println(LOG_FILE[], parts...)
    flush(LOG_FILE[])
end

const LAMBDA_GRID = [0.0; 10.0 .^ (-12:1:-2)]

# ---------------------------------------------------------------- data
const SXS_FILE = joinpath(HERE, "data", "sxs_bbh_0305.h5")

"""
    load_sxs(path = SXS_FILE; drop_m0 = true) -> (; t, X, dXdt, ell, m)

Load the strain written by `fetch_sxs.py`.

- `t`:      times (length n_t), in units of M
- `X`:      states x(t) = [Re h_lm; Im h_lm], size (2·n_modes, n_t)
- `dXdt`:   their time derivatives, same layout
- `ell, m`: the (ℓ, m) of each mode, so row i of X is Re h for mode i and row n_modes + i is Im h

With `drop_m0 = true` the m = 0 modes are left out. After merger they are dominated by the
gravitational-wave memory, a non-oscillating offset (|h20| ≈ 0.08) that swamps the decaying
ringdown in any norm, and they are the least reliable modes of extrapolated SXS waveforms.
"""
function load_sxs(path = SXS_FILE; drop_m0 = true)
    jldopen(path, "r") do file
        t   = vec(file["t"])
        ell = vec(file["ell"])
        m   = vec(file["m"])
        if drop_m0
            modes = findall(m .!= 0)
        else
            modes = eachindex(m)
        end

        stack_re_im(name) = vcat(file["$(name)_re"][modes, :], file["$(name)_im"][modes, :])

        return (; t, X = stack_re_im("h"), dXdt = stack_re_im("hdot"), ell = ell[modes], m = m[modes])
    end
end

# ---------------------------------------------------------------- OpInf pieces
"The r(r+1)/2 distinct products x_i x_j (j ≤ i): x ⊗ x without the repeated entries."
compact_kron(x) = [x[i] * x[j] for i in eachindex(x) for j in 1:i]

"Number of unknowns per row of O, i.e. the number of columns of D."
n_unknowns(r, quadratic) = quadratic ? r + r * (r + 1) ÷ 2 : r

"""
OpInf data matrix D, one row per snapshot: [q̂ᵀ] for the linear model,
[q̂ᵀ  compact_kron(q̂)ᵀ] for the quadratic one. `reduced_states` has one snapshot per column.
"""
function data_matrix(reduced_states, quadratic)
    linear_part = Matrix(reduced_states')
    if !quadratic
        return linear_part
    end
    quadratic_part = reduce(hcat, compact_kron.(eachcol(reduced_states)))'
    return hcat(linear_part, quadratic_part)
end

"Tikhonov-regularized least squares min ‖D Oᵀ − R‖² + λ‖O‖², via Householder QR of [D; √λ I]."
function solve_least_squares(D, R, λ)
    n_cols = size(D, 2)
    D_stacked = vcat(D, sqrt(λ) * I)
    R_stacked = vcat(R, zeros(n_cols, size(R, 2)))
    qr_D = qr(D_stacked)
    Oᵀ = UpperTriangular(qr_D.R) \ (qr_D.Q' * R_stacked)[1:n_cols, :]
    return Matrix(Oᵀ')
end

"Right-hand side x̂ ↦ Â x̂ (+ Ĥ (x̂ ⊗ x̂)) of the learned ROM, with O = [Â Ĥ]."
function rom_rhs(O, r, quadratic)
    Â = O[:, 1:r]
    if !quadratic
        return x̂ -> Â * x̂
    end
    Ĥ = O[:, r+1:end]
    return x̂ -> Â * x̂ + Ĥ * compact_kron(x̂)
end

"""
RK4 from `x0` over `times`, with `substeps` steps per interval. Returns the state at each time,
one per column. If the state blows up (non-finite or norm ≥ `max_norm`), the integration stops
and the remaining columns stay NaN.
"""
function integrate(rhs, x0, times; substeps = 5, max_norm = Inf)
    states = fill(NaN, length(x0), length(times))
    x = copy(x0)
    states[:, 1] = x
    for k in 2:length(times)
        dt = (times[k] - times[k-1]) / substeps
        for _ in 1:substeps
            k1 = rhs(x); k2 = rhs(x + dt/2*k1); k3 = rhs(x + dt/2*k2); k4 = rhs(x + dt*k3)
            x = x + dt/6*(k1 + 2k2 + 2k3 + k4)
        end
        blew_up = !all(isfinite, x) || norm(x) >= max_norm
        if blew_up
            return states
        end
        states[:, k] = x
    end
    return states
end

"‖A − B‖ / ‖B‖, or Inf if A contains NaN (a ROM that blew up)."
relative_error(A, B) = any(isnan, A) ? Inf : norm(A - B) / norm(B)

"Relative error of each column (time) separately."
relative_error_per_time(A, B) = [norm(A[:, k] - B[:, k]) / norm(B[:, k]) for k in axes(B, 2)]

"Learn O = [Â Ĥ] at a given λ from the snapshots in `snapshot_idx`."
function fit_operators(V, X, dXdt, snapshot_idx, quadratic, λ)
    D = data_matrix(V' * X[:, snapshot_idx], quadratic)
    R = (V' * dXdt[:, snapshot_idx])'
    return solve_least_squares(D, R, λ)
end

"""
Choose λ using the training window only: fit on the first 75 % of it, run the ROM through the
whole window, and keep the λ with the smallest error on the held-out last 25 %.
"""
function choose_lambda(V, X, dXdt, t, train_idx, r, quadratic, λ_grid)
    n_fit = round(Int, 0.75 * length(train_idx))
    fit_idx     = train_idx[1:n_fit]
    holdout_idx = train_idx[n_fit:end]
    max_norm = 10 * maximum(norm.(eachcol(V' * X[:, train_idx])))

    best_err, best_λ = Inf, λ_grid[1]
    for λ in λ_grid
        O = fit_operators(V, X, dXdt, fit_idx, quadratic, λ)
        reduced_rom = integrate(rom_rhs(O, r, quadratic), V' * X[:, fit_idx[1]],
                                t[fit_idx[1]:holdout_idx[end]]; max_norm)
        # reduced_rom starts at the first training snapshot, so its columns n_fit:end are
        # the times holdout_idx.
        holdout_err = relative_error(V * reduced_rom[:, n_fit:end], X[:, holdout_idx])
        if holdout_err < best_err
            best_err, best_λ = holdout_err, λ
        end
    end
    return best_λ
end

# ---------------------------------------------------------------- one experiment
"""
Train linear or quadratic OpInf ROMs on `train_idx` for each r in `r_values`, predict over
`predict_idx`, log a table, and return one named tuple of results per r.
"""
function run_experiment(name, t, X, dXdt, train_idx, predict_idx, r_values, quadratic; λ_grid = LAMBDA_GRID)
    model = quadratic ? "quadratic" : "linear"
    logln("\n== $name, $model model ==")
    logln(@sprintf("train t ∈ [%.1f, %.1f] (%d snapshots), predict t ∈ (%.1f, %.1f]",
                   t[train_idx[1]], t[train_idx[end]], length(train_idx), t[predict_idx[1]], t[predict_idx[end]]))
    logln(@sprintf("%4s %5s %10s %10s %11s %11s %11s %12s", "r", "cols", "σ_r/σ_1",
                   "λ", "κ(D)", "train err", "pred err", "POD floor pr"))
    pod_modes, singular_values, _ = svd(X[:, train_idx])
    window_idx = train_idx[1]:predict_idx[end]
    n_train = length(train_idx)
    results = []
    for r in r_values
        n_cols = n_unknowns(r, quadratic)
        if n_cols >= n_train      # underdetermined: more unknowns than snapshots
            continue
        end
        V = pod_modes[:, 1:r]
        λ = choose_lambda(V, X, dXdt, t, train_idx, r, quadratic, λ_grid)
        O = fit_operators(V, X, dXdt, train_idx, quadratic, λ)
        κ = cond(data_matrix(V' * X[:, train_idx], quadratic))

        max_norm = 10 * maximum(norm.(eachcol(X[:, window_idx]))) # mult by 10 for a little extra safety
        X_rom = V * integrate(rom_rhs(O, r, quadratic), V' * X[:, train_idx[1]], t[window_idx]; max_norm)
        train_err   = relative_error(X_rom[:, 1:n_train], X[:, train_idx])
        predict_err = relative_error(X_rom[:, n_train:end], X[:, train_idx[end]:predict_idx[end]])
        X_predict = X[:, predict_idx]
        pod_floor = norm(X_predict - V * (V' * X_predict)) / norm(X_predict)

        logln(@sprintf("%4d %5d %10.2e %10.1e %11.2e %11.2e %11.2e %12.2e",
                       r, n_cols, singular_values[r] / singular_values[1], λ, κ, train_err, predict_err, pod_floor))
        push!(results, (; r, λ, κ, train_err, predict_err, pod_floor, X_rom, O, V))
    end
    return results
end

# ---------------------------------------------------------------- QNM reference
# Remnant of SXS:BBH:0305 (Lev6 metadata.json): mass M_f = 0.95203, dimensionless spin χ_f = 0.69209.
# Fundamental (ℓ, m, n) = (2, 2, 0) Kerr quasinormal mode from the fits of
# Berti, Cardoso & Will, PRD 73, 064030 (2006), Table VIII:
#   M_f ω_R = 1.5251 − 1.1568 (1 − χ)^0.1292,   Q = 0.7000 + 1.4187 (1 − χ)^(−0.4990),   ω_I = ω_R / (2Q)
function kerr_qnm_220(Mf = 0.95203, χ = 0.69209)
    ωR = (1.5251 - 1.1568 * (1 - χ)^0.1292) / Mf
    Q = 0.7000 + 1.4187 * (1 - χ)^(-0.4990)
    return complex(-ωR / (2Q), ωR)   # growth rate of the state: x ∝ e^{(−ω_I ± i ω_R) t}
end

"Eigenvalue of the learned Â = O[:, 1:r] closest to `target`."
function closest_eigenvalue(O, r, target)
    eigenvalues = eigvals(O[:, 1:r])
    return eigenvalues[argmin(abs.(eigenvalues .- target))]
end

# ---------------------------------------------------------------- main
if abspath(PROGRAM_FILE) == @__FILE__
LOG_FILE[] = open(joinpath(OUTDIR, "sxs_log.txt"), "w")
t, X, dXdt, ell, m = load_sxs()
n_modes = length(ell)
row_h22 = findfirst((ell .== 2) .& (m .== 2))          # row of Re h22 in X; Im h22 is row n_modes + row_h22
amplitude_h22 = hypot.(X[row_h22, :], X[n_modes + row_h22, :])
t_peak = t[argmax(amplitude_h22)]
time_window(t_start, t_end) = findall(t_start .<= t .<= t_end)
nearest_time(t_target) = argmin(abs.(t .- t_target))

logln("SXS:BBH:0305 Lev6 strain (doi:10.26138/SXS:BBH:0305), Δt = $(t[2]-t[1]) M, ",
      "peak of |h22| at t_peak = $(round(t_peak, digits=1)) M")
logln("State: the $(n_modes) modes with m ≠ 0 (ℓ = 2..8), as [Re; Im] ∈ R^$(2n_modes). ",
      "Errors are relative 2-norm errors over the sphere.")
logln("λ is chosen from training data only (fit on the first 75 % of the window, forecast the last 25 %).")

# memory (m = 0) modes, reported but not modelled
_, X_with_m0, _, ell_with_m0, m_with_m0 = load_sxs(drop_m0 = false)
row_h20 = findfirst((ell_with_m0 .== 2) .& (m_with_m0 .== 0))
logln(@sprintf("Memory: h20 = %.3f at t = 3000, %.3f at t_peak, %.3f at t_peak + 50 (vs |h22| = %.1e there).",
               X_with_m0[row_h20, nearest_time(3000)], X_with_m0[row_h20, nearest_time(t_peak)],
               X_with_m0[row_h20, nearest_time(t_peak + 50)], amplitude_h22[nearest_time(t_peak + 50)]))
logln("  This non-oscillating offset would dominate every norm after merger, so m = 0 is left out.")

qnm = kerr_qnm_220()
logln(@sprintf("Kerr QNM (2,2,0) for the remnant: %.4f %+.4fi", real(qnm), imag(qnm)))

# ---- 1. Ringdown: train 20–50 M after the peak, predict to 80 M after the peak
ringdown_train   = time_window(t_peak + 20, t_peak + 50)
ringdown_predict = time_window(t_peak + 50.5, t_peak + 80)
ringdown_linear    = run_experiment("Ringdown", t, X, dXdt, ringdown_train, ringdown_predict, 2:2:20, false)
ringdown_quadratic = run_experiment("Ringdown", t, X, dXdt, ringdown_train, ringdown_predict, 2:2:8, true)

logln("\nLearned Â eigenvalue closest to the Kerr QNM (linear model, ringdown window above):")
logln(@sprintf("%4s %22s %12s", "r", "eigenvalue", "rel. diff"))
for result in ringdown_linear
    eigenvalue = closest_eigenvalue(result.O, result.r, qnm)
    logln(@sprintf("%4d %10.4f %+.4fi %12.2e", result.r, real(eigenvalue), imag(eigenvalue),
                   abs(eigenvalue - qnm) / abs(qnm)))
end

# ---- 2. When does the post-merger signal become linear? Slide the 30 M training window.
logln("\n== Ringdown start-time sweep: linear model, r = 2 and 16, 30 M training window ==")
logln(@sprintf("%12s %22s %10s %22s %10s", "start − t_peak", "eig (r = 2)", "rel diff", "eig (r = 16)", "rel diff"))
sweep_r_values = (2, 16)
start_offsets = [0, 5, 10, 15, 20, 25, 30]
# sweep_eigenvalues[i][j]: closest eigenvalue for start_offsets[i] and sweep_r_values[j]
sweep_eigenvalues = map(start_offsets) do offset
    train_idx = time_window(t_peak + offset, t_peak + offset + 30)
    pod_modes = svd(X[:, train_idx]).U
    eigenvalues = map(sweep_r_values) do r
        V = pod_modes[:, 1:r]
        λ = choose_lambda(V, X, dXdt, t, train_idx, r, false, LAMBDA_GRID)
        closest_eigenvalue(fit_operators(V, X, dXdt, train_idx, false, λ), r, qnm)
    end
    logln(@sprintf("%12d %10.4f %+.4fi %10.2e %10.4f %+.4fi %10.2e", offset,
                   real(eigenvalues[1]), imag(eigenvalues[1]), abs(eigenvalues[1] - qnm) / abs(qnm),
                   real(eigenvalues[2]), imag(eigenvalues[2]), abs(eigenvalues[2] - qnm) / abs(qnm)))
    eigenvalues
end

# ---- 3. Inspiral: train on [3300, 3500] M, predict [3500, 3600] M (merger is at ≈ 3692 M)
inspiral_train   = time_window(3300, 3500)
inspiral_predict = time_window(3500.5, 3600)
inspiral_linear    = run_experiment("Inspiral", t, X, dXdt, inspiral_train, inspiral_predict, 2:2:12, false)
inspiral_quadratic = run_experiment("Inspiral", t, X, dXdt, inspiral_train, inspiral_predict, 2:2:10, true)

# ---------------------------------------------------------------- figures
best_prediction(results) = results[argmin([result.predict_err for result in results])]

function plot_h22(name, train_idx, predict_idx, results_linear, results_quadratic, filename)
    window_idx = train_idx[1]:predict_idx[end]
    best_linear, best_quadratic = best_prediction(results_linear), best_prediction(results_quadratic)

    waveform_plot = plot(t[window_idx], X[row_h22, window_idx], lw = 3, c = :black, label = "true (SXS)",
                         ylabel = "Re h₂₂", title = "$name: h₂₂ mode")
    plot!(waveform_plot, t[window_idx], best_linear.X_rom[row_h22, :], lw = 1.5, ls = :dash, c = :blue,
          label = "OpInf linear, r = $(best_linear.r)")
    plot!(waveform_plot, t[window_idx], best_quadratic.X_rom[row_h22, :], lw = 1.5, ls = :dot, c = :red,
          label = "OpInf quadratic, r = $(best_quadratic.r)")
    vline!(waveform_plot, [t[train_idx[end]]], c = :gray, ls = :dash, label = "end of training")

    error_plot = plot(t[window_idx], relative_error_per_time(best_linear.X_rom, X[:, window_idx]),
                      yscale = :log10, c = :blue, lw = 1.5,
                      label = "linear", xlabel = "t / M", ylabel = "‖x_ROM − x‖ / ‖x‖")
    quadratic_error = relative_error_per_time(best_quadratic.X_rom, X[:, window_idx])
    quadratic_error[.!isfinite.(quadratic_error)] .= NaN      # leave gaps where the ROM blew up
    plot!(error_plot, t[window_idx], quadratic_error, c = :red, lw = 1.5, label = "quadratic")
    vline!(error_plot, [t[train_idx[end]]], c = :gray, ls = :dash, label = "")

    savefig(plot(waveform_plot, error_plot, layout = (2, 1), size = (800, 650)), joinpath(OUTDIR, filename))
end

function plot_error_vs_r(name, results_linear, results_quadratic, filename)
    clip(values) = min.(values, 10.0)
    r_of(results) = [result.r for result in results]
    predict_errors(results) = clip([result.predict_err for result in results])

    plt = plot(yscale = :log10, xlabel = "r", ylabel = "relative error (prediction window)",
               title = "$name: prediction error vs r", legend = :bottomleft)
    plot!(plt, r_of(results_linear), predict_errors(results_linear), marker = :circle, label = "OpInf linear")
    plot!(plt, r_of(results_quadratic), predict_errors(results_quadratic), marker = :square, label = "OpInf quadratic")
    plot!(plt, r_of(results_linear), [result.pod_floor for result in results_linear], ls = :dash, c = :black,
          label = "POD floor (training basis)")
    savefig(plt, joinpath(OUTDIR, filename))
end

function plot_qnm(filename)
    result_r16 = ringdown_linear[findfirst(result -> result.r == 16, ringdown_linear)]
    eigenvalues = eigvals(result_r16.O[:, 1:result_r16.r])
    spectrum_plot = scatter(real.(eigenvalues), imag.(eigenvalues), label = "eigenvalues of learned Â (r = 16)",
                            xlabel = "Re (decay rate)", ylabel = "Im (frequency)",
                            title = "Ringdown: learned spectrum vs Kerr QNM")
    scatter!(spectrum_plot, [real(qnm), real(qnm)], [imag(qnm), -imag(qnm)], marker = :star5, markersize = 10,
             c = :red, label = "Kerr (2,2,0) QNM, Berti et al. fit")

    relative_diff(j) = [abs(eigenvalues_at_start[j] - qnm) / abs(qnm) for eigenvalues_at_start in sweep_eigenvalues]
    sweep_plot = plot(start_offsets, relative_diff(1), marker = :circle, yscale = :log10,
                      label = "r = $(sweep_r_values[1])", xlabel = "training start − t_peak (M)",
                      ylabel = "|λ − ω_QNM| / |ω_QNM|", title = "Learned frequency vs training start")
    plot!(sweep_plot, start_offsets, relative_diff(2), marker = :square, label = "r = $(sweep_r_values[2])")

    savefig(plot(spectrum_plot, sweep_plot, layout = (1, 2), size = (1100, 450), margin = 5Plots.mm),
            joinpath(OUTDIR, filename))
end

plot_h22("Ringdown", ringdown_train, ringdown_predict, ringdown_linear, ringdown_quadratic, "ringdown_h22.png")
plot_error_vs_r("Ringdown", ringdown_linear, ringdown_quadratic, "ringdown_error_vs_r.png")
plot_qnm("ringdown_qnm.png")
plot_h22("Inspiral", inspiral_train, inspiral_predict, inspiral_linear, inspiral_quadratic, "inspiral_h22.png")
plot_error_vs_r("Inspiral", inspiral_linear, inspiral_quadratic, "inspiral_error_vs_r.png")
logln("\nFigures written to $OUTDIR")
close(LOG_FILE[])
end
