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
# Run:  julia einstein/opinf_sxs.jl

using JLD2, LinearAlgebra, Printf, Plots

const HERE = @__DIR__
const OUTDIR = mkpath(joinpath(HERE, "output"))
const LOG = Ref{IO}(devnull)
logln(s...) = (println(s...); println(LOG[], s...); flush(LOG[]))

# ---------------------------------------------------------------- data
# drop_m0: leave out the m = 0 modes. After merger they are dominated by the gravitational-wave
# memory, a non-oscillating offset (|h20| ≈ 0.08) that swamps the decaying ringdown in any norm,
# and they are the least reliable modes of extrapolated SXS waveforms.
function load_sxs(path = joinpath(HERE, "data", "sxs_bbh_0305.h5"); drop_m0 = true)
    f = jldopen(path, "r")
    t = vec(f["t"])
    # h5py writes row-major (n_t, n_modes); Julia reads it transposed as (n_modes, n_t)
    ell, m = vec(f["ell"]), vec(f["m"])
    keep = drop_m0 ? findall(m .!= 0) : collect(eachindex(m))
    X = vcat(f["h_re"][keep, :], f["h_im"][keep, :])
    Xd = vcat(f["hdot_re"][keep, :], f["hdot_im"][keep, :])
    close(f)
    return t, X, Xd, ell[keep], m[keep]
end

# ---------------------------------------------------------------- OpInf pieces
compact_kron(x) = [x[i] * x[j] for i in eachindex(x) for j in 1:i]

function data_matrix(Qh, quad)
    quad || return Matrix(Qh')
    return hcat(Qh', reduce(hcat, compact_kron.(eachcol(Qh)))')
end

# Tikhonov-regularized least squares via Householder QR of [D; √λ I]
function solve_ls(D, R, λ)
    n = size(D, 2)
    A = vcat(D, sqrt(λ) * Matrix(I, n, n))
    B = vcat(R, zeros(n, size(R, 2)))
    F = qr(A)
    return Matrix((UpperTriangular(F.R) \ (F.Q' * B)[1:n, :])')   # O = [Â Ĥ]
end

function rom_rhs(O, r, quad)
    Â = O[:, 1:r]
    quad || return x -> Â * x
    Ĥ = O[:, r+1:end]
    return x -> Â * x + Ĥ * compact_kron(x)
end

# RK4 on the saved times with `sub` substeps; stops (NaN) if the state leaves a bound
function integrate(f, x0, ts; sub = 5, bound = Inf)
    Xs = fill(NaN, length(x0), length(ts))
    x = copy(x0); Xs[:, 1] = x
    for k in 2:length(ts)
        h = (ts[k] - ts[k-1]) / sub
        for _ in 1:sub
            k1 = f(x); k2 = f(x + h/2*k1); k3 = f(x + h/2*k2); k4 = f(x + h*k3)
            x = x + h/6*(k1 + 2k2 + 2k3 + k4)
        end
        (all(isfinite, x) && norm(x) < bound) || return Xs
        Xs[:, k] = x
    end
    return Xs
end

relerr(A, B) = any(isnan, A) ? Inf : norm(A - B) / norm(B)

# Fit at a given λ on training indices, return operators
fit(V, X, Xd, idx, r, quad, λ) = solve_ls(data_matrix(V' * X[:, idx], quad), (V' * Xd[:, idx])', λ)

# Choose λ using the training window only: fit on the first 75 %, forecast the last 25 %.
function choose_lambda(V, X, Xd, t, itr, r, quad, λs)
    ncut = round(Int, 0.75 * length(itr))
    ifit, ival = itr[1:ncut], itr[ncut:end]
    bound = 10 * maximum(norm.(eachcol(V' * X[:, itr])))
    best = (Inf, λs[1])
    for λ in λs
        O = fit(V, X, Xd, ifit, r, quad, λ)
        Xr = integrate(rom_rhs(O, r, quad), V' * X[:, ifit[1]], t[ifit[1]:ival[end]]; bound)
        e = relerr(V * Xr[:, ncut:end], X[:, ival])
        e < best[1] && (best = (e, λ))
    end
    return best[2]
end

# ---------------------------------------------------------------- one experiment
function run_experiment(name, t, X, Xd, itr, ipr, rs, quad; λs = [0.0; 10.0 .^ (-12:1:-2)])
    model = quad ? "quadratic" : "linear"
    logln("\n== $name, $model model ==")
    logln(@sprintf("train t ∈ [%.1f, %.1f] (%d snapshots), predict t ∈ (%.1f, %.1f]",
                   t[itr[1]], t[itr[end]], length(itr), t[ipr[1]], t[ipr[end]]))
    logln(@sprintf("%4s %5s %10s %10s %11s %11s %11s %12s", "r", "cols", "σ_r/σ_1",
                   "λ", "κ(D)", "train err", "pred err", "POD floor pr"))
    U, S, _ = svd(X[:, itr])
    iall = itr[1]:ipr[end]
    results = []
    for r in rs
        p = quad ? r + r * (r + 1) ÷ 2 : r
        p < length(itr) || continue
        V = U[:, 1:r]
        λ = choose_lambda(V, X, Xd, t, itr, r, quad, λs)
        O = fit(V, X, Xd, itr, r, quad, λ)
        κ = cond(data_matrix(V' * X[:, itr], quad))
        bound = 10 * maximum(norm.(eachcol(X[:, iall])))
        Xr = V * integrate(rom_rhs(O, r, quad), V' * X[:, itr[1]], t[iall]; bound)
        ntr = length(itr)
        etr = relerr(Xr[:, 1:ntr], X[:, itr])
        epr = relerr(Xr[:, ntr:end], X[:, itr[end]:ipr[end]])
        floor_pr = norm(X[:, ipr] - V * (V' * X[:, ipr])) / norm(X[:, ipr])
        logln(@sprintf("%4d %5d %10.2e %10.1e %11.2e %11.2e %11.2e %12.2e",
                       r, p, S[r] / S[1], λ, κ, etr, epr, floor_pr))
        push!(results, (; r, λ, κ, etr, epr, floor_pr, Xr, O, V))
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

# eigenvalue of the learned Â closest to the Kerr QNM (upper half plane)
function closest_eig(O, r, target)
    ev = eigvals(O[:, 1:r])
    return ev[argmin(abs.(ev .- target))]
end

relerr_t(A, B) = [norm(A[:, k] - B[:, k]) / norm(B[:, k]) for k in axes(B, 2)]

# ---------------------------------------------------------------- main
if abspath(PROGRAM_FILE) == @__FILE__
LOG[] = open(joinpath(OUTDIR, "sxs_log.txt"), "w")
t, X, Xd, ell, m = load_sxs()
i22 = findfirst((ell .== 2) .& (m .== 2))
nm = length(ell)
amp22 = hypot.(X[i22, :], X[nm+i22, :])
tpk = t[argmax(amp22)]
window(a, b) = findall(a .<= t .<= b)

logln("SXS:BBH:0305 Lev6 strain (doi:10.26138/SXS:BBH:0305), Δt = $(t[2]-t[1]) M, ",
      "peak of |h22| at t_peak = $(round(tpk, digits=1)) M")
logln("State: the $(nm) modes with m ≠ 0 (ℓ = 2..8), as [Re; Im] ∈ R^$(2nm). ",
      "Errors are relative 2-norm errors over the sphere.")
logln("λ is chosen from training data only (fit on the first 75 % of the window, forecast the last 25 %).")

# memory (m = 0) modes, reported but not modelled
_, X0, _, ell0, m0 = load_sxs(drop_m0 = false)
i20 = findfirst((ell0 .== 2) .& (m0 .== 0))
logln(@sprintf("Memory: h20 = %.3f at t = 3000, %.3f at t_peak, %.3f at t_peak + 50 (vs |h22| = %.1e there).",
               X0[i20, argmin(abs.(t .- 3000))], X0[i20, argmin(abs.(t .- tpk))],
               X0[i20, argmin(abs.(t .- tpk .- 50))], amp22[argmin(abs.(t .- tpk .- 50))]))
logln("  This non-oscillating offset would dominate every norm after merger, so m = 0 is left out.")

qnm = kerr_qnm_220()
logln(@sprintf("Kerr QNM (2,2,0) for the remnant: %.4f %+.4fi", real(qnm), imag(qnm)))

# ---- 1. Ringdown: train 20–50 M after the peak, predict to 80 M after the peak
rd_tr, rd_pr = window(tpk + 20, tpk + 50), window(tpk + 50.5, tpk + 80)
rd_lin  = run_experiment("Ringdown", t, X, Xd, rd_tr, rd_pr, 2:2:20, false)
rd_quad = run_experiment("Ringdown", t, X, Xd, rd_tr, rd_pr, 2:2:8, true)

logln("\nLearned Â eigenvalue closest to the Kerr QNM (linear model, ringdown window above):")
logln(@sprintf("%4s %22s %12s", "r", "eigenvalue", "rel. diff"))
for x in rd_lin
    e = closest_eig(x.O, x.r, qnm)
    logln(@sprintf("%4d %10.4f %+.4fi %12.2e", x.r, real(e), imag(e), abs(e - qnm) / abs(qnm)))
end

# ---- 2. When does the post-merger signal become linear? Slide the 30 M training window.
logln("\n== Ringdown start-time sweep: linear model, r = 2 and 16, 30 M training window ==")
logln(@sprintf("%12s %22s %10s %22s %10s", "start − t_peak", "eig (r = 2)", "rel diff", "eig (r = 16)", "rel diff"))
starts = [0, 5, 10, 15, 20, 25, 30]
sweep = map(starts) do s0
    itr = window(tpk + s0, tpk + s0 + 30)
    U = svd(X[:, itr]).U
    row = map((2, 16)) do r
        V = U[:, 1:r]
        λ = choose_lambda(V, X, Xd, t, itr, r, false, [0.0; 10.0 .^ (-12:1:-2)])
        closest_eig(fit(V, X, Xd, itr, r, false, λ), r, qnm)
    end
    logln(@sprintf("%12d %10.4f %+.4fi %10.2e %10.4f %+.4fi %10.2e", s0,
                   real(row[1]), imag(row[1]), abs(row[1] - qnm) / abs(qnm),
                   real(row[2]), imag(row[2]), abs(row[2] - qnm) / abs(qnm)))
    row
end

# ---- 3. Inspiral: train on [3300, 3500] M, predict [3500, 3600] M (merger is at ≈ 3692 M)
in_tr, in_pr = window(3300, 3500), window(3500.5, 3600)
in_lin  = run_experiment("Inspiral", t, X, Xd, in_tr, in_pr, 2:2:12, false)
in_quad = run_experiment("Inspiral", t, X, Xd, in_tr, in_pr, 2:2:10, true)

# ---------------------------------------------------------------- figures
best(res) = res[argmin([x.epr for x in res])]

function plot_h22(name, itr, ipr, res_lin, res_quad, fname; logamp = false)
    iall = itr[1]:ipr[end]
    bl, bq = best(res_lin), best(res_quad)
    p1 = plot(t[iall], X[i22, iall], lw = 3, c = :black, label = "true (SXS)",
              ylabel = "Re h₂₂", title = "$name: h₂₂ mode")
    plot!(p1, t[iall], bl.Xr[i22, :], lw = 1.5, ls = :dash, c = :blue, label = "OpInf linear, r = $(bl.r)")
    plot!(p1, t[iall], bq.Xr[i22, :], lw = 1.5, ls = :dot, c = :red, label = "OpInf quadratic, r = $(bq.r)")
    vline!(p1, [t[itr[end]]], c = :gray, ls = :dash, label = "end of training")
    p2 = plot(t[iall], relerr_t(bl.Xr, X[:, iall]), yscale = :log10, c = :blue, lw = 1.5,
              label = "linear", xlabel = "t / M", ylabel = "‖x_ROM − x‖ / ‖x‖")
    e = relerr_t(bq.Xr, X[:, iall]); e[.!isfinite.(e)] .= NaN
    plot!(p2, t[iall], e, c = :red, lw = 1.5, label = "quadratic")
    vline!(p2, [t[itr[end]]], c = :gray, ls = :dash, label = "")
    savefig(plot(p1, p2, layout = (2, 1), size = (800, 650)), joinpath(OUTDIR, fname))
end

function plot_err(name, rl, rq, fname)
    clip(v) = min.(v, 10.0)
    plt = plot(yscale = :log10, xlabel = "r", ylabel = "relative error (prediction window)",
               title = "$name: prediction error vs r", legend = :bottomleft)
    plot!(plt, [x.r for x in rl], clip([x.epr for x in rl]), m = :o, label = "OpInf linear")
    plot!(plt, [x.r for x in rq], clip([x.epr for x in rq]), m = :s, label = "OpInf quadratic")
    plot!(plt, [x.r for x in rl], [x.floor_pr for x in rl], ls = :dash, c = :black,
          label = "POD floor (training basis)")
    savefig(plt, joinpath(OUTDIR, fname))
end

function plot_qnm(fname)
    x = rd_lin[findfirst(z -> z.r == 16, rd_lin)]
    ev = eigvals(x.O[:, 1:x.r])
    plt = scatter(real.(ev), imag.(ev), label = "eigenvalues of learned Â (r = 16)",
                  xlabel = "Re (decay rate)", ylabel = "Im (frequency)",
                  title = "Ringdown: learned spectrum vs Kerr QNM")
    scatter!(plt, [real(qnm), real(qnm)], [imag(qnm), -imag(qnm)], m = :star5, ms = 10,
             c = :red, label = "Kerr (2,2,0) QNM, Berti et al. fit")
    p2 = plot(starts, [abs(r[1] - qnm) / abs(qnm) for r in sweep], m = :o, yscale = :log10,
              label = "r = 2", xlabel = "training start − t_peak (M)",
              ylabel = "|λ − ω_QNM| / |ω_QNM|", title = "Learned frequency vs training start")
    plot!(p2, starts, [abs(r[2] - qnm) / abs(qnm) for r in sweep], m = :s, label = "r = 16")
    savefig(plot(plt, p2, layout = (1, 2), size = (1100, 450), margin = 5Plots.mm), joinpath(OUTDIR, fname))
end

plot_h22("Ringdown", rd_tr, rd_pr, rd_lin, rd_quad, "ringdown_h22.png")
plot_err("Ringdown", rd_lin, rd_quad, "ringdown_error_vs_r.png")
plot_qnm("ringdown_qnm.png")
plot_h22("Inspiral", in_tr, in_pr, in_lin, in_quad, "inspiral_h22.png")
plot_err("Inspiral", in_lin, in_quad, "inspiral_error_vs_r.png")
logln("\nFigures written to $OUTDIR")
close(LOG[])
end
