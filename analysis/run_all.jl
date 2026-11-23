# Runs the analyses from docs/project_plan.html on the Burgers' problem and
# writes figures and a text summary (results.txt) to analysis/output/.
#
#   cd analysis && julia run_all.jl          (≈ 1–2 min, mostly Plots compiling)
#
# Training window t ∈ [0, 2] (the same data as ../burgers/output/burgers_data.jld2); the FOM is
# also run to t = 4 so predictions beyond the training window can be scored.

include(joinpath(@__DIR__, "common.jl"))

const OUT = open(joinpath(FIGDIR, "results.txt"), "w")
say(args...) = (println(args...); println(OUT, args...); flush(OUT))

const RS  = 2:18                         # r + r(r+1)/2 ≤ 201 snapshots needs r ≤ 18
const λS  = 10.0 .^ range(-12, 2, length = 57)
const RREG = 15                          # r used for the regularization study (as in ../burgers/opInf.jl)

data = burgers_data()
(; xs, ts, U, dUdt, itr, dx, ν) = data
ipr  = (itr[end]+1):length(ts)
Utr  = U[:, itr]
dt   = ts[2] - ts[1]
n    = length(xs)
Uall = U
σU   = svd(Utr).S
say("Burgers' ν = $ν, N = $n, $(length(itr)) training snapshots (t ≤ $(data.t_train)), ",
    "$(length(ipr)) prediction snapshots (t ≤ $(ts[end]))\n")

# reduced quantities for one r
function reduced(r; Ut = Utr)
    V  = pod_basis(Ut, r)
    Qh = V' * Ut
    D  = data_matrix(Qh)
    Âg, Ĥg = galerkin_ops(V, ν, dx)
    R_exact  = Matrix((V' * dUdt[:, itr])')
    R_fd     = Matrix(fd_time_derivative(Qh, dt)')
    R_reproj = Matrix(reduce(hcat, [V' * burgers_rhs!(zeros(n), V * q, ν, dx) for q in eachcol(Qh)])')
    return (; r, V, Qh, D, Og = [Âg Ĥg], R_exact, R_fd, R_reproj)
end
red = Dict(r => reduced(r) for r in RS)

# ================================================================ Q1 recovery
say("== Q1  Operator recovery: ‖Ô − Ô_Galerkin‖/‖Ô_Galerkin‖ (QR, λ = 0) ==")
say(@sprintf("%4s %12s %12s %12s", "r", "re-projected", "exact dU/dt", "FD dQ̂/dt"))
rec = Dict(k => Float64[] for k in (:reproj, :exact, :fd))
for r in RS
    s = red[r]
    for (k, R) in ((:reproj, s.R_reproj), (:exact, s.R_exact), (:fd, s.R_fd))
        push!(rec[k], relerr(solve_qr(s.D, R), s.Og))
    end
    say(@sprintf("%4d %12.2e %12.2e %12.2e", r, rec[:reproj][end], rec[:exact][end], rec[:fd][end]))
end
plt = plot(RS, logsafe(rec[:reproj]), yscale = :log10, marker = :circle, label = "re-projected data",
           xlabel = "r", ylabel = "‖Ô − Ô_Gal‖ / ‖Ô_Gal‖", title = "Q1: does OpInf recover the intrusive operators?",
           legend = :right)
plot!(plt, RS, logsafe(rec[:exact]), marker = :utriangle, label = "projected exact dU/dt")
plot!(plt, RS, logsafe(rec[:fd]), marker = :square, label = "FD of snapshots (Δt = $dt)")
savefig_here(plt, "q1_operator_recovery.png")
say("""
Re-projected data satisfy the reduced Galerkin equations exactly, so OpInf
recovers Â, Ĥ up to rounding amplified by κ(D). Projected FOM derivatives
include the closure error Vᵀf(u) − Vᵀf(VVᵀu), so the operators differ from
Galerkin by an amount that shrinks only as the projection error shrinks.
""")

# ================================================================ Q2 conditioning
say("== Q2  Conditioning of D ==")
say(@sprintf("%4s %6s %12s %12s %12s %14s", "r", "cols", "κ(D)", "σ1/σr", "(σ1/σr)²", "κ(D col-scaled)"))
κD = Float64[]; κlin = Float64[]; κsc = Float64[]
for r in 1:last(RS)
    s  = r in RS ? red[r] : (; D = data_matrix(pod_basis(Utr, r)' * Utr))
    D  = s.D
    sv = svdvals(D)
    Dn = D ./ norm.(eachcol(D))'
    push!(κD, sv[1] / sv[end]); push!(κlin, σU[1] / σU[r]); push!(κsc, cond(Dn))
    say(@sprintf("%4d %6d %12.2e %12.2e %12.2e %14.2e", r, ncols(r), κD[end], κlin[end], κlin[end]^2, κsc[end]))
end
rr = 1:last(RS)
plt = plot(rr, κD, yscale = :log10, marker = :circle, label = "κ(D)", xlabel = "r", ylabel = "condition number",
           title = "Q2: κ(D) vs r", legend = :topleft)
plot!(plt, rr, κlin .^ 2, ls = :dash, label = "(σ₁/σᵣ)²  (snapshot singular values)")
plot!(plt, rr, κlin, ls = :dot, label = "σ₁/σᵣ = κ(linear block)")
plot!(plt, rr, κsc, marker = :diamond, label = "κ(D), columns scaled to unit norm")
hline!(plt, [1 / eps()], color = :gray, ls = :dashdot, label = "1/ε_mach")
savefig_here(plt, "q2_conditioning.png")
plt = plot(xlabel = "index i", ylabel = "σᵢ(D) / σ₁(D)", yscale = :log10, title = "Singular values of D",
           legend = :bottomleft)
for r in (6, 12, 18)
    sv = svdvals(red[r].D); plot!(plt, sv ./ sv[1], marker = :circle, ms = 2, label = "r = $r ($(ncols(r)) cols)")
end
savefig_here(plt, "q2_singular_values_D.png")
say("""
κ(D) grows much faster than (σ₁/σᵣ)², and scaling the columns to unit norm
only helps for small r. The rest is genuine near-dependence: with a single
trajectory every column of D is a smooth function of the one parameter t,
so the r + r(r+1)/2 columns are numerically dependent (like a Vandermonde
matrix). See Q6 for what more trajectories do.
""")

# ================================================================ Q3 solvers
say("== Q3  Solvers on a consistent problem with known answer (re-projected data, λ = 0) ==")
say(@sprintf("%4s %10s %12s %12s %12s %10s %10s %10s", "r", "κ(D)", "err NE", "err QR", "err SVD", "t NE", "t QR", "t SVD"))
err = Dict(k => Float64[] for k in (:ne, :qr, :svd)); tim = Dict(k => Float64[] for k in (:ne, :qr, :svd))
κ3 = Float64[]
for r in RS
    s = red[r]; push!(κ3, cond(s.D))
    for (k, f) in ((:ne, solve_normal), (:qr, solve_qr), (:svd, solve_svd))
        O = try f(s.D, s.R_reproj) catch; nothing end
        push!(err[k], O === nothing ? NaN : relerr(O, s.Og))
        push!(tim[k], O === nothing ? NaN : minimum(@elapsed(f(s.D, s.R_reproj)) for _ in 1:20))
    end
    say(@sprintf("%4d %10.2e %12.2e %12.2e %12.2e %10.1e %10.1e %10.1e", r, κ3[end],
                 err[:ne][end], err[:qr][end], err[:svd][end], tim[:ne][end], tim[:qr][end], tim[:svd][end]))
end
nfail = count(isnan, err[:ne])
nfail > 0 && say("normal equations: Cholesky failed (DᵀD not numerically positive definite) for $nfail value(s) of r")
plt = plot(κ3, logsafe(err[:ne]), xscale = :log10, yscale = :log10, seriestype = :scatter, label = "normal equations",
           xlabel = "κ(D)", ylabel = "relative operator error", title = "Q3: solver accuracy vs conditioning",
           legend = :topleft)
scatter!(plt, κ3, logsafe(err[:qr]), marker = :utriangle, label = "Householder QR")
scatter!(plt, κ3, logsafe(err[:svd]), marker = :diamond, label = "SVD")
κg = 10 .^ range(log10(minimum(κ3)), log10(maximum(κ3)), length = 50)
plot!(plt, κg, eps() .* κg, color = :gray, ls = :dash, label = "ε·κ")
plot!(plt, κg, min.(eps() .* κg .^ 2, 10.0), color = :black, ls = :dot, label = "ε·κ²")
savefig_here(plt, "q3_solver_accuracy.png")
say("""
Backward-stable QR and SVD give errors ~ ε·κ(D); forming DᵀD squares the
condition number, so the normal equations lose about twice as many digits
and break down entirely once κ(D) ≳ 1/√ε ≈ 7e7. All three are cheap here
(D is at most 201 × 189), so accuracy, not cost, decides.
""")

# ================================================================ Q4 regularization
say("== Q4  Choosing λ  (r = $RREG, projected exact derivatives) ==")
s4 = red[RREG]
S  = svd(s4.D)
K  = size(s4.D, 1)
ρ  = Float64[]; η = Float64[]; gcv = Float64[]; trainerr = Float64[]; stable = Bool[]; absc = Float64[]
fullerr_tr = Float64[]; fullerr_pr = Float64[]
bound = 1.5 * maximum(abs, s4.Qh)
for λ in λS
    O = solve_svd(s4.D, s4.R_exact, λ; S)
    f = S.S .^ 2 ./ (S.S .^ 2 .+ λ)
    resid = s4.D * O' - s4.R_exact
    push!(ρ, norm(resid)); push!(η, norm(O))
    push!(gcv, norm(resid)^2 / (K - sum(f))^2)
    Â, Ĥ = split_ops(O, RREG)
    push!(absc, maximum(real, eigvals(Â)))
    Q = integrate_rom(Â, Ĥ, s4.Qh[:, 1], ts; bound)
    ok = !any(isnan, Q)
    push!(stable, ok)
    push!(trainerr, ok ? norm(Q[:, itr] - s4.Qh) / norm(s4.Qh) : NaN)
    push!(fullerr_tr, rom_error(s4.V, Q, U, itr)); push!(fullerr_pr, rom_error(s4.V, Q, U, ipr))
end
# L-curve corner = point of maximum curvature of (log ρ, log η) as a function of log λ
lr, le, ll = log.(ρ), log.(η), log.(λS)
d1f(y) = [(y[min(i+1, end)] - y[max(i-1, 1)]) / (ll[min(i+1, end)] - ll[max(i-1, 1)]) for i in eachindex(y)]
r1, e1 = d1f(lr), d1f(le); r2, e2 = d1f(r1), d1f(e1)
curv = (r1 .* e2 .- r2 .* e1) ./ (r1 .^ 2 .+ e1 .^ 2) .^ 1.5
iL   = argmax(curv[2:end-1]) + 1
iG   = argmin(gcv)
iS   = argmin([st ? te : Inf for (st, te) in zip(stable, trainerr)])
say(@sprintf("%-22s %10s %12s %12s %12s", "method", "λ", "train err", "predict err", "max Re λ(Â)"))
for (name, i) in (("L-curve corner", iL), ("GCV", iG), ("stability-based", iS))
    say(@sprintf("%-22s %10.1e %12.2e %12.2e %12.2e", name, λS[i], fullerr_tr[i], fullerr_pr[i], absc[i]))
end
say(@sprintf("%-22s %10s %12.2e %12.2e", "POD projection floor", "—",
             norm(s4.V * (s4.V' * U[:, itr]) - U[:, itr]) / norm(U[:, itr]),
             norm(s4.V * (s4.V' * U[:, ipr]) - U[:, ipr]) / norm(U[:, ipr])))
Âg = s4.Og[:, 1:RREG]
say(@sprintf("Galerkin Â: max Re eigenvalue %.2e;  %d of %d λ values give a ROM that stays bounded to t = %g",
             maximum(real, eigvals(Âg)), count(stable), length(λS), ts[end]))

plt = plot(ρ, η, xscale = :log10, yscale = :log10, marker = :circle, ms = 2, label = "λ from 1e-12 to 1e2",
           xlabel = "residual ‖D Ôᵀ − Rᵀ‖", ylabel = "‖Ô‖", title = "Q4: L-curve (r = $RREG)", legend = :topright)
for (name, i, m) in (("L-curve corner", iL, :star5), ("GCV", iG, :diamond), ("stability", iS, :square))
    scatter!(plt, [ρ[i]], [η[i]], marker = m, ms = 8, label = @sprintf("%s  λ = %.0e", name, λS[i]))
end
savefig_here(plt, "q4_lcurve.png")
plt = plot(λS, logsafe(fullerr_tr), xscale = :log10, yscale = :log10, label = "ROM error, training window",
           xlabel = "λ", ylabel = "relative error", title = "Q4: ROM error vs λ (r = $RREG); gaps = unstable ROM",
           legend = :topleft)
plot!(plt, λS, logsafe(fullerr_pr), ls = :dash, label = "ROM error, prediction t ∈ (2, 4]")
plot!(plt, λS, gcv ./ maximum(gcv), ls = :dot, label = "GCV(λ), normalized")
vline!(plt, [λS[iL]], color = :gray, ls = :dash, label = "L-curve")
vline!(plt, [λS[iG]], color = :purple, ls = :dashdot, label = "GCV")
vline!(plt, [λS[iS]], color = :green, label = "stability")
savefig_here(plt, "q4_error_vs_lambda.png")
plt = plot(λS, absc, xscale = :log10, label = "OpInf", xlabel = "λ", ylabel = "max Re eigenvalue of Â",
           title = "Q4: stability of the learned linear operator (r = $RREG)", legend = :topright)
hline!(plt, [maximum(real, eigvals(Âg))], ls = :dash, label = "Galerkin")
hline!(plt, [0.0], color = :black, lw = 1, label = "")
savefig_here(plt, "q4_spectral_abscissa.png")
plt = scatter(real(eigvals(Âg)), imag(eigvals(Âg)), marker = :x, ms = 6, label = "Galerkin",
              xlabel = "Re", ylabel = "Im", title = "Eigenvalues of Â (r = $RREG)", legend = :bottomleft)
for (lab, i, m) in (("OpInf λ = $(@sprintf("%.0e", λS[1]))", 1, :circle), ("OpInf, stability λ", iS, :utriangle))
    ev = eigvals(split_ops(solve_svd(s4.D, s4.R_exact, λS[i]; S), RREG)[1])
    scatter!(plt, real(ev), imag(ev), marker = m, label = lab)
end
savefig_here(plt, "q4_eigenvalues.png")
iG == 1 && say("GCV is minimized at the smallest λ on the grid, i.e. it fails to find an interior minimum.")
say("""
GCV and the L-curve only look at the regression; the stability criterion
looks at the ROM it produces (using training data only), which is what
actually matters for prediction.
""")

# ================================================================ Q5 robustness + accuracy vs r
say("== Q5a  ROM accuracy vs r (no noise; λ by stability criterion) ==")
say(@sprintf("%4s %11s %11s %11s %11s %11s %11s %11s %11s", "r", "POD tr", "POD pr", "Gal tr", "Gal pr",
             "OpI-ex tr", "OpI-ex pr", "OpI-FD tr", "OpI-FD pr"))
acc = Dict(k => Float64[] for k in (:pod_tr, :pod_pr, :gal_tr, :gal_pr, :ex_tr, :ex_pr, :fd_tr, :fd_pr))
for r in RS
    s = red[r]
    push!(acc[:pod_tr], norm(s.V * (s.V' * U[:, itr]) - U[:, itr]) / norm(U[:, itr]))
    push!(acc[:pod_pr], norm(s.V * (s.V' * U[:, ipr]) - U[:, ipr]) / norm(U[:, ipr]))
    Qg = integrate_rom(split_ops(s.Og, r)..., s.Qh[:, 1], ts; bound = 1e3 * maximum(abs, s.Qh))
    push!(acc[:gal_tr], rom_error(s.V, Qg, U, itr)); push!(acc[:gal_pr], rom_error(s.V, Qg, U, ipr))
    for (k, R) in ((:ex, s.R_exact), (:fd, s.R_fd))
        λ, O, _ = select_lambda_stability(s.D, R, s.Qh, ts, itr, λS)
        Q = isnan(λ) ? fill(NaN, r, length(ts)) : integrate_rom(split_ops(O, r)..., s.Qh[:, 1], ts)
        push!(acc[Symbol(k, :_tr)], rom_error(s.V, Q, U, itr)); push!(acc[Symbol(k, :_pr)], rom_error(s.V, Q, U, ipr))
    end
    say(@sprintf("%4d", r), join(@sprintf(" %11.2e", acc[k][end])
        for k in (:pod_tr, :pod_pr, :gal_tr, :gal_pr, :ex_tr, :ex_pr, :fd_tr, :fd_pr)))
end
plt = plot(xlabel = "r", ylabel = "relative error", yscale = :log10, legend = :bottomleft,
           title = "Q5: ROM error vs r  (solid: t ≤ 2 training,  dashed: 2 < t ≤ 4 prediction)")
for (k, lab, c) in ((:pod, "POD projection", :gray), (:gal, "Galerkin", :black), (:ex, "OpInf, exact dU/dt", :blue),
                    (:fd, "OpInf, FD derivatives", :orange))
    plot!(plt, RS, logsafe(acc[Symbol(k, :_tr)]), color = c, marker = :circle, ms = 3, label = lab)
    plot!(plt, RS, logsafe(acc[Symbol(k, :_pr)]), color = c, ls = :dash, label = "")
end
savefig_here(plt, "q5_error_vs_r.png")

say("\n== Q5b  Noise robustness (Gaussian noise, std = η·max|u|; FD derivatives; λ by stability; median of 3 seeds) ==")
ηs = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
rnoise = (6, 10, 15)
say(@sprintf("%9s %4s %11s %11s %10s", "η", "r", "train err", "pred err", "λ chosen"))
noise = Dict((r, k) => Float64[] for r in rnoise, k in (:tr, :pr))
for η in ηs, r in rnoise
    tr = Float64[]; pr = Float64[]; lam = Float64[]
    for seed in 1:3
        Un = Utr .+ η * maximum(abs, Utr) .* randn(MersenneTwister(seed), size(Utr))
        Un[1, :] .= 0; Un[end, :] .= 0                       # boundary values are known exactly
        V  = pod_basis(Un, r); Qh = V' * Un
        D  = data_matrix(Qh); R = Matrix(fd_time_derivative(Qh, dt)')
        λ, O, _ = select_lambda_stability(D, R, Qh, ts, itr, λS)
        Q = isnan(λ) ? fill(NaN, r, length(ts)) : integrate_rom(split_ops(O, r)..., Qh[:, 1], ts)
        push!(tr, rom_error(V, Q, U, itr)); push!(pr, rom_error(V, Q, U, ipr)); push!(lam, λ)
    end
    med(v) = sort(v)[2]
    push!(noise[(r, :tr)], med(tr)); push!(noise[(r, :pr)], med(pr))
    say(@sprintf("%9.0e %4d %11.2e %11.2e %10.0e", η, r, med(tr), med(pr), med(lam)))
end
ηplot = [max(η, 1e-6) for η in ηs]
plt = plot(xlabel = "noise level η  (η = 0 plotted at 1e-6)", ylabel = "relative error vs clean FOM",
           xscale = :log10, yscale = :log10, legend = :topleft,
           title = "Q5: noise robustness (solid: training,  dashed: prediction)")
for (r, c) in zip(rnoise, (:blue, :orange, :green))
    plot!(plt, ηplot, logsafe(noise[(r, :tr)]), color = c, marker = :circle, label = "r = $r")
    plot!(plt, ηplot, logsafe(noise[(r, :pr)]), color = c, ls = :dash, marker = :utriangle, label = "")
end
plot!(plt, ηplot, ηplot, color = :gray, ls = :dot, label = "error = η")
savefig_here(plt, "q5_noise.png")

# ================================================================ Q6 more trajectories
say("\n== Q6  Remedy: train on several trajectories  u0 = sin(πx) + c·sin(2πx), c ∈ {0, 0.5, −0.5, 0.25} ==")
cs = (0.0, 0.5, -0.5, 0.25)
trajs = [solve_burgers(merge(default_params, (t_max = data.t_train, u0 = x -> sin(π * x) + c * sin(2π * x),
                                              make_plot = false, make_gif = false))) for c in cs]
Ucat  = reduce(hcat, [t[3] for t in trajs]); dUcat = reduce(hcat, [t[4] for t in trajs])
@assert trajs[1][3] ≈ Utr
say(@sprintf("%4s %11s %11s | %11s %11s | %11s %11s | %11s %11s", "r", "κ(D) 1tr", "κ(D) 4tr",
             "rec-rp 1tr", "rec-rp 4tr", "rec-ex 1tr", "rec-ex 4tr", "pred 1tr", "pred 4tr"))
m6 = Dict(k => Float64[] for k in (:κ1, :κ4, :rp1, :rp4, :ex1, :ex4, :tr1, :tr4, :pr1, :pr4))
for r in RS
    s1 = red[r]
    V  = pod_basis(Ucat, r)
    Qs = [V' * t[3] for t in trajs]
    D  = reduce(vcat, [data_matrix(Q) for Q in Qs])
    Rex = Matrix((V' * dUcat)')
    Rrp = Matrix(reduce(hcat, [V' * burgers_rhs!(zeros(n), V * q, ν, dx) for Q in Qs for q in eachcol(Q)])')
    Og  = hcat(galerkin_ops(V, ν, dx)...)
    push!(m6[:κ1], cond(s1.D)); push!(m6[:κ4], cond(D))
    push!(m6[:rp1], relerr(solve_qr(s1.D, s1.R_reproj), s1.Og)); push!(m6[:rp4], relerr(solve_qr(D, Rrp), Og))
    push!(m6[:ex1], relerr(solve_qr(s1.D, s1.R_exact), s1.Og));  push!(m6[:ex4], relerr(solve_qr(D, Rex), Og))
    for (k, DD, RR, VV, Q1) in ((1, s1.D, s1.R_exact, s1.V, s1.Qh), (4, D, Rex, V, Qs[1]))
        λ, O, _ = select_lambda_stability(DD, RR, Q1, ts, itr, λS)
        Q = isnan(λ) ? fill(NaN, r, length(ts)) : integrate_rom(split_ops(O, r)..., Q1[:, 1], ts)
        push!(m6[Symbol(:tr, k)], rom_error(VV, Q, U, itr)); push!(m6[Symbol(:pr, k)], rom_error(VV, Q, U, ipr))
    end
    say(@sprintf("%4d %11.2e %11.2e | %11.2e %11.2e | %11.2e %11.2e | %11.2e %11.2e", r, m6[:κ1][end], m6[:κ4][end],
                 m6[:rp1][end], m6[:rp4][end], m6[:ex1][end], m6[:ex4][end], m6[:pr1][end], m6[:pr4][end]))
end
plt = plot(RS, m6[:κ1], yscale = :log10, marker = :circle, label = "1 trajectory", xlabel = "r", ylabel = "κ(D)",
           title = "Q6: conditioning with more training trajectories", legend = :topleft)
plot!(plt, RS, m6[:κ4], marker = :utriangle, label = "4 trajectories")
hline!(plt, [1 / eps()], color = :gray, ls = :dashdot, label = "1/ε_mach")
savefig_here(plt, "q6_conditioning_trajectories.png")
plt = plot(RS, logsafe(m6[:rp1]), yscale = :log10, marker = :circle, color = :blue, label = "1 traj, re-projected",
           xlabel = "r", ylabel = "‖Ô − Ô_Gal‖ / ‖Ô_Gal‖", title = "Q6: operator recovery with more trajectories",
           legend = :topleft)
plot!(plt, RS, logsafe(m6[:rp4]), marker = :utriangle, color = :blue, ls = :dash, label = "4 traj, re-projected")
plot!(plt, RS, logsafe(m6[:ex1]), marker = :circle, color = :orange, label = "1 traj, exact dU/dt")
plot!(plt, RS, logsafe(m6[:ex4]), marker = :utriangle, color = :orange, ls = :dash, label = "4 traj, exact dU/dt")
savefig_here(plt, "q6_recovery_trajectories.png")
plt = plot(RS, logsafe(acc[:gal_pr]), yscale = :log10, color = :black, label = "Galerkin", xlabel = "r",
           ylabel = "prediction error, 2 < t ≤ 4", title = "Q6: prediction error of the c = 0 trajectory",
           legend = :bottomleft)
plot!(plt, RS, logsafe(m6[:pr1]), marker = :circle, color = :blue, label = "OpInf, 1 trajectory")
plot!(plt, RS, logsafe(m6[:pr4]), marker = :utriangle, color = :green, label = "OpInf, 4 trajectories")
savefig_here(plt, "q6_prediction_trajectories.png")

close(OUT)
println("\nsummary → analysis/output/results.txt")
