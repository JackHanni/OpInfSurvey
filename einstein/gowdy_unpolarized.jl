# Part B: unpolarized Gowdy T³ — own solver (validated) + OpInf with and without lifting.
#
# Berger & Moncrief 1993 (PRD 48, 4676), eqs. 3.2-3.5, time τ (singularity at τ → ∞):
#   P_ττ - e^{-2τ} P_θθ - e^{2P}(Q_τ² - e^{-2τ} Q_θ²) = 0
#   Q_ττ - e^{-2τ} Q_θθ + 2(P_τ Q_τ - e^{-2τ} P_θ Q_θ) = 0
#   momentum constraint  λ_θ = 2(P_θ P_τ + e^{2P} Q_θ Q_τ)  ⇒  ∮ (P_θ P_τ + e^{2P} Q_θ Q_τ) dθ = 0
#
# Full-order model (FOM): state [P; Π; Q; Ξ], Π = P_τ, Ξ = Q_τ, 4th-order periodic FD in θ,
# RK4 in τ.  The e^{2P} factor makes it NON-polynomial.
#
# Lifting (derived here; verified numerically in `lift_check`):
#   G = e^{-τ} P_θ,   R = e^{P} Q_τ,   S = e^{P-τ} Q_θ
# gives the closed system, with input v(τ) = e^{-τ}:
#   Π_τ = v G_θ + R² - S²
#   G_τ = -G + v Π_θ
#   R_τ = -Π R + v S_θ + G S
#   S_τ = -S + Π S + v R_θ - G R
# i.e.  dy/dτ = A y + v(τ) N y + H(y ⊗ y)  — exactly quadratic, bilinear in the input.
# P is passive (P_τ = Π) and is recovered by quadrature.  Π² + G² + R² + S² is the
# Hamiltonian density of eq. 3.8, and the momentum constraint becomes ∮ (G Π + S R) dθ = 0.
#
# Run:  julia gowdy_unpolarized.jl   (a few minutes)

include(joinpath(@__DIR__, "gowdy_common.jl"))
using SpecialFunctions

# ------------------------------------------------------------------ parameters
const unp_params = (
    N        = 256,         # grid points in θ
    v0       = 2.0,         # Berger-Moncrief initial data: P = 0, P_τ = v0 cos θ, Q = cos θ, Q_τ = 0
    τ_train  = 2.0,         # end of training window
    τ_end    = 4.0,         # end of prediction window
    save_dτ  = 0.001,
    cfl      = 0.25,        # dτ ≤ cfl·dθ  (wave speed e^{-τ} ≤ 1)
    rom_sub  = 2,           # RK4 substeps per saved interval for ROMs
    ks       = [2, 4, 6, 8, 10],
    λs       = [1e-8, 1e-6, 1e-4, 1e-2],
    conv_Ns  = [64, 128, 256, 512],
    save_every = 5,         # snapshot subsampling for the saved .jld2 (keeps the file small)
    outdir   = mkpath(joinpath(@__DIR__, "output")),
    name     = "unpolarized",   # output file prefix
)

# harder case: larger initial velocity, sharper features (onset of Berger-Moncrief "spikes"),
# longer extrapolation.  N doubled because the lifting residual at N = 256 is 7e-4 here.
const unp_hard_params = merge(unp_params, (v0 = 3.0, N = 512, τ_train = 2.5, τ_end = 5.0,
                                           name = "unpolarized_v0_3"))

# ------------------------------------------------------------------ full-order model
function fom_rhs!(dx, x, τ, D1, D2)
    N = length(x) ÷ 4
    P, Π, Q, Ξ = (view(x, (b-1)*N+1:b*N) for b in 1:4)
    w = exp(-2τ)
    Pθ, Qθ = D1 * P, D1 * Q
    @views begin
        dx[1:N]       .= Π
        dx[N+1:2N]    .= w .* (D2 * P) .+ exp.(2 .* P) .* (Ξ .^ 2 .- w .* Qθ .^ 2)
        dx[2N+1:3N]   .= Ξ
        dx[3N+1:4N]   .= w .* (D2 * Q) .- 2 .* (Π .* Ξ .- w .* Pθ .* Qθ)
    end
    return dx
end

function solve_fom(x0, N, τs, cfl)
    D1, D2 = d1_matrix(N), d2_matrix(N)
    sub = ceil(Int, (τs[2] - τs[1]) / (cfl * 2π / N))
    X = integrate((dx, x, τ) -> fom_rhs!(dx, x, τ, D1, D2), x0, τs, s -> (s, 1.0), sub)
    Xd = similar(X)
    for j in axes(X, 2)
        fom_rhs!(view(Xd, :, j), X[:, j], τs[j], D1, D2)
    end
    return X, Xd
end

bm_initial(θs, v0) = [zeros(length(θs)); v0 .* cos.(θs); cos.(θs); zeros(length(θs))]

"Exact pseudo-unpolarized solution (B&M eqs. 3.18-3.20): φ = Y₀(e^{-τ}) cos θ is a polarized
solution, and P = ln cosh φ, Q = tanh φ solves the full nonlinear system."
function pseudo_exact(θs, τ)
    t = exp(-τ)
    φ  = bessely0(t) .* cos.(θs)
    φτ = t * bessely1(t) .* cos.(θs)          # dφ/dτ = -t dφ/dt,  Y₀' = -Y₁
    [log.(cosh.(φ)); tanh.(φ) .* φτ; tanh.(φ); sech.(φ) .^ 2 .* φτ]
end

"∮ (P_θ P_τ + e^{2P} Q_θ Q_τ) dθ, which must vanish on T³ (B&M eq. 3.4)."
function momentum_constraint(x, D1)
    N = length(x) ÷ 4
    P, Π, Q, Ξ = (x[(b-1)*N+1:b*N] for b in 1:4)
    (2π / N) * sum((D1 * P) .* Π .+ exp.(2 .* P) .* (D1 * Q) .* Ξ)
end

# ------------------------------------------------------------------ lifting
"Lifted state [Π; G; R; S] and its exact τ-derivative (chain rule on FOM data)."
function lift(X, Xd, τs, D1)
    N = size(X, 1) ÷ 4
    Y, Yd = zeros(4N, size(X, 2)), zeros(4N, size(X, 2))
    for j in axes(X, 2)
        P, Π, Q, Ξ = (X[(b-1)*N+1:b*N, j] for b in 1:4)
        Pd, Πd, Qd, Ξd = (Xd[(b-1)*N+1:b*N, j] for b in 1:4)
        v, eP = exp(-τs[j]), exp.(P)
        G, R, S = v .* (D1 * P), eP .* Ξ, v .* eP .* (D1 * Q)
        Y[:, j]  .= [Π; G; R; S]
        Yd[:, j] .= [Πd; -G .+ v .* (D1 * Pd); Pd .* R .+ eP .* Ξd; (Pd .- 1) .* S .+ v .* eP .* (D1 * Qd)]
    end
    return Y, Yd
end

"Full-order lifted operators: dy/dτ = A y + v N y + B(y, y)."
function lifted_operators(N)
    D1 = d1_matrix(N); Z = spzeros(N, N); Id = sparse(1.0I, N, N)
    A  = [Z Z Z Z; Z -Id Z Z; Z Z Z Z; Z Z Z -Id]
    Nv = [Z D1 Z Z; D1 Z Z Z; Z Z Z D1; Z Z D1 Z]
    return A, Nv
end
"Symmetric-in-use bilinear form with B(y, y) = quadratic part of the lifted RHS."
function lifted_B(y, z)
    N = length(y) ÷ 4
    Πy, Gy, Ry, Sy = (y[(b-1)*N+1:b*N] for b in 1:4)
    Πz, Gz, Rz, Sz = (z[(b-1)*N+1:b*N] for b in 1:4)
    [Ry .* Rz .- Sy .* Sz; zeros(N); -Πy .* Rz .+ Gy .* Sz; Πy .* Sz .- Gy .* Rz]
end

"Residual of the lifted quadratic system on lifted FOM data (should be O(h⁴), not O(1))."
function lift_check(Y, Yd, τs)
    N = size(Y, 1) ÷ 4
    A, Nv = lifted_operators(N)
    F = reduce(hcat, [A * Y[:, j] + exp(-τs[j]) * (Nv * Y[:, j]) + lifted_B(Y[:, j], Y[:, j]) for j in axes(Y, 2)])
    relerr(F, Yd)
end

# ------------------------------------------------------------------ validation
function validate(p)
    println("== solver validation ==")
    τs = collect(0:p.save_dτ*10:p.τ_end)
    # (1) exact pseudo-unpolarized solution, convergence in N
    errs = Float64[]
    for N in p.conv_Ns
        θs = theta_grid(N)
        X, _ = solve_fom(pseudo_exact(θs, 0.0), N, τs, p.cfl)
        push!(errs, relerr(X[:, end], pseudo_exact(θs, τs[end])))
    end
    println("exact pseudo-unpolarized solution, rel. error at τ = $(p.τ_end):")
    for (i, N) in enumerate(p.conv_Ns)
        @printf "  N = %4d  err = %.3e  %s\n" N errs[i] (i > 1 ? @sprintf("order %.2f", log2(errs[i-1] / errs[i])) : "")
    end
    # (2) self-convergence for the Berger-Moncrief data (compare P on coarse grid points)
    sols = Dict{Int, Vector{Float64}}()
    for N in p.conv_Ns
        X, _ = solve_fom(bm_initial(theta_grid(N), p.v0), N, τs, p.cfl)
        sols[N] = X[:, end]
    end
    Nc = p.conv_Ns[1]
    sub(x, N) = (s = N ÷ Nc; reduce(vcat, [x[(b-1)*N+1:s:b*N] for b in 1:4]))
    diffs = [norm(sub(sols[p.conv_Ns[i]], p.conv_Ns[i]) - sub(sols[p.conv_Ns[i+1]], p.conv_Ns[i+1])) /
             norm(sub(sols[p.conv_Ns[end]], p.conv_Ns[end])) for i in 1:length(p.conv_Ns)-1]
    println("B&M data (v0 = $(p.v0)) self-convergence, ‖x_N - x_2N‖/‖x‖ at τ = $(p.τ_end):")
    for i in eachindex(diffs)
        @printf "  N = %4d vs %4d: %.3e  %s\n" p.conv_Ns[i] p.conv_Ns[i+1] diffs[i] (i > 1 ? @sprintf("order %.2f", log2(diffs[i-1] / diffs[i])) : "")
    end
    return (; errs, diffs)
end

# ------------------------------------------------------------------ OpInf experiment
"Scale each of the nb variable blocks by its max |value| on the training set."
function block_scales(X, nb, itr)
    N = size(X, 1) ÷ nb
    [maximum(abs, X[(b-1)*N+1:b*N, itr]) for b in 1:nb]
end
scale_vec(s, N) = reduce(vcat, [fill(si, N) for si in s])

function run_unpolarized(p = unp_params)
    val = validate(p)
    N = p.N; θs = theta_grid(N); D1 = d1_matrix(N)
    τs = collect(0:p.save_dτ:p.τ_end)
    t_fom = @elapsed X, Xd = solve_fom(bm_initial(θs, p.v0), N, τs, p.cfl)
    mc = [momentum_constraint(X[:, j], D1) for j in axes(X, 2)]
    @printf "FOM: N = %d, %d snapshots, %.1f s;  max |∮(P_θP_τ + e^{2P}Q_θQ_τ)dθ| = %.2e\n" N length(τs) t_fom maximum(abs, mc)
    Y, Yd = lift(X, Xd, τs, D1)
    lres = lift_check(Y, Yd, τs)
    @printf "lifted quadratic system residual on FOM data: %.2e  (discrete commutator error only)\n" lres
    js = 1:p.save_every:length(τs)      # Xd, Y, Yd are recomputable from X with fom_rhs!/lift
    jldsave(joinpath(p.outdir, "gowdy_$(p.name)_data.jld2"); θs, τs = τs[js], X = X[:, js], v0 = p.v0,
            N, τ_train = p.τ_train, validation = val, lift_residual = lres)

    itr = findall(τ -> τ <= p.τ_train + 1e-12, τs)
    ipr = findall(τ -> τ >  p.τ_train + 1e-12, τs)
    vfun(τ) = exp(-τ)
    u = vfun.(τs[itr])
    P = X[1:N, :]

    # quadrature P(τ) = P(0) + ∫ Π dτ (trapezoid on saved snapshots) for lifted ROMs
    function recover_P(Π)
        Pr = similar(Π); Pr[:, 1] .= P[:, 1]
        for j in 2:size(Π, 2)
            Pr[:, j] .= Pr[:, j-1] .+ p.save_dτ / 2 .* (Π[:, j-1] .+ Π[:, j])
        end
        return Pr
    end
    floor_P = relerr(recover_P(Y[1:N, :])[:, ipr], P[:, ipr])
    @printf "P recovered from exact Π by trapezoid quadrature: rel. err (predict window) %.1e\n" floor_P

    rows = []; store = Dict(); opdiff = []
    errs(Zr, Z, idx) = isnan(Zr[1, idx[end]]) ? NaN : relerr(Zr[:, idx], Z[:, idx])
    for (label, Z, Zd, nb) in (("lifted", Y, Yd, 4), ("naive", X, Xd, 4))
        s  = scale_vec(block_scales(Z, nb, itr), N)
        Zs, Zds = Z ./ s, Zd ./ s
        _, sv = block_pod(Zs[:, itr], nb, 1)
        store[(label, :svals)] = sv
        for k in p.ks
            V, _ = block_pod(Zs[:, itr], nb, k)
            Zh, Zhd = V' * Zs[:, itr], V' * Zds[:, itr]
            proj = s .* (V * (V' * Zs))
            push!(rows, (label, k, "POD floor", NaN, errs(proj, Z, itr), errs(proj, Z, ipr), NaN))
            models = Any[]
            if label == "lifted"     # intrusive Galerkin on the scaled lifted system
                Vu = s .* V          # columns in unscaled coordinates; test with V'/s
                Wt = (V ./ s)'
                A, Nv = lifted_operators(N)
                H = zeros(size(V, 2), size(V, 2) * (size(V, 2) + 1) ÷ 2); c = 0
                for i in 1:size(V, 2), j in 1:i
                    c += 1
                    H[:, c] .= Wt * (i == j ? lifted_B(Vu[:, i], Vu[:, i]) :
                                     lifted_B(Vu[:, i], Vu[:, j]) .+ lifted_B(Vu[:, j], Vu[:, i]))
                end
                push!(models, ("Galerkin", NaN, (A = Wt * A * Vu, N = Wt * Nv * Vu, H = H)))
            end
            for λ in p.λs
                push!(models, ("OpInf", λ, opinf_fit(Zh, Zhd, u; input = true, quad = true, λ)))
            end
            for (mname, λ, ops) in models
                if mname == "OpInf"
                    gal = label == "lifted" ? models[1][3] : nothing
                    push!(opdiff, (label, k, λ, ops.condD,
                        gal === nothing ? NaN : relerr(ops.A, gal.A), gal === nothing ? NaN : relerr(ops.N, gal.N),
                        gal === nothing ? NaN : relerr(ops.H, gal.H)))
                end
                Zr = s .* (V * integrate((dq, q, τ) -> rom_f!(dq, q, τ, ops, vfun), Zh[:, 1], τs, s -> (s, 1.0), p.rom_sub))
                Pr = label == "lifted" ? recover_P(Zr[1:N, :]) : Zr[1:N, :]
                eP = isnan(Pr[1, end]) ? NaN : relerr(Pr[:, ipr], P[:, ipr])
                eP_tr = isnan(Pr[1, itr[end]]) ? NaN : relerr(Pr[:, itr], P[:, itr])
                push!(rows, (label, k, mname, λ, errs(Zr, Z, itr), errs(Zr, Z, ipr), eP_tr, eP))
                store[(label, k, mname, λ)] = Zr
            end
        end
    end

    println("\nrelative error ‖Z_rom - Z‖_F/‖Z‖_F of the ROM state, and of P  (train τ ≤ $(p.τ_train) | predict τ ≤ $(p.τ_end))")
    @printf "%-7s %3s %-10s %7s  %10s %10s  %10s %10s\n" "state" "k" "model" "λ" "train" "predict" "P train" "P predict"
    for r in rows
        length(r) == 7 ? @printf("%-7s %3d %-10s %7s  %10.2e %10.2e\n", r[1], r[2], r[3], "", r[5], r[6]) :
                         @printf("%-7s %3d %-10s %7.0e  %10.2e %10.2e  %10.2e %10.2e\n", r...)
    end

    println("\ncond(D) and distance of the learned operators from the intrusive (Galerkin) ones, ‖Ô - O_gal‖/‖O_gal‖:")
    @printf "%-7s %3s %7s %10s %10s %10s %10s\n" "state" "k" "λ" "cond(D)" "A" "N" "H"
    for r in opdiff
        @printf "%-7s %3d %7.0e %10.2e %10.2e %10.2e %10.2e\n" r...
    end

    # ---------------------------------------------------------------- figures
    q = plot(yscale = :log10, xlabel = "index", ylabel = "σᵢ / σ₁", title = "Unpolarized Gowdy: POD singular values (τ ≤ $(p.τ_train))",
             legend = :topright, size = (750, 480))
    for (label, names) in (("lifted", ("Π", "G", "R", "S")), ("naive", ("P", "Π", "Q", "Ξ = Q_τ")))
        for (b, nm) in enumerate(names)
            label == "naive" && nm == "Π" && continue
            sv = store[(label, :svals)][b]
            plot!(q, 1:40, max.(sv[1:40] / sv[1], 1e-17), marker = :circle, ms = 2, label = "$nm ($label)")
        end
    end
    savefig(q, joinpath(p.outdir, "$(p.name)_svals.png"))

    function pick(label, k, m)
        sel = [r for r in rows if length(r) == 8 && r[1] == label && r[2] == k && r[3] == m]
        sel[argmin([isnan(r[5]) ? Inf : r[5] for r in sel])]
    end
    q = plot(yscale = :log10, xlabel = "POD modes per variable k  (r = 4k)", ylabel = "relative error in P",
             title = "Unpolarized Gowdy (v0 = $(p.v0)): error in P, λ chosen on training data", legend = :bottomleft,
             size = (800, 500), left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)
    for (label, m, ls) in (("lifted", "OpInf", :solid), ("lifted", "Galerkin", :dash), ("naive", "OpInf", :dot))
        plot!(q, p.ks, [min(coalesce_nan(pick(label, k, m)[7]), 10) for k in p.ks], ls = ls, marker = :circle, label = "$label $m (train)")
        plot!(q, p.ks, [min(coalesce_nan(pick(label, k, m)[8]), 10) for k in p.ks], ls = ls, marker = :utriangle, label = "$label $m (predict)")
    end
    savefig(q, joinpath(p.outdir, "$(p.name)_error_vs_r.png"))

    # snapshots of P (and the lifted R) for a representative k
    kshow = p.ks[end]
    bl, bn, bg = pick("lifted", kshow, "OpInf"), pick("naive", kshow, "OpInf"), pick("lifted", kshow, "Galerkin")
    Pl = recover_P(store[("lifted", kshow, "OpInf", bl[4])][1:N, :])
    Pg = recover_P(store[("lifted", kshow, "Galerkin", NaN)][1:N, :])
    Pn = store[("naive", kshow, "OpInf", bn[4])][1:N, :]
    pl = []
    for τt in (0.5, 1.0, p.τ_train, p.τ_end)
        j = argmin(abs.(τs .- τt))
        g = plot(θs, P[:, j], lw = 2, label = "FOM", title = "τ = $(τs[j])", xlabel = "θ", ylabel = "P")
        plot!(g, θs, Pl[:, j], lw = 2, ls = :dash, label = "lifted OpInf")
        plot!(g, θs, Pg[:, j], lw = 1.5, ls = :dashdot, label = "lifted Galerkin")
        plot!(g, θs, Pn[:, j], lw = 2, ls = :dot, label = "naive OpInf")
        push!(pl, g)
    end
    savefig(plot(pl..., layout = (2, 2), size = (950, 700)), joinpath(p.outdir, "$(p.name)_snapshots.png"))

    et(A) = [isnan(A[1, j]) ? NaN : norm(A[:, j] - P[:, j]) / norm(P[:, j]) for j in eachindex(τs)]
    g = plot(τs[2:end], et(Pl)[2:end], yscale = :log10, xlabel = "τ", ylabel = "‖P_rom - P‖ / ‖P‖",
             label = "lifted OpInf (λ=$(bl[4]))", title = "Unpolarized Gowdy (v0 = $(p.v0)), k = $kshow per variable", legend = :bottomright)
    plot!(g, τs[2:end], et(Pg)[2:end], label = "lifted Galerkin", ls = :dashdot)
    plot!(g, τs[2:end], et(Pn)[2:end], label = "naive OpInf (λ=$(bn[4]))", ls = :dot)
    vline!(g, [p.τ_train], color = :black, ls = :dash, label = "end of training")
    savefig(g, joinpath(p.outdir, "$(p.name)_error_vs_tau.png"))

    hm = heatmap(τs, θs, P, xlabel = "τ", ylabel = "θ", title = "FOM P(τ, θ), v0 = $(p.v0)", c = :balance)
    hq = heatmap(τs, θs, X[2N+1:3N, :], xlabel = "τ", ylabel = "θ", title = "FOM Q(τ, θ)", c = :balance)
    savefig(plot(hm, hq, layout = (1, 2), size = (1000, 400)), joinpath(p.outdir, "$(p.name)_fom.png"))
    println("figures written to ", p.outdir)
    return (; rows, val, lres, opdiff)
end
coalesce_nan(x) = isnan(x) ? 10.0 : x

if abspath(PROGRAM_FILE) == @__FILE__
    run_unpolarized(unp_params)
    run_unpolarized(unp_hard_params)
end
