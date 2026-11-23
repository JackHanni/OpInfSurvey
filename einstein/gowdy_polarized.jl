# Part A: polarized Gowdy T³ (Q ≡ 0) — exact Bessel-function solutions + OpInf.
#
#   P_tt + P_t/t - P_θθ = 0                   (Berger & Moncrief 1993, eq. 3.2 with Q = 0,
#                                              rewritten in t = e^{-τ})
#   general solution (ibid. eq. 3.18):  P = α + β ln t + Σ_n Z₀(n t)(a_n cos nθ + b_n sin nθ),
#   Z₀ = any solution of Bessel's equation of order zero (J₀, Y₀).
#
# First-order form, state x = [P; Π] with Π = P_t:
#   dP/dt = Π,   dΠ/dt = P_θθ - (1/t) Π
# i.e. dx/dt = A x + u(t) N x with u = 1/t.  Linear but NON-autonomous.
#
# Run:  julia gowdy_polarized.jl     (≈ 1 min incl. Plots compilation)

include(joinpath(@__DIR__, "gowdy_common.jl"))
using SpecialFunctions, Random

# ------------------------------------------------------------------ parameters
const pol_params = (
    N       = 128,          # grid points in θ
    nmax    = 10,           # highest Fourier mode in the exact solution
    decay   = 0.75,         # coefficient amplitude ∝ decay^n
    α = 0.3, β = 0.5,       # homogeneous part α + β ln t
    seed    = 1234,
    t0      = 1.0,          # initial time (t = e^{-τ}; expanding direction)
    t_train = 11.0,         # end of training window
    t_end   = 16.0,         # end of prediction window
    save_dt = 0.02,
    sub     = 4,            # RK4 substeps per saved interval for ROMs
    ks      = [2, 4, 6, 8, 10, 12, 14, 16, 18, 21],   # POD modes per variable
    λs      = [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2],
    outdir  = mkpath(joinpath(@__DIR__, "output")),
)

# ------------------------------------------------------------------ exact solution
"Random coefficients (a_n^J, a_n^Y, b_n^J, b_n^Y) for modes n = 1..nmax."
function pol_coeffs(p)
    rng = MersenneTwister(p.seed)
    [p.decay^n .* randn(rng, 4) for n in 1:p.nmax]
end

"""
    pol_exact(θs, t, c, p) -> P, Π, P_θθ

Exact P, P_t and P_θθ on the grid at time t (Bessel identities J₀' = -J₁, Y₀' = -Y₁).
"""
function pol_exact(θs, t, c, p)
    P   = fill(p.α + p.β * log(t), length(θs))
    Π   = fill(p.β / t, length(θs))
    Pθθ = zeros(length(θs))
    for n in 1:p.nmax
        aJ, aY, bJ, bY = c[n]
        Z  = (aJ * besselj0(n*t) + aY * bessely0(n*t)) .* cos.(n .* θs) .+
             (bJ * besselj0(n*t) + bY * bessely0(n*t)) .* sin.(n .* θs)
        Zt = -n .* ((aJ * besselj1(n*t) + aY * bessely1(n*t)) .* cos.(n .* θs) .+
                    (bJ * besselj1(n*t) + bY * bessely1(n*t)) .* sin.(n .* θs))
        P .+= Z;  Π .+= Zt;  Pθθ .-= n^2 .* Z
    end
    return P, Π, Pθθ
end

function pol_data(p)
    θs = theta_grid(p.N)
    c  = pol_coeffs(p)
    ts = collect(p.t0:p.save_dt:p.t_end)
    X, Xdot = zeros(2p.N, length(ts)), zeros(2p.N, length(ts))
    for (k, t) in enumerate(ts)
        P, Π, Pθθ = pol_exact(θs, t, c, p)
        X[:, k]    .= [P; Π]
        Xdot[:, k] .= [Π; Pθθ .- Π ./ t]        # exact time derivative of [P; Π]
    end
    return θs, ts, X, Xdot, c
end

# ------------------------------------------------------------------ checks
"""
Independent checks of the exact data:
 (i)  PDE residual P_tt + P_t/t - P_θθ computed from P ALONE by centered differences in t
      (step h) and the FD matrix in θ — should fall like h² until the FD-in-θ floor;
 (ii) consistency of the closed-form Π with a centered difference of P;
 (iii) 4th-order FD error of P_θθ on the grid (this is what intrusive Galerkin sees).
"""
function pol_checks(θs, ts, X, Xdot, p)
    N, D2 = p.N, d2_matrix(p.N)
    c = pol_coeffs(p)
    Pof(t) = pol_exact(θs, t, c, p)[1]
    tc = ts[1:25:end]
    res = Float64[]
    for h in (1e-2, 5e-3, 2.5e-3)
        num = 0.0; den = 0.0
        for t in tc[2:end-1]
            Pm, P0, Pp = Pof(t - h), Pof(t), Pof(t + h)
            Ptt = (Pp - 2P0 + Pm) / h^2
            Pt  = (Pp - Pm) / (2h)
            Pθθ = pol_exact(θs, t, c, p)[3]          # spectral (exact) P_θθ
            num += norm(Ptt + Pt / t - Pθθ)^2;  den += norm(Pθθ)^2
        end
        push!(res, sqrt(num / den))
    end
    e_Π = relerr(reduce(hcat, [(Pof(t + 1e-4) - Pof(t - 1e-4)) / 2e-4 for t in tc]),
                 reduce(hcat, [pol_exact(θs, t, c, p)[2] for t in tc]))
    e_x = relerr(D2 * X[1:N, :], Xdot[N+1:end, :] + X[N+1:end, :] ./ ts')
    @printf "check: PDE residual of exact P (centered FD in t), h = 1e-2, 5e-3, 2.5e-3: %.2e %.2e %.2e\n" res...
    @printf "check: closed-form Π vs centered difference of P (h = 1e-4):            %.2e\n" e_Π
    @printf "check: ‖D2_FD P - P_θθ(exact)‖/‖P_θθ‖ (4th-order FD, N = %d):           %.2e\n" N e_x
    return (; pde_residual = res, e_Π, e_x)
end

# ------------------------------------------------------------------ experiment
function run_polarized(p = pol_params)
    θs, ts, X, Xdot, c = pol_data(p)
    N = p.N
    checks = pol_checks(θs, ts, X, Xdot, p)
    jldsave(joinpath(p.outdir, "gowdy_polarized_data.jld2"); θs, ts, X, Xdot,
            coeffs = reduce(hcat, c), α = p.α, β = p.β, nmax = p.nmax, t_train = p.t_train)

    itr = findall(t -> t <= p.t_train + 1e-12, ts)
    ipr = findall(t -> t >  p.t_train + 1e-12, ts)
    tmap(s) = (s, 1.0)
    u = 1 ./ ts[itr]

    # full-order (intrusive) operators of the FD semi-discretization, for Galerkin
    D2 = d2_matrix(N)
    Afom = [spzeros(N, N) sparse(1.0I, N, N); D2 spzeros(N, N)]
    Nfom = [spzeros(N, N) spzeros(N, N); spzeros(N, N) -sparse(1.0I, N, N)]

    _, svals = block_pod(X[:, itr], 2, 1)
    results = Dict{Any, Any}()
    err(Xr, idx) = relerr(Xr[:, idx], X[:, idx])
    rows = []
    for k in p.ks
        V, _ = block_pod(X[:, itr], 2, k)
        Xh, Xhd = V' * X[:, itr], V' * Xdot[:, itr]
        floor_tr = err(V * (V' * X), itr);  floor_pr = err(V * (V' * X), ipr)
        gal = (A = V' * Afom * V, N = V' * Nfom * V, H = nothing)
        Xg = V * integrate((dq, q, t) -> rom_f!(dq, q, t, gal, inv), V' * X[:, 1], ts, tmap, p.sub)
        push!(rows, (k, "POD floor", NaN, floor_tr, floor_pr))
        push!(rows, (k, "Galerkin", NaN, err(Xg, itr), err(Xg, ipr)))
        for λ in p.λs, (name, inp) in (("OpInf LTI", false), ("OpInf A+N/t", true))
            ops = opinf_fit(Xh, Xhd, u; input = inp, λ)
            Xr = V * integrate((dq, q, t) -> rom_f!(dq, q, t, ops, inv), V' * X[:, 1], ts, tmap, p.sub)
            push!(rows, (k, name, λ, err(Xr, itr), err(Xr, ipr)))
            results[(Symbol(name), k, λ)] = Xr
        end
        results[(:Galerkin, k)] = Xg
        println("  k = $k done")
    end

    # ---------------------------------------------------------------- report
    println("\nrelative error ‖X_rom - X‖_F/‖X‖_F   (train t ≤ $(p.t_train) | predict t > $(p.t_train))")
    @printf "%4s  %-12s %8s   %10s %10s\n" "k" "model" "λ" "train" "predict"
    for (k, name, λ, a, b) in rows
        @printf "%4d  %-12s %8.0e   %10.2e %10.2e\n" k name λ a b
    end
    best(name, k) = begin
        sel = [r for r in rows if r[1] == k && r[2] == name]
        sel[argmin([isnan(r[4]) ? Inf : r[4] for r in sel])]   # pick λ by training error
    end

    # ---------------------------------------------------------------- operator recovery (full rank)
    # With k = 2nmax+1 the POD basis spans the data exactly, so the learned operators can be
    # compared with the exact ones: N̂ should equal Vᵀ N V (the -I on the Π block), and the
    # Π-P block of Â should match the projected *continuous* Laplacian (Fourier, -n²),
    # not the 4th-order FD matrix the intrusive Galerkin model uses.
    # A single trajectory turns out NOT to determine the operators (D is numerically rank
    # deficient), so we also train on several trajectories (other random coefficients, same PDE)
    # and test on the original one.
    kfull = 2p.nmax + 1
    opcheck = []
    m = -(N ÷ 2 - 1):(N ÷ 2)
    Lspec = [sum(-(mm^2) * cos(mm * (θs[i] - θs[j])) for mm in m) / N for i in 1:N, j in 1:N]
    trajs = [(X, Xdot)]
    for s in 1:3
        _, _, Xs, Xds, _ = pol_data(merge(p, (seed = p.seed + s,)))
        push!(trajs, (Xs, Xds))
    end
    println("\noperator recovery at k = $kfull, test on the original trajectory:")
    @printf "%6s %7s %9s %10s %10s %10s %10s %10s\n" "ntraj" "λ" "cond(D)" "‖N̂-N‖" "‖Â-L_sp‖" "‖Â-L_FD‖" "train" "predict"
    for ntraj in (1, 2, 4), λ in (0.0, 1e-8)
        Xtr  = reduce(hcat, [trajs[i][1][:, itr] for i in 1:ntraj])
        Xdtr = reduce(hcat, [trajs[i][2][:, itr] for i in 1:ntraj])
        V, _ = block_pod(Xtr, 2, kfull)
        ops = opinf_fit(V' * Xtr, V' * Xdtr, repeat(u, ntraj); input = true, λ)
        VP, VΠ = V[1:N, 1:kfull], V[N+1:end, kfull+1:end]
        Aπp = ops.A[kfull+1:end, 1:kfull]
        Xr = V * integrate((dq, q, t) -> rom_f!(dq, q, t, ops, inv), V' * X[:, 1], ts, tmap, p.sub)
        row = (; ntraj, λ, condD = ops.condD, eN = relerr(ops.N, V' * Nfom * V),
               eA_spec = relerr(Aπp, VΠ' * Lspec * VP), eA_fd = relerr(Aπp, VΠ' * D2 * VP),
               train = err(Xr, itr), predict = err(Xr, ipr))
        push!(opcheck, row)
        @printf "%6d %7.0e %9.1e %10.2e %10.2e %10.2e %10.2e %10.2e\n" row...
    end
    V, _ = block_pod(X[:, itr], 2, kfull)
    @printf "(projected FD vs spectral Laplacian differ by %.2e)\n" relerr(V[N+1:end, kfull+1:end]' * D2 * V[1:N, 1:kfull],
                                                                     V[N+1:end, kfull+1:end]' * Lspec * V[1:N, 1:kfull])

    # ---------------------------------------------------------------- figures
    plt = plot(yscale = :log10, xlabel = "index", ylabel = "σᵢ / σ₁", title = "Polarized Gowdy: POD singular values",
               legend = :topright)
    for (b, nm) in enumerate(("P", "Π = P_t"))
        s = svals[b] / svals[b][1]
        plot!(plt, 1:30, max.(s[1:30], 1e-17), marker = :circle, ms = 3, label = nm)
    end
    savefig(plt, joinpath(p.outdir, "polarized_svals.png"))

    cols = Dict("POD floor" => :black, "Galerkin" => :green, "OpInf LTI" => :orange, "OpInf A+N/t" => :blue)
    panels = []
    for (col, ttl) in ((4, "training window t ≤ $(p.t_train)"), (5, "prediction t ∈ ($(p.t_train), $(p.t_end)]"))
        q = plot(yscale = :log10, xlabel = "POD modes per variable k  (r = 2k)", ylabel = "relative error",
                 title = ttl, legend = :bottomleft, ylims = (1e-8, 10))
        for name in ("POD floor", "Galerkin", "OpInf LTI", "OpInf A+N/t")
            plot!(q, p.ks, [clamp(best(name, k)[col], 1e-8, 10.0) for k in p.ks], color = cols[name],
                  ls = name == "POD floor" ? :dot : :solid, marker = :circle, label = name)
        end
        push!(panels, q)
    end
    savefig(plot(panels..., layout = (1, 2), size = (1100, 450), left_margin = 5Plots.mm, bottom_margin = 5Plots.mm,
                 plot_title = "Polarized Gowdy: ROM error vs k (λ picked on training error; values clipped to [1e-8, 10])",
                 plot_titlefontsize = 10), joinpath(p.outdir, "polarized_error_vs_r.png"))

    kshow = 12
    ib = best("OpInf A+N/t", kshow); il = best("OpInf LTI", kshow)
    Xb = results[(Symbol("OpInf A+N/t"), kshow, ib[3])]; Xl = results[(Symbol("OpInf LTI"), kshow, il[3])]
    pl = []
    for tt in (1.0, 6.0, 11.0, 16.0)
        j = argmin(abs.(ts .- tt))
        q = plot(θs, X[1:N, j], label = "exact", lw = 2, title = "t = $(ts[j])", xlabel = "θ", ylabel = "P")
        plot!(q, θs, Xb[1:N, j], ls = :dash, lw = 2, label = "OpInf A+N/t")
        plot!(q, θs, Xl[1:N, j], ls = :dot, lw = 2, label = "OpInf LTI")
        push!(pl, q)
    end
    savefig(plot(pl..., layout = (2, 2), size = (900, 650), plot_title = "Polarized Gowdy, k = $kshow per variable",
                 plot_titlefontsize = 11), joinpath(p.outdir, "polarized_snapshots.png"))

    # error in time
    et(Xr) = [norm(Xr[:, j] - X[:, j]) / norm(X[:, j]) for j in eachindex(ts)]
    panels = []
    for kk in (kshow, p.ks[end])
        b1, b2 = best("OpInf A+N/t", kk), best("OpInf LTI", kk)
        q = plot(ts, et(results[(Symbol("OpInf A+N/t"), kk, b1[3])]), yscale = :log10, color = :blue,
                 label = "OpInf A+N/t (λ=$(b1[3]))", xlabel = "t", ylabel = "‖x_rom(t) - x(t)‖ / ‖x(t)‖",
                 title = "k = $kk per variable", legend = :bottomright)
        plot!(q, ts, et(results[(Symbol("OpInf LTI"), kk, b2[3])]), color = :orange, label = "OpInf LTI (λ=$(b2[3]))")
        plot!(q, ts, et(results[(:Galerkin, kk)]), color = :green, label = "Galerkin (FD operator)", ls = :dash)
        vline!(q, [p.t_train], color = :black, ls = :dash, label = "end of training")
        push!(panels, q)
    end
    savefig(plot(panels..., layout = (1, 2), size = (1100, 430), left_margin = 5Plots.mm, bottom_margin = 5Plots.mm),
            joinpath(p.outdir, "polarized_error_vs_t.png"))

    # λ sweep
    panels = []
    for kk in (kshow, p.ks[end])
        q = plot(xscale = :log10, yscale = :log10, xlabel = "λ", ylabel = "relative error", legend = :topleft,
                 title = "λ sweep, k = $kk (λ = 0 in the table)")
        for (name, cl) in (("OpInf LTI", :orange), ("OpInf A+N/t", :blue))
            sel = [r for r in rows if r[1] == kk && r[2] == name && r[3] > 0]
            plot!(q, [r[3] for r in sel], [clamp(r[4], 1e-8, 10) for r in sel], color = cl, marker = :circle, label = "$name train")
            plot!(q, [r[3] for r in sel], [clamp(r[5], 1e-8, 10) for r in sel], color = cl, ls = :dash, marker = :utriangle, label = "$name predict")
        end
        push!(panels, q)
    end
    savefig(plot(panels..., layout = (1, 2), size = (1100, 430), left_margin = 5Plots.mm, bottom_margin = 5Plots.mm),
            joinpath(p.outdir, "polarized_lambda_sweep.png"))
    println("figures written to ", p.outdir)
    return (; rows, checks, svals, opcheck)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_polarized()
end
