# Shared pieces for the Gowdy T³ experiments: periodic grid, finite-difference
# operators, block POD, operator inference with an input term, ROM integration.
#
# Gowdy T³ vacuum Einstein equations (Berger & Moncrief 1993, eqs. 3.2-3.3),
# written in the areal time t = e^{-τ} (singularity at t = 0):
#
#   P_tt + P_t/t - P_θθ = e^{2P} (Q_t² - Q_θ²)
#   Q_tt + Q_t/t - Q_θθ = -2 (P_t Q_t - P_θ Q_θ)
#
# θ ∈ [0, 2π) periodic.  Polarized case: Q ≡ 0.
#
# Every model here has the form   dx/dt = A x + H (x ⊗ x) + u(t) N x,  u(t) = 1/t,
# so the explicit time dependence is treated as a known scalar input
# (u = 1/t when t is the time variable, u = e^{-τ} when τ = -ln t is; see the
# individual scripts).

using LinearAlgebra, SparseArrays, JLD2, Printf
include(joinpath(@__DIR__, "..", "burgers", "burgers_fd.jl"))   # rk4_step! (also loads Plots)

# ------------------------------------------------------------------ grid / operators
theta_grid(N) = collect(range(0, 2π, length = N + 1)[1:N])

"4th-order centered periodic first-derivative matrix."
function d1_matrix(N)
    h = 2π / N
    I_, J_, V_ = Int[], Int[], Float64[]
    for i in 1:N, (o, c) in ((-2, 1), (-1, -8), (1, 8), (2, -1))
        push!(I_, i); push!(J_, mod1(i + o, N)); push!(V_, c / (12h))
    end
    sparse(I_, J_, V_, N, N)
end

"4th-order centered periodic second-derivative matrix."
function d2_matrix(N)
    h = 2π / N
    I_, J_, V_ = Int[], Int[], Float64[]
    for i in 1:N, (o, c) in ((-2, -1), (-1, 16), (0, -30), (1, 16), (2, -1))
        push!(I_, i); push!(J_, mod1(i + o, N)); push!(V_, c / (12h^2))
    end
    sparse(I_, J_, V_, N, N)
end

# ------------------------------------------------------------------ OpInf utilities
compact_kron(x) = [x[i] * x[j] for i in eachindex(x) for j in 1:i]
relerr(A, B) = norm(A - B) / norm(B)

"""
    block_pod(X, nvar, k) -> V, svals

Separate POD basis (k modes) for each of the `nvar` stacked variables of the
snapshot matrix X (each block has N = size(X,1) ÷ nvar rows).  Returns the block
diagonal basis V (nvar*N × nvar*k) and the singular values of each block.
"""
function block_pod(X, nvar, k)
    N = size(X, 1) ÷ nvar
    Vs, svals = Matrix{Float64}[], Vector{Float64}[]
    for b in 1:nvar
        F = svd(X[(b-1)*N+1:b*N, :])
        push!(Vs, F.U[:, 1:k]); push!(svals, F.S)
    end
    V = zeros(nvar * N, nvar * k)
    for b in 1:nvar
        V[(b-1)*N+1:b*N, (b-1)*k+1:b*k] .= Vs[b]
    end
    return V, svals
end

"""
    opinf_fit(Xh, Xh_dot, u; linear=true, input=true, quad=false, λ=0.0)

Regularized least squares for dx̂/dt = Â x̂ + Ĥ (x̂ ⊗ x̂)_compact + u(t) N̂ x̂.
Solved as the augmented problem  min ‖D Oᵀ - Ẏ‖² + λ‖Oᵀ‖²  via QR (backslash on
[D; √λ I]) rather than the normal equations, which would square cond(D).
Returns a NamedTuple of operators (missing terms are `nothing`) and cond(D).
"""
function opinf_fit(Xh, Xh_dot, u; linear = true, input = true, quad = false, λ = 0.0)
    r, K = size(Xh)
    blocks = Matrix{Float64}[]
    linear && push!(blocks, Xh')
    input  && push!(blocks, (Xh .* u')')
    quad   && push!(blocks, reduce(vcat, [compact_kron(x)' for x in eachcol(Xh)]))
    D = reduce(hcat, blocks)
    Y = Xh_dot'
    p = size(D, 2)
    O = (λ > 0 ? [D; sqrt(λ) * I(p)] \ [Y; zeros(p, r)] : D \ Y)'
    c = 0
    A = linear ? O[:, c+1:c+r] : nothing;  c += linear ? r : 0
    N = input  ? O[:, c+1:c+r] : nothing;  c += input  ? r : 0
    H = quad   ? O[:, c+1:end] : nothing
    return (; A, N, H, condD = cond(D), ncols = p)
end

"Reduced right-hand side dx̂/ds = Â x̂ + u(s) N̂ x̂ + Ĥ(x̂⊗x̂) at time s, input function `ufun`."
function rom_f!(dq, q, s, ops, ufun)
    dq .= 0
    ops.A === nothing || mul!(dq, ops.A, q, 1.0, 1.0)
    ops.N === nothing || mul!(dq, ops.N, q, ufun(s), 1.0)
    ops.H === nothing || (dq .+= ops.H * compact_kron(q))
    return dq
end

"""
    integrate(f!, x0, ss, tmap, sub) -> X

RK4 integration of dx/ds = t'(s) f(x, t(s)), saving at the points `ss`
(uniformly spaced) with `sub` substeps between saves.  `tmap(s)` returns
(t, dt/ds).  Using s = τ with t = e^{-τ} lets us march toward the singularity
with uniform steps in τ.  Stops early (rest = NaN) if the state blows up.
"""
function integrate(f!, x0, ss, tmap, sub; blowup = 1e8)
    x = copy(x0)
    X = fill(NaN, length(x), length(ss)); X[:, 1] .= x
    k1, k2, k3, k4, tmp = (similar(x) for _ in 1:5)
    scale = blowup * max(norm(x0), 1.0)
    for n in 1:length(ss)-1
        h = (ss[n+1] - ss[n]) / sub
        for m in 0:sub-1
            s0 = ss[n] + m * h
            # rk4_step! wants an autonomous f!(du, u); track the stage time by hand
            stage = Ref(0)
            g!(du, v) = begin
                si = s0 + (stage[] == 0 ? 0.0 : stage[] == 3 ? h : h / 2)
                stage[] += 1
                t, dtds = tmap(si)
                f!(du, v, t); du .*= dtds
            end
            rk4_step!(x, h, g!, k1, k2, k3, k4, tmp)
        end
        (all(isfinite, x) && norm(x) < scale) || return X
        X[:, n+1] .= x
    end
    return X
end
