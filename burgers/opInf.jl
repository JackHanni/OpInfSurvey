# Operator Inference ROM for the Burgers' data written by burgers_fd.jl.
#
# Fits dq̂/dt = Â q̂ + Ĥ (q̂ ⊗ q̂) in an r-dimensional POD basis by Tikhonov-regularized least
# squares (normal equations), integrates the ROM with RK4 and writes output/burgers_rom.gif.
#
# Run:  julia burgers/opInf.jl     (after julia burgers/burgers_fd.jl)

using LinearAlgebra, JLD2
include("burgers_fd.jl")   # burgers_rhs!, rk4_step!

datafile = joinpath(@__DIR__, "output", "burgers_data.jld2")
@load datafile xs ts U ν

r = 15                     # ROM dimension
λ = 1e-4                   # Tikhonov regularization parameter
V = svd(U).U[:, 1:r]       # POD basis
Q̂ = V' * U                 # reduced snapshots, one per column

"The r(r+1)/2 distinct products q_i q_j (j ≤ i): q ⊗ q without the repeated entries."
compact_kron(q) = [q[i] * q[j] for i in eachindex(q) for j in 1:i]

# Data matrix D = [Q̂ᵀ  c(Q̂)ᵀ] (one row per snapshot) and right-hand side R = (Vᵀ dU/dt)ᵀ
D = [Q̂' reduce(vcat, [compact_kron(q̂)' for q̂ in eachcol(Q̂)])]
dUdt = reduce(hcat, [burgers_rhs!(similar(U[:, k]), U[:, k], ν, xs[2] - xs[1]) for k in axes(U, 2)])
R = (V' * dUdt)'
O = ((D' * D + λ * I(size(D, 2))) \ (D' * R))'     # normal equations; O = [Â Ĥ]
Â = O[:, 1:r]
Ĥ = O[:, r+1:end]

# Integrate the ROM from the first reduced snapshot, with RK4 substeps between saved times
rom_rhs!(dq, q) = (dq .= Â * q .+ Ĥ * compact_kron(q))
q = copy(Q̂[:, 1])
k1, k2, k3, k4, stage = (similar(q) for _ in 1:5)
Q = zeros(r, length(ts))
Q[:, 1] .= q
substeps = 10
dt = (ts[2] - ts[1]) / substeps
for n in 1:length(ts)-1
    for _ in 1:substeps
        rk4_step!(q, dt, rom_rhs!, k1, k2, k3, k4, stage)
    end
    Q[:, n+1] .= q
end
U_rom = V * Q
println("relative error: ", norm(U_rom - U) / norm(U))

# Animation: full model vs ROM
u_min, u_max = extrema(U)
padding = 0.05 * max(u_max - u_min, eps())
max_abs_error = [norm(U[:, k] - U_rom[:, k], Inf) for k in eachindex(ts)]
animation = @animate for k in eachindex(ts)
    plot(xs, U[:, k], label = "full model", xlabel = "x", ylabel = "u",
         ylims = (u_min - padding, u_max + padding),
         title = "Burgers, r = $r,  t = $(round(ts[k], digits = 2)),  ‖res‖∞ = $(round(max_abs_error[k], sigdigits = 3))")
    plot!(xs, U_rom[:, k], label = "OpInf ROM", linestyle = :dash)
end
gif(animation, joinpath(@__DIR__, "output", "burgers_rom.gif"), fps = 30)
