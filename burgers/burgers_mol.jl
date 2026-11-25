# 1D viscous Burgers' equation solved with MethodOfLines.jl, as a cross-check of burgers_fd.jl:
#
#   u_t + u u_x = ν u_xx,   x ∈ [0, 1],  t ∈ [0, 2]
#   u(x, 0) = sin(πx),  u(0, t) = u(1, t) = 0
#
# Run:  julia burgers/burgers_mol.jl     (slow: ≈ 30 min, mostly ModelingToolkit compiling)

using MethodOfLines, ModelingToolkit, OrdinaryDiffEq, Plots, JLD2
using MethodOfLines: Interval   # no longer exported by ModelingToolkit
using OrdinaryDiffEqBDF: DFBDF  # DAE solver, no longer re-exported by OrdinaryDiffEq v7

@parameters x t
@variables u(..)
Dt = Differential(t)
Dx = Differential(x)
Dxx = Differential(x)^2

x_min, x_max = 0.0, 1.0
t_min, t_max = 0.0, 2.0
ν = 0.02                                  # viscosity: small ν -> steep front, nonlinear term matters
u0(x) = sin(π * x)                        # initial condition

equation = Dt(u(x, t)) + u(x, t) * Dx(u(x, t)) ~ ν * Dxx(u(x, t))

initial_and_boundary_conditions = [u(x, t_min) ~ u0(x),
                                   u(x_min, t) ~ 0.0,
                                   u(x_max, t) ~ 0.0]

domains = [x ∈ Interval(x_min, x_max),
           t ∈ Interval(t_min, t_max)]

@named pdesys = PDESystem(equation, initial_and_boundary_conditions, domains, [x, t], [u(x, t)])

# Method of lines: finite differences in x, ODE solver in t
N = 201                                   # grid points (keep N ≳ 1/ν to resolve the front)
dx = (x_max - x_min) / (N - 1)
discretization = MOLFiniteDifference([x => dx], t)

problem = discretize(pdesys, discretization)    # PDE -> ODE/DAE problem
# MOL returns a DAE problem (the boundary nodes are algebraic equations), so use a DAE solver
solution = solve(problem, DFBDF(), saveat = 0.01)

xs = solution[x]
ts = solution[t]
U = solution[u(x, t)]

# save solution data next to this script
datafile = joinpath(@__DIR__, "output", "burgers_mol_data.jld2")
jldsave(datafile; xs, ts, U, ν)
println("saved solution to ", datafile)

# static snapshot plot
snapshot_idx = 1:20:length(ts)
plot(xs, U[:, snapshot_idx], xlabel = "x", ylabel = "u", label = permutedims(["t = $(ts[k])" for k in snapshot_idx]))
savefig(joinpath(@__DIR__, "output", "burgers_mol_snapshots.png"))

# animation
animation = @animate for k in eachindex(ts)
    plot(xs, U[:, k], xlabel = "x", ylabel = "u", ylims = (-0.05, 1.05), legend = false,
         title = "Burgers, ν = $(ν),  t = $(round(ts[k], digits = 2))")
end
gif(animation, joinpath(@__DIR__, "output", "burgers_mol.gif"), fps = 30)
