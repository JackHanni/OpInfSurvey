# 3D animation of the SXS:BBH:0305 merger: the two black holes orbit, merge and ring down, while
# the gravitational wave they emit streams outward through the orbital plane.
#
# Black holes: coordinate centres and areal masses of the simulation's apparent horizons
# (A and B before merger, the common horizon C after), drawn with size proportional to mass.
#
# Wave: the surface height is the plus-polarized strain in the orbital plane at the retarded time,
#     height(ρ, φ, t) ∝ h₊(t − ρ, θ = π/2, φ),
# so each ring of radius ρ shows what the wave looked like ρ M earlier and the pattern moves
# outward at the speed of light. The real strain falls off like 1/ρ; here the height falls off like
# 1/√ρ, so the outer rings stay visible (an exaggeration, like the height itself). This draws
# the far-zone waveform in the near zone, so it is a visualization, not the metric near the holes:
# the height is exaggerated and faded out close to the black holes.
#
# Writes einstein/output/merger_3d.mp4, which then plays at full speed. Frames are rendered by
# several Julia processes in parallel. Needs both data files:
#   python einstein/fetch_sxs.py      (or  --horizons-only  if the strain file already exists)
# Run:  julia einstein/animate_merger_3d.jl [number of worker processes, default 6]

include("opinf_sxs.jl")        # load_sxs, OUTDIR, HERE
include("waveform_tools.jl")   # spin-weighted harmonics, frames_by_phase, interpolate_linear

const HORIZONS_FILE = joinpath(HERE, "data", "sxs_bbh_0305_horizons.h5")
const PLANE_RADIUS = 100.0              # radius of the plane shown, in M
const GRID_POINTS = 201                 # grid points per side of the plane
const FADE_RADII = (5.0, 25.0)          # wave height ramps up between these radii, then falls off like 1/√ρ
const MAX_WAVE_HEIGHT = 7.0             # height of the largest crest, in M (exaggeration)
const SHOW_FROM = 2900.0                # first time shown (merger is at ≈ 3690 M)
const SHOW_AFTER_PEAK = 50.0            # keep going this long after the peak of the wave
const PHASE_STEP = 0.22                 # h22 phase advance per frame, in radians
const TRAIL_LENGTH = 120.0              # length of the orbital trail drawn behind each hole, in M
const VIDEO_FPS = 30
const RENDER_SIZE = (2000, 1430)        # GR leaves wide margins around 3D plots, so render large ...
const CROP = (w = 1040, h = 660, x = 500, y = 370)   # ... and crop to the scene with ffmpeg

# dark where the plane is flat, electric blue in troughs, orange on crests (full colour at ±MAX_WAVE_HEIGHT)
const WAVE_COLORS = cgrad([RGB(0.25, 0.65, 1.0), RGB(0.25, 0.65, 1.0), RGB(0.10, 0.10, 0.16),
                           RGB(1.0, 0.55, 0.15), RGB(1.0, 0.55, 0.15)], [0, 0.3, 0.5, 0.7, 1])

"Horizon trajectories: for each of A, B, C the times, centres (3 × n_t) and areal masses."
function load_horizons(path = HORIZONS_FILE)
    jldopen(path, "r") do file
        horizon(name) = (; t = file["$(name)_t"], center = file["$(name)_center"], mass = file["$(name)_areal_mass"])
        return (; A = horizon("A"), B = horizon("B"), C = horizon("C"))
    end
end

"Centre (x, y, z) and mass of a horizon at time τ (linear interpolation)."
function horizon_at(horizon, τ)
    center = [interpolate_linear(horizon.t, horizon.center[axis, :], [τ])[1] for axis in 1:3]
    mass = interpolate_linear(horizon.t, horizon.mass, [τ])[1]
    return center, mass
end

"Smooth ramp from 0 at ρ = ρ_start to 1 at ρ = ρ_end (cubic smoothstep, flat at both ends)."
function smooth_ramp(ρ, ρ_start, ρ_end)
    fraction = clamp((ρ - ρ_start) / (ρ_end - ρ_start), 0, 1)
    return fraction^2 * (3 - 2fraction)
end

# ---------------------------------------------------------------- wave in the orbital plane
t, X, _, ell, m = load_sxs()
h_modes = complex_modes(X)
h22 = h_modes[findfirst((ell .== 2) .& (m .== 2)), :]
t_peak = t[argmax(abs.(h22))]
dt = t[2] - t[1]

grid_coords = range(-PLANE_RADIUS, PLANE_RADIUS, length = GRID_POINTS)
point_x = [x for y in grid_coords, x in grid_coords]     # rows are y, columns are x (Plots convention)
point_y = [y for y in grid_coords, x in grid_coords]
point_radius = hypot.(point_x, point_y)
# height = strain × weight: 0 at the holes, ramping up, then a 1/√ρ fall-off
point_weight = smooth_ramp.(point_radius, FADE_RADII...) .* sqrt.(min.(1, FADE_RADII[2] ./ point_radius))
point_weight[point_radius .> PLANE_RADIUS] .= NaN   # round plane: NaN is not drawn
Y_plane = harmonics_matrix(fill(π / 2, length(point_x)), vec(atan.(point_y, point_x)), ell, m)

"""
h₊ at every grid point at simulation time `clock`, each point read at its retarded time
clock − ρ. Points the wave has not reached yet (retarded time before the data starts) are 0.
`modes` holds h_ℓm on the time grid `t`: the true modes by default, or a ROM's.
"""
function plane_strain(clock, modes = h_modes)
    strain = zeros(size(point_x))
    for p in eachindex(strain)
        retarded_time = clock - point_radius[p]
        if retarded_time < t[1] || retarded_time >= t[end]
            continue
        end
        position = (retarded_time - t[1]) / dt
        k = floor(Int, position) + 1
        weight = position - (k - 1)
        h = zero(ComplexF64)
        for j in axes(modes, 1)
            h += Y_plane[p, j] * ((1 - weight) * modes[j, k] + weight * modes[j, k+1])
        end
        strain[p] = real(h)
    end
    return strain
end

horizons = load_horizons()
merger_time = horizons.C.t[1]
frame_indices = frames_by_phase(t, h22, SHOW_FROM, t_peak + SHOW_AFTER_PEAK, PHASE_STEP)
frame_clocks = t[frame_indices]

# one height scale for the whole video, so the wave visibly grows towards merger
largest_strain = maximum(maximum(abs, filter(!isnan, plane_strain(clock) .* point_weight)) for clock in frame_clocks[1:10:end])
height_per_strain = MAX_WAVE_HEIGHT / largest_strain

# ---------------------------------------------------------------- drawing
"Black holes present at time τ, as (centre, mass) pairs: A and B before merger, C after."
function black_holes_at(τ)
    if τ < merger_time
        return [horizon_at(horizons.A, τ), horizon_at(horizons.B, τ)]
    else
        return [horizon_at(horizons.C, τ)]
    end
end

"Recent path of a horizon, from τ − TRAIL_LENGTH to τ."
function trail(horizon, τ)
    times = range(max(τ - TRAIL_LENGTH, horizon.t[1]), min(τ, horizon.t[end]), length = 200)
    return [interpolate_linear(horizon.t, horizon.center[axis, :], times) for axis in 1:2]
end

"Name of the phase of the coalescence at time τ, for the caption."
function stage_at(τ)
    if τ < merger_time - 10
        return "inspiral"
    elseif τ < t_peak + 10
        return "merger"
    else
        return "ringdown"
    end
end

"""
3D scene at time `clock`: the wave surface built from `modes`, plus the simulation's black holes.
Heights are clamped to the plot box, so a ROM that blows up stays drawable.
"""
function draw_frame(clock, frame_number, n_frames = length(frame_clocks); modes = h_modes)
    heights = clamp.(height_per_strain .* plane_strain(clock, modes) .* point_weight,
                     -2.5MAX_WAVE_HEIGHT, 2.5MAX_WAVE_HEIGHT)
    azimuth = 35 + 60 * frame_number / n_frames                      # slow camera orbit
    plt = surface(grid_coords, grid_coords, heights;
                  c = WAVE_COLORS, zlims = (-2.5MAX_WAVE_HEIGHT, 2.5MAX_WAVE_HEIGHT), colorbar = false,
                  camera = (azimuth, 36), axis = false, grid = false, legend = false,
                  size = RENDER_SIZE, background_color = :black)
    if clock < merger_time + TRAIL_LENGTH
        for horizon in (horizons.A, horizons.B)
            trail_x, trail_y = trail(horizon, clock)
            plot!(plt, trail_x, trail_y, fill(0.3, length(trail_x)); c = :gray70, lw = 2)
        end
    end
    for (center, mass) in black_holes_at(clock)
        scatter3d!(plt, [center[1]], [center[2]], [0.5];
                   markersize = 38mass, markercolor = :black, markerstrokecolor = :white, markerstrokewidth = 2)
    end
    return plt
end

"Crop a rendered frame to the scene and write the caption onto it, with ffmpeg."
function crop_and_caption(raw_png, out_png, caption)
    caption_file = raw_png * ".txt"
    write(caption_file, caption)
    filter = "crop=$(CROP.w):$(CROP.h):$(CROP.x):$(CROP.y)," *
             "drawtext=textfile=$(caption_file):fontcolor=white:fontsize=22:line_spacing=8:x=20:y=16:expansion=none"
    run(`ffmpeg -y -v error -i $raw_png -vf $filter $out_png`)
end

# ---------------------------------------------------------------- render
"Render frames worker_id, worker_id + n_workers, ... into `frame_dir` as frame_0001.png etc."
function render_frames(worker_id, n_workers, frame_dir)
    raw_png = joinpath(frame_dir, "raw_worker$(worker_id).png")
    for frame_number in worker_id:n_workers:length(frame_clocks)
        clock = frame_clocks[frame_number]
        savefig(draw_frame(clock, frame_number), raw_png)
        caption = "SXS:BBH:0305 binary black hole merger (simulation data)\n" *
                  @sprintf("t - t_peak = %+6.1f M    %s", clock - t_peak, stage_at(clock))
        crop_and_caption(raw_png, joinpath(frame_dir, @sprintf("frame_%04d.png", frame_number)), caption)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) == 4 && ARGS[1] == "--worker"
        render_frames(parse(Int, ARGS[2]), parse(Int, ARGS[3]), ARGS[4])
    else
        n_workers = isempty(ARGS) ? 6 : parse(Int, ARGS[1])
        println("Rendering $(length(frame_clocks)) frames with $n_workers processes, ",
                "t ∈ [$(frame_clocks[1]), $(frame_clocks[end])] M, ",
                "merger (common horizon) at $(round(merger_time, digits = 1)) M, peak of h22 at $t_peak M")
        frame_dir = mktempdir()
        julia = Base.julia_cmd()
        workers = [run(`$julia $(@__FILE__) --worker $id $n_workers $frame_dir`; wait = false) for id in 1:n_workers]
        foreach(wait, workers)
        video = joinpath(OUTDIR, "merger_3d.mp4")
        run(`ffmpeg -y -v error -framerate $VIDEO_FPS -i $(joinpath(frame_dir, "frame_%04d.png"))
                 -c:v libx264 -pix_fmt yuv420p -crf 20 $video`)
        rm(frame_dir; recursive = true)
        println("  → merger_3d.mp4 ($(round(length(frame_clocks) / VIDEO_FPS, digits = 1)) s at $VIDEO_FPS fps)")
    end
end
