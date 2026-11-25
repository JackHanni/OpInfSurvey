# Animations of the SXS:BBH:0305 binary-black-hole strain and of the OpInf ROMs fit to it.
#
# Writes to einstein/output/:
#   sky_strain.gif        h₊ over the whole sky (Mollweide map), inspiral → merger → ringdown
#   ringdown_sky_rom.gif  true strain vs the OpInf ROM on the sky, over the ringdown window
#   qnm_convergence.gif   eigenvalues of the learned Â vs the Kerr QNM as the training window slides
#   test_particles.gif    a ring of free test particles stretched by the wave, seen face-on
#   chirp.wav, chirp.mp4  the face-on h₊ waveform as sound, for a total mass of 65 solar masses
#
# Time is in units of the total mass M (G = c = 1): 1 M = 4.93 μs × (M / M_sun).
# Animations that cover the whole signal use frames evenly spaced in the phase of h22, not in
# time, so the slow inspiral passes quickly and the merger plays in slow motion.
#
# Run:  julia einstein/animate_sxs.jl     (after fetch_sxs.py; ≈ 2–3 min)

include("opinf_sxs.jl")       # load_sxs, OpInf pieces, kerr_qnm_220; its main block does not run here
include("waveform_tools.jl")   # spin-weighted harmonics, frames_by_phase, interpolate_linear

const SECONDS_PER_SOLAR_MASS = 4.925490947e-6   # G M_sun / c³
const CHIRP_TOTAL_MASS = 65.0                   # in solar masses, close to GW150914
const AUDIO_SAMPLE_RATE = 44_100

# ---------------------------------------------------------------- helpers
"""
Mollweide (equal-area) map of the sky, n_rows × 2n_rows pixels. Returns the polar angle θ and
azimuth φ that each pixel inside the ellipse shows, and a mask of those pixels.
"""
function mollweide_grid(n_rows)
    n_cols = 2n_rows
    xs = ((1:n_cols) .- 0.5) ./ n_cols .* 4√2 .- 2√2      # pixel centres
    ys = ((1:n_rows) .- 0.5) ./ n_rows .* 2√2 .- √2
    θ = zeros(n_rows, n_cols)
    φ = zeros(n_rows, n_cols)
    inside = falses(n_rows, n_cols)
    for (col, x) in enumerate(xs), (row, y) in enumerate(ys)
        if (x / 2√2)^2 + (y / √2)^2 < 1
            auxiliary_angle = asin(y / √2)
            latitude = asin((2auxiliary_angle + sin(2auxiliary_angle)) / π)
            θ[row, col] = π / 2 - latitude
            φ[row, col] = π * x / (2√2 * cos(auxiliary_angle))
            inside[row, col] = true
        end
    end
    return (; xs, ys, θ, φ, inside)
end

"Values on the pixels inside the map, placed on the full grid with NaN (blank) outside."
function on_map(grid, values_inside)
    image = fill(NaN, size(grid.inside))
    image[grid.inside] = values_inside
    return image
end

"h₊ and h× for an observer looking down the orbital axis (θ = 0): only the m = 2 modes contribute."
function face_on_polarizations(h_modes, ell, m)
    h = vec(sum(spin_weighted_ylm(-2, ell[j], m[j], 0.0, 0.0) .* h_modes[j:j, :]
                for j in eachindex(ell) if m[j] == 2))
    return real.(h), -imag.(h)          # h = h₊ − i h×
end

"Write mono 16-bit PCM WAV."
function write_wav(path, samples, sample_rate)
    pcm = round.(Int16, clamp.(samples, -1, 1) .* typemax(Int16))
    n_bytes = 2length(pcm)
    open(path, "w") do io
        write(io, "RIFF", UInt32(36 + n_bytes), "WAVE")
        write(io, "fmt ", UInt32(16), UInt16(1), UInt16(1), UInt32(sample_rate),
              UInt32(2sample_rate), UInt16(2), UInt16(16))
        write(io, "data", UInt32(n_bytes), pcm)
    end
end

function save_gif(animation, filename; fps)
    gif(animation, joinpath(OUTDIR, filename); fps)
    println("  → $filename")
end

# ---------------------------------------------------------------- data
t, X, dXdt, ell, m = load_sxs()
h_modes = complex_modes(X)
row_h22 = findfirst((ell .== 2) .& (m .== 2))
h22 = h_modes[row_h22, :]
t_peak = t[argmax(abs.(h22))]
qnm = kerr_qnm_220()
println("Loaded $(length(ell)) modes, t ∈ [$(t[1]), $(t[end])] M, peak of |h22| at t = $t_peak M")

sky = mollweide_grid(110)
Y_sky = harmonics_matrix(sky.θ[sky.inside], sky.φ[sky.inside], ell, m)

# ---------------------------------------------------------------- 1. strain on the sky
println("1/5 sky_strain.gif")
sky_frames = frames_by_phase(t, h22, t[1], t_peak + 60, 0.6)
h_plus_sky = real.(Y_sky * h_modes[:, sky_frames])
color_limit = maximum(abs, h_plus_sky)

animation = @animate for (frame, time_idx) in enumerate(sky_frames)
    map_panel = heatmap(sky.xs, sky.ys, on_map(sky, h_plus_sky[:, frame]);
                        c = :RdBu, clims = (-color_limit, color_limit), aspect_ratio = :equal,
                        axis = false, grid = false, colorbar_title = "h₊ (r/M)",
                        title = @sprintf("Gravitational-wave strain on the sky,  t − t_peak = %7.1f M", t[time_idx] - t_peak))
    waveform_panel = plot(t .- t_peak, real.(h22); c = :gray, lw = 1, label = "Re h₂₂",
                          xlabel = "t − t_peak (M)", xlims = (t[1] - t_peak, 60), legend = :topleft)
    vline!(waveform_panel, [t[time_idx] - t_peak]; c = :red, lw = 2, label = "")
    plot(map_panel, waveform_panel; layout = grid(2, 1, heights = [0.72, 0.28]), size = (900, 650))
end
save_gif(animation, "sky_strain.gif"; fps = 20)

# ---------------------------------------------------------------- 2. true vs ROM on the sky (ringdown)
println("2/5 ringdown_sky_rom.gif (fitting OpInf ROMs; this prints their table)")
ringdown_train   = findall(t_peak + 20 .<= t .<= t_peak + 50)
ringdown_predict = findall(t_peak + 50.5 .<= t .<= t_peak + 80)
ringdown_results = run_experiment("Ringdown", t, X, dXdt, ringdown_train, ringdown_predict, 2:2:20, false)
best_rom = ringdown_results[argmin([result.predict_err for result in ringdown_results])]
window_idx = ringdown_train[1]:ringdown_predict[end]
h_true_sky = real.(Y_sky * h_modes[:, window_idx])
h_rom_sky  = real.(Y_sky * complex_modes(best_rom.X_rom))
error_per_time = relative_error_per_time(best_rom.X_rom, X[:, window_idx])
t_end_of_training = t[ringdown_train[end]] - t_peak

animation = @animate for frame in eachindex(window_idx)
    scale = maximum(abs, h_true_sky[:, frame])            # signal decays ~e^(−0.09 t): rescale each frame
    sky_panel(values, title) = heatmap(sky.xs, sky.ys, on_map(sky, values ./ scale);
                                       c = :RdBu, clims = (-1, 1), aspect_ratio = :equal,
                                       axis = false, grid = false, colorbar = false, title)
    phase_label = t[window_idx[frame]] - t_peak <= t_end_of_training ? "training" : "prediction"
    true_panel = sky_panel(h_true_sky[:, frame], "True (SXS)")
    rom_panel  = sky_panel(h_rom_sky[:, frame], "OpInf linear ROM, r = $(best_rom.r)")
    error_panel = plot(t[window_idx] .- t_peak, error_per_time; yscale = :log10, c = :black, label = "",
                       xlabel = "t − t_peak (M)", ylabel = "relative error",
                       title = @sprintf("t − t_peak = %.1f M  (%s; colour scale normalised per frame)",
                                        t[window_idx[frame]] - t_peak, phase_label))
    vline!(error_panel, [t_end_of_training]; c = :gray, ls = :dash, label = "end of training")
    scatter!(error_panel, [t[window_idx[frame]] - t_peak], [error_per_time[frame]]; c = :red, markersize = 6, label = "")
    plot(true_panel, rom_panel, error_panel; layout = @layout([a b; c]), size = (1000, 620))
end
save_gif(animation, "ringdown_sky_rom.gif"; fps = 15)

# ---------------------------------------------------------------- 3. QNM convergence
println("3/5 qnm_convergence.gif")
qnm_r = 16
start_offsets = 0:1:30
learned_spectra = map(start_offsets) do offset
    train_idx = findall(t_peak + offset .<= t .<= t_peak + offset + 30)
    V = svd(X[:, train_idx]).U[:, 1:qnm_r]
    λ = choose_lambda(V, X, dXdt, t, train_idx, qnm_r, false, LAMBDA_GRID)
    eigvals(fit_operators(V, X, dXdt, train_idx, false, λ)[:, 1:qnm_r])
end
closest_to_qnm = [spectrum[argmin(abs.(spectrum .- qnm))] for spectrum in learned_spectra]
relative_diff = abs.(closest_to_qnm .- qnm) ./ abs(qnm)

animation = @animate for (window_number, offset) in enumerate(start_offsets)
    spectrum = learned_spectra[window_number]
    closest = closest_to_qnm[window_number]
    plane = scatter(real.(spectrum), imag.(spectrum); c = :steelblue, markersize = 5, label = "eigenvalues of learned Â (r = $qnm_r)",
                    xlims = (-0.5, 0.1), ylims = (-1.3, 1.3), xlabel = "Re (decay rate, 1/M)", ylabel = "Im (frequency, 1/M)",
                    title = "Training window: t_peak + [$offset, $(offset + 30)] M", legend = :bottomleft)
    scatter!(plane, [real(qnm), real(qnm)], [imag(qnm), -imag(qnm)]; marker = :star5, markersize = 12, c = :red,
             label = "Kerr (2,2,0) quasinormal mode")
    scatter!(plane, [real(closest)], [imag(closest)]; marker = :circle, markersize = 9,
             markercolor = :transparent, markerstrokecolor = :orange, markerstrokewidth = 2.5,
             label = "closest eigenvalue")
    convergence = plot(start_offsets[1:window_number], relative_diff[1:window_number];
                       marker = :circle, c = :orange, yscale = :log10, label = "",
                       xlims = (-1, 31), ylims = (0.5minimum(relative_diff), 2maximum(relative_diff)),
                       xlabel = "training start − t_peak (M)", ylabel = "|λ − ω_QNM| / |ω_QNM|",
                       title = @sprintf("relative difference %.1f %%", 100relative_diff[window_number]))
    plot(plane, convergence; layout = (1, 2), size = (1100, 480), margin = 5Plots.mm)
end
save_gif(animation, "qnm_convergence.gif"; fps = 4)

# ---------------------------------------------------------------- 4. ring of test particles
println("4/5 test_particles.gif")
h_plus, h_cross = face_on_polarizations(h_modes, ell, m)
particle_frames = frames_by_phase(t, h22, t[1], t_peak + 60, 0.35)
n_particles = 24
angles = range(0, 2π, length = n_particles + 1)[1:end-1]
ring_x, ring_y = cos.(angles), sin.(angles)
exaggeration = 0.5 / maximum(hypot.(h_plus, h_cross))   # largest stretch = 25 % of the radius

animation = @animate for time_idx in particle_frames
    # displacement of a free particle at (x, y) in the transverse-traceless gauge
    stretch_plus, stretch_cross = exaggeration .* (h_plus[time_idx], h_cross[time_idx])
    x = ring_x .+ 0.5 .* (stretch_plus .* ring_x .+ stretch_cross .* ring_y)
    y = ring_y .+ 0.5 .* (stretch_cross .* ring_x .- stretch_plus .* ring_y)
    ring = scatter(x, y; markersize = 7, c = :black, label = "", aspect_ratio = :equal, xlims = (-1.5, 1.5),
                   ylims = (-1.5, 1.5), axis = false, grid = false,
                   title = @sprintf("Free particles, viewed along the orbital axis\nt − t_peak = %7.1f M   (strain exaggerated)",
                                    t[time_idx] - t_peak))
    plot!(ring, cos.(range(0, 2π, 100)), sin.(range(0, 2π, 100)); c = :lightgray, ls = :dash, label = "")
    waveform = plot(t .- t_peak, h_plus; c = :gray, label = "h₊", xlabel = "t − t_peak (M)",
                    xlims = (t[1] - t_peak, 60), legend = :topleft)
    vline!(waveform, [t[time_idx] - t_peak]; c = :red, lw = 2, label = "")
    plot(ring, waveform; layout = grid(2, 1, heights = [0.75, 0.25]), size = (600, 750))
end
save_gif(animation, "test_particles.gif"; fps = 30)

# ---------------------------------------------------------------- 5. chirp as sound
println("5/5 chirp.wav, chirp.mp4  (total mass $(CHIRP_TOTAL_MASS) M_sun)")
seconds_per_M = CHIRP_TOTAL_MASS * SECONDS_PER_SOLAR_MASS
signal_seconds = (t .- t[1]) .* seconds_per_M          # audio clock: 0 at the first data point
signal_duration = signal_seconds[end]
padding = 0.5                                          # seconds of silence before and after
fade_in_duration = 0.1                                 # ramp up the start to avoid a click
audio_times = range(-padding, signal_duration + padding, step = 1 / AUDIO_SAMPLE_RATE)

during_signal = 0 .<= audio_times .<= signal_duration
audio = during_signal .* interpolate_linear(signal_seconds, h_plus, clamp.(audio_times, 0, signal_duration))
audio .*= clamp.(audio_times ./ fade_in_duration, 0, 1)
audio ./= 1.1maximum(abs, audio)                       # peak at 1/1.1 of full scale
write_wav(joinpath(OUTDIR, "chirp.wav"), audio, AUDIO_SAMPLE_RATE)
println("  → chirp.wav ($(round(audio_times[end] - audio_times[1], digits = 2)) s)")

video_fps = 30
video_times = range(audio_times[1], audio_times[end], step = 1 / video_fps)
seconds_from_peak = signal_seconds .- signal_duration .+ (t[end] - t_peak) * seconds_per_M
peak_on_audio_clock = (t_peak - t[1]) * seconds_per_M
animation = @animate for audio_time in video_times
    plt = plot(seconds_from_peak, h_plus;
               c = :steelblue, label = "", xlabel = "time to peak (s)", ylabel = "h₊ (face-on)",
               title = "SXS:BBH:0305 as sound: total mass $(Int(CHIRP_TOTAL_MASS)) M_sun, played in real time",
               size = (900, 400), margin = 5Plots.mm)
    vline!(plt, [audio_time - peak_on_audio_clock]; c = :red, lw = 2, label = "")
end
silent_video = joinpath(OUTDIR, "chirp_silent.mp4")
mp4(animation, silent_video; fps = video_fps)
run(pipeline(`ffmpeg -y -loglevel error -i $silent_video -i $(joinpath(OUTDIR, "chirp.wav"))
               -c:v copy -c:a aac -shortest $(joinpath(OUTDIR, "chirp.mp4"))`))
rm(silent_video)
println("  → chirp.mp4")
println("Done. Files are in $OUTDIR")
