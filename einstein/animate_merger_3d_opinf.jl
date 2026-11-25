# Side-by-side 3D animation: the true SXS:BBH:0305 wave (left) and what OpInf predicts (right).
#
# No single OpInf model covers the whole signal, so the right panel switches models, using the
# same two experiments as opinf_sxs.jl:
#   until t_peak + 20 M:  quadratic ROM (r = 6) trained on the inspiral, t ∈ [3300, 3500] M,
#                         then run on through merger, which it cannot reproduce;
#   from  t_peak + 20 M:  linear ROM (r = 20) trained on the ringdown, t_peak + [20, 50] M,
#                         started from the true state at t_peak + 20 and run to t_peak + 80.
# Before 3300 M (where the first ROM starts) the right panel shows the true data.
# OpInf models only the wave, so both panels draw the simulation's black holes for reference.
# See animate_merger_3d.jl for how the wave surface is drawn.
#
# Writes einstein/output/merger_3d_opinf.mp4.
# Run:  julia einstein/animate_merger_3d_opinf.jl [number of worker processes, default 6]

include("animate_merger_3d.jl")   # wave surface, black holes, draw_frame, crop settings

const INSPIRAL_TRAIN = (3300.0, 3500.0)
const INSPIRAL_R = 6
const RINGDOWN_TRAIN_AFTER_PEAK = (20.0, 50.0)
const RINGDOWN_R = 20
const COMPARE_UNTIL_AFTER_PEAK = 80.0

time_window(t_start, t_end) = findall(t_start .<= t .<= t_end)

"""
Fit an OpInf ROM on `train_idx` (λ chosen from the training window only, as in opinf_sxs.jl)
and integrate it from the first training snapshot to `t_stop`.
Returns the time indices covered and the ROM's full-state prediction there.
"""
function rom_prediction(train_idx, r, quadratic, t_stop)
    V = svd(X[:, train_idx]).U[:, 1:r]
    λ = choose_lambda(V, X, dXdt, t, train_idx, r, quadratic, LAMBDA_GRID)
    O = fit_operators(V, X, dXdt, train_idx, quadratic, λ)
    covered = train_idx[1]:findlast(t .<= t_stop)
    max_norm = 10 * maximum(norm.(eachcol(X)))
    X_rom = V * integrate(rom_rhs(O, r, quadratic), V' * X[:, train_idx[1]], t[covered]; max_norm)
    return covered, X_rom
end

sxs_data = load_sxs()
X, dXdt = sxs_data.X, sxs_data.dXdt
t_switch = t_peak + RINGDOWN_TRAIN_AFTER_PEAK[1]
t_compare_end = t_peak + COMPARE_UNTIL_AFTER_PEAK

inspiral_train = time_window(INSPIRAL_TRAIN...)
ringdown_train = time_window((t_peak .+ RINGDOWN_TRAIN_AFTER_PEAK)...)
inspiral_covered, inspiral_X = rom_prediction(inspiral_train, INSPIRAL_R, true, t_switch)
ringdown_covered, ringdown_X = rom_prediction(ringdown_train, RINGDOWN_R, false, t_compare_end)

# OpInf's version of the whole history: true data before the first ROM starts, then each ROM
# over its own time span. Same layout as X.
X_opinf = copy(X)
X_opinf[:, inspiral_covered] = inspiral_X
X_opinf[:, ringdown_covered] = ringdown_X
opinf_modes = complex_modes(X_opinf)

comparison_frames = frames_by_phase(t, h22, INSPIRAL_TRAIN[1], t_compare_end, PHASE_STEP)
comparison_clocks = t[comparison_frames]

"Which OpInf model is active at time τ, and whether τ is in its training window."
function opinf_status(τ)
    if τ < t_switch
        model = "inspiral model (quadratic, r = $INSPIRAL_R)"
        in_training = τ <= INSPIRAL_TRAIN[2]
    else
        model = "ringdown model (linear, r = $RINGDOWN_R)"
        in_training = τ <= t_peak + RINGDOWN_TRAIN_AFTER_PEAK[2]
    end
    return model, in_training ? "training window" : "prediction"
end

"Relative error of the OpInf state against the truth at time index k."
state_error(k) = norm(X_opinf[:, k] - X[:, k]) / norm(X[:, k])

"Crop both panels, label them, put them side by side and add the caption, with ffmpeg."
function combine_panels(true_png, opinf_png, out_png, caption)
    caption_file = out_png * ".txt"
    write(caption_file, caption)
    crop = "crop=$(CROP.w):$(CROP.h):$(CROP.x):$(CROP.y)"
    label(text) = "drawtext=text='$text':fontcolor=white:fontsize=26:x=(w-text_w)/2:y=h-50"
    filter = "[0]$crop,$(label("True (SXS simulation)"))[left];" *
             "[1]$crop,$(label("OpInf"))[right];" *
             "[left][right]hstack," *
             "drawtext=textfile=$(caption_file):fontcolor=white:fontsize=22:line_spacing=8:x=20:y=16:expansion=none"
    run(`ffmpeg -y -v error -i $true_png -i $opinf_png -filter_complex $filter $out_png`)
end

"Render comparison frames worker_id, worker_id + n_workers, ... into `frame_dir`."
function render_comparison_frames(worker_id, n_workers, frame_dir)
    true_png = joinpath(frame_dir, "true_worker$(worker_id).png")
    opinf_png = joinpath(frame_dir, "opinf_worker$(worker_id).png")
    n_frames = length(comparison_clocks)
    for frame_number in worker_id:n_workers:n_frames
        k = comparison_frames[frame_number]
        clock = t[k]
        savefig(draw_frame(clock, frame_number, n_frames), true_png)
        savefig(draw_frame(clock, frame_number, n_frames; modes = opinf_modes), opinf_png)
        model, window = opinf_status(clock)
        caption = @sprintf("t - t_peak = %+6.1f M    %s\n", clock - t_peak, stage_at(clock)) *
                  "OpInf: $model, $window,  error at the binary " *
                  @sprintf("%.0f %%\n", 100state_error(k)) *
                  "OpInf predicts only the wave; black holes are from the simulation in both panels"
        combine_panels(true_png, opinf_png, joinpath(frame_dir, @sprintf("frame_%04d.png", frame_number)), caption)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) == 4 && ARGS[1] == "--worker"
        render_comparison_frames(parse(Int, ARGS[2]), parse(Int, ARGS[3]), ARGS[4])
    else
        n_workers = isempty(ARGS) ? 6 : parse(Int, ARGS[1])
        println("Rendering $(length(comparison_clocks)) comparison frames with $n_workers processes, ",
                "t − t_peak ∈ [$(comparison_clocks[1] - t_peak), $(comparison_clocks[end] - t_peak)] M; ",
                "OpInf switches models at t_peak + $(RINGDOWN_TRAIN_AFTER_PEAK[1]) M")
        frame_dir = mktempdir()
        julia = Base.julia_cmd()
        workers = [run(`$julia $(@__FILE__) --worker $id $n_workers $frame_dir`; wait = false) for id in 1:n_workers]
        foreach(wait, workers)
        video = joinpath(OUTDIR, "merger_3d_opinf.mp4")
        run(`ffmpeg -y -v error -framerate $VIDEO_FPS -i $(joinpath(frame_dir, "frame_%04d.png"))
                 -c:v libx264 -pix_fmt yuv420p -crf 20 $video`)
        rm(frame_dir; recursive = true)
        println("  → merger_3d_opinf.mp4 ($(round(length(comparison_clocks) / VIDEO_FPS, digits = 1)) s at $VIDEO_FPS fps)")
    end
end
