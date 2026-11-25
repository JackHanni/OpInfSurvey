# Helpers for turning SXS strain modes h_ℓm(t) into strain at a point on the sky.
# Used by animate_sxs.jl and animate_merger_3d.jl.

"Wigner small-d matrix element d^ℓ_{m′m}(β), from the explicit sum formula."
function wigner_d(ell, m_prime, m, β)
    fact(n) = Float64(factorial(n))
    prefactor = sqrt(fact(ell + m) * fact(ell - m) * fact(ell + m_prime) * fact(ell - m_prime))
    total = 0.0
    for k in max(0, m - m_prime):min(ell + m, ell - m_prime)
        denominator = fact(ell + m - k) * fact(k) * fact(m_prime - m + k) * fact(ell - m_prime - k)
        total += (-1)^(m_prime - m + k) / denominator *
                 cos(β / 2)^(2ell + m - m_prime - 2k) * sin(β / 2)^(m_prime - m + 2k)
    end
    return prefactor * total
end

"Spin-weighted spherical harmonic ₛY_ℓm(θ, φ) = (−1)^s √((2ℓ+1)/4π) d^ℓ_{m,−s}(θ) e^{imφ}."
spin_weighted_ylm(s, ell, m, θ, φ) = (-1)^s * sqrt((2ell + 1) / 4π) * wigner_d(ell, m, -s, θ) * cis(m * φ)

"Matrix whose (point, mode) entry is ₋₂Y_ℓm at that point, so strain at the points = Y * h_modes."
harmonics_matrix(θs, φs, ell, m) = [spin_weighted_ylm(-2, ell[j], m[j], θs[i], φs[i])
                                    for i in eachindex(θs), j in eachindex(ell)]

"Complex mode amplitudes h_ℓm(t), one mode per row, from the [Re; Im] state matrix."
function complex_modes(X)
    n_modes = size(X, 1) ÷ 2
    return X[1:n_modes, :] .+ im .* X[n_modes+1:end, :]
end

"""
Indices of `t` between `t_start` and `t_end`, spaced so the phase of h22 advances by about
`phase_step` radians from one frame to the next.
"""
function frames_by_phase(t, h22, t_start, t_end, phase_step)
    phase_jumps = abs.(diff(angle.(h22)))
    phase_jumps = min.(phase_jumps, 2π .- phase_jumps)     # unwrap
    phase = [0.0; cumsum(phase_jumps)]
    candidates = findall(t_start .<= t .<= t_end)
    frames = [candidates[1]]
    for idx in candidates
        if phase[idx] - phase[frames[end]] >= phase_step
            push!(frames, idx)
        end
    end
    return frames
end

"Linear interpolation of samples `values` at sorted `times` onto `new_times`."
function interpolate_linear(times, values, new_times)
    return map(new_times) do new_time
        left = clamp(searchsortedlast(times, new_time), 1, length(times) - 1)   # times[left] ≤ new_time
        weight = (new_time - times[left]) / (times[left+1] - times[left])
        (1 - weight) * values[left] + weight * values[left+1]
    end
end
