"""
Constitutive driver.
"""
function driver(stress,
                ats,
                matstat,
                matpar,
                time_history::Vector,
                strain_history::Matrix,
                stress_history::Matrix,
                strain_control::Vector{Bool};
                xtol::Real = 0.0,
                ftol::Real = 1e-5,
                iterations::Integer = 100,
                method::Symbol = :trust_region,
                show_trace::Bool = false)

    ɛ_dim, nɛ = size(strain_history)
    σ_dim, nσ = size(stress_history)
    nt = length(time_history)

    nɛ == nt || throw(ArgumentError("number of strains must be equal to number of time steps"))
    nσ == nt || throw(ArgumentError("number of stesses must be equal to number of time steps"))
    ɛ_dim == σ_dim || throw(ArgumentError("stresses and strains must have equal number of components"))

    mp = matpar
    ms = matstat
    # Stress control indices
    sc = !strain_control

    matstats = [ms]
    εs = strain_history[:,1]
    σs = stress_history[:,1]
    ɛ_sol = zeros(sum(sc))
    ms_grad = deepcopy(ms)

    i = 1
    t_prev = time_history[1]
    ɛ_sol = zeros(sum(sc))
    ∆εₙ₋₁ = zeros(sum(sc))
    ∆ε₀ = zeros(sum(sc))
    ∆ε = zeros(sum(sc))
    for t in time_history[2:end]
        i += 1
        dt = t - t_prev
        # Guess total strain
        ɛ = strain_history[:, i]
        σ = stress_history[:, i]
        ms_new = deepcopy(ms)
        if !(sum(sc) == 0)
            copy!(∆ε₀, ∆εₙ₋₁)

            function f(∆ε_red)
                dε = zeros(ɛ_dim)
                dε[sc] = ∆ε_red
                σ_res, ms_new = stress(ɛ + dε, dt, mp, ms)
                σ_res[sc] - σ[sc]
            end

            function g(∆ε_red)
                dε = zeros(ɛ_dim)
                dε[sc] = ∆ε_red
                grad = ats(ɛ + dε, dt, mp, ms_new)
                grad[sc, sc]
            end
            try
                res = nlsolve(not_in_place(f, g), ∆ε₀; xtol=xtol, ftol=ftol,
                              iterations=iterations, method=method, show_trace=show_trace)
                if !converged(res)
                    i -= 1
                    warn(NON_CONV_MESSAGE)
                    @goto ret
                end
                copy!(∆ε, res.zero)
            catch e
                isa(e, MaterialNonConvergenceError) || rethrow(e)
                i -= 1
                warn(MAT_NON_CONV_MESSAGE)
                @goto ret
            end
        end

        copy!(∆εₙ₋₁, ∆ε)

        ɛ[sc] += ∆ε

        # Update σ and material status at converged point
        try
            σ, ms = stress(ɛ, dt, mp, ms)
            push!(matstats, ms)
            append!(σs, σ)
            append!(εs, ɛ)
        catch e
            println(e)
            isa(e, MaterialNonConvergenceError) || rethrow(e)
            i -= 1
            warn(MAT_NON_CONV_MESSAGE)
            @goto ret
        end

        t_prev = t
    end # time

    @label ret
    ɛs_mat = reshape(εs, (ɛ_dim, i))
    σs_mat = reshape(σs, (σ_dim, i))
    return ɛs_mat, σs_mat, matstats
end
