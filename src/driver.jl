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
                err_on_nonconv::Bool = true,
                warn_on_nonconv::Bool = true,
                xtol::Real = 0.0,
                ftol::Real = 1e-5,
                iterations::Integer = 100,
                method::Symbol = :trust_region)

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
    εs = zeros(ɛ_dim, nt)
    σs = zeros(ɛ_dim, nt)
    ɛ_sol = zeros(sum(sc))
    ms_grad = deepcopy(ms)

    i = 1
    t_prev = time_history[1]
    ɛ_sol = zeros(sum(sc))
    ∆εₙ₋₁ = zeros(sum(sc))
    ∆ε₀ = zeros(sum(sc))
    ∆ε = zeros(sum(sc))
    for (i, t) in enumerate(time_history[2:end])
        i += 1
        dt = t - t_prev
        ɛ = strain_history[:, i]
        σ = stress_history[:, i]
        # Guess total strain
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

            res = nlsolve(not_in_place(f, g), ∆ε₀; xtol=xtol, ftol=ftol, iterations=iterations,
                                                method=method, show_trace=true)
            if !converged(res)
                if err_on_nonconv
                    throw(NonConvergenceError())
                elseif warn_on_nonconv
                    warn(NON_CONV_MESSAGE)
                end
            end
            copy!(∆ε, res.zero)
        end

        copy!(∆εₙ₋₁, ∆ε)

        ɛ[sc] += ∆ε

        # Update σ and material status at converged point
        σ, ms = stress(ɛ, dt, mp, ms)
        push!(matstats, ms)

        σs[:, i] = σ
        εs[:, i] = ɛ

        t_prev = t
    end # time

    return εs, σs, matstats
end
