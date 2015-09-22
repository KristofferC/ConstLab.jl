"""
Constitutive driver.
"""
function driver(matstat::MatStatus,
                matpar::MatParameter,
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


    i = 1
    t_prev = time_history[1]
    for (i, t) in enumerate(time_history[2:end])
        i += 1
        dt = t - t_prev
        ɛ = strain_history[:, i]
        σ = stress_history[:, i]

        # Guess total strain
        ɛ_iter = copy(ɛ)

        # Full strain control -> correct strain is just the one from strain_history
        if sum(sc) == 0
            ɛ_sol = ɛ[sc]
        else
            function fg!(x, fx, gx)
                ɛ[sc] = x
                σ_res, ATS, _ = stress(ɛ, dt, mp, ms)
                copy!(fx, σ_res[sc] - σ[sc])
                copy!(gx, ATS[sc,sc])
            end

            res = nlsolve(only_fg!(fg!), ɛ_sol; xtol=xtol, ftol=ftol, iterations=iterations,
                                                method=method)
            if !converged(res)
                if err_on_nonconv
                    throw(NonConvergenceError())
                elseif warn_on_nonconv
                    warn(NON_CONV_MESSAGE)
                end
            end
            ɛ_sol = res.zero
        end

        ɛ[sc] = ɛ_sol

        # Update σ and material status at converged point
        σ, ATS, ms = stress(ɛ, dt, mp, ms)

        push!(matstats, ms)
        σs[:, i] = σ
        εs[:, i] = ɛ

        t_prev = t
    end # time

    return εs, σs, matstats
end
