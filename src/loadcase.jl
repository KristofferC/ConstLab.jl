function loadcase(case::Symbol, ε_max::Number, ts, voigt_struct::Symbol = :sym)

    if voigt_struct == :unsym
        voigt_size = 9
        off_diags = 6
    elseif voigt_struct == :sym
        voigt_size = 6
        off_diags = 3
    else
        throw(ArgumentError("Invalid voigt structure, :sym or :unsym valid"))
    end

    εpt = ε_max / maximum(ts)

    if case == :uniaxial_strain
        ε_control = ones(Bool, voigt_size)
        εs = [εpt; zeros(voigt_size-1)] * ts'

    elseif case == :uniaxial_stress
        ε_control = [true; false; false; ones(Bool, off_diags)]
        εs = [εpt; zeros(voigt_size-1)] * ts'

    elseif case == :biaxial_strain_plstrain
        ε_control = ones(Bool, voigt_size)
        εs = [εpt; εpt; zeros(voigt_size-2)] * ts'

    elseif case == :biaxial_strain_plstress
        ε_control = [true; true; false; ones(Bool, off_diags)]
        εs = [εpt; εpt; zeros(voigt_size-2)] * ts'

    elseif case == :simpleshear
        ε_control = ones(Bool, voigt_size)
        if voigt_struct == :unsym
            εs = [0.0, 0.0, 0.0, εpt, 0.0, 0.0, 0.0, 0.0, εpt] * ts'
        else
            εs = [0.0, 0.0, 0.0, εpt, 0.0, 0.0] * ts'
        end
    else
        throw(ArgumentError("invalid case"))
    end

    size(εs, 1) == voigt_size || throw(ErrorException("invalid creation of strains, bug!"))
    length(ε_control) == voigt_size || throw(ErrorException("invalid creation of strains, bug!"))

    σs = zeros(voigt_size) * ts'
    return εs, σs, ε_control
end
