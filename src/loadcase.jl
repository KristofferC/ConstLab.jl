function loadcase(case::Symbol, ε_max::Number, ts)

    εpt = ε_max / maximum(ts)

    if case == :uniaxial_strain
        ε_control = map(Bool, [1, 1, 1, 1, 1, 1])
        εs = [εpt, 0.0, 0.0, 0.0, 0.0, 0.0] * ts'

    elseif case == :uniaxial_stress
        ε_control = map(Bool, [1, 0, 0, 1, 1, 1])
        εs = [εpt, 0.0, 0.0, 0.0, 0.0, 0.0] * ts'

    elseif case == :biaxial_strain_plstrain
        ε_control =  map(Bool, [1, 1, 1, 1, 1, 1])
        εs = [εpt, εpt, 0.0, 0.0, 0.0, 0.0] * ts'

    elseif case == :biaxial_strain_plstress
        ε_control = map(Bool, [1, 1, 0, 1, 1, 1])
        εs = [εpt, εpt, 0.0, 0.0, 0.0, 0.0] * ts'

    elseif case == :simpleshear
        ε_control = map(Bool, [1, 1, 1, 1, 1, 1])
        εs = [0.0, 0.0, 0.0, 0.0, 0.0, εpt] * ts'
    else
        throw(ArgumentError("invalid case"))
    end
    σs = zeros(6) * ts'
    return εs, σs, ε_control
end
