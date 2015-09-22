const ValidMods = Union{Function, NTuple{2, Int}, Number, Void}
const ValidVars = Union{Symbol, Vector, Matrix}

function plot{T <: MatStatus}(x::ValidVars, y::ValidVars, mss::Vector{T};
                               x_mod::ValidMods = nothing, y_mod::ValidMods = nothing)
    plot!(plot(), x, y, mss; x_mod=x_mod, y_mod=y_mod)
end

function plot!{T <: MatStatus}(p::Plots.Plot, x::ValidVars, y::ValidVars, mss::Vector{T};
                               x_mod::ValidMods = nothing, y_mod::ValidMods = nothing)

    length(mss) != 0 || throw(ArgumentError("length of mss must be > 0"))

    for (z, z_str) in ((x, "x"),
                       (y, "y"))
        if isa(z, Vector)
            length(z) == length(mss) || throw(ArgumentError("Length of $z_str must be equal to length of mss"))
        end
    end

    xs = zeros(length(mss))
    ys = zeros(length(mss))

    for (i, ms) in enumerate(mss)
        for (res_vec, val, mod) in ((xs, x, x_mod),
                                    (ys, y, y_mod))
            if isa(val, Symbol)
                res_vec[i] = get_value(ms.(val), mod)
            elseif isa(x, Vector)
                res_vec[i] = get_value(val[i], mod)
            elseif isa(x, Matrix)
                res_vec[i] = get_value(val[:,i], mod)
            end
        end
    end

    p = Plots.plot!(p, xs, ys)

    return p
end

get_value(q::Number, ::Function) = f(q)::Number # Ambiguity fix
get_value(q, f::Function) = f(q)::Number
get_value(q::Vector, i::Number) = q[i]::Number
get_value(q::Matrix, ij::NTuple{2, Int}) = q[ij[1], ij[2]]::Number
get_value(q::Number, ::Any) = q


