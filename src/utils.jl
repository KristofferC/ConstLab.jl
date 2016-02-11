function create_component_macros{N}(name::ASCIIString, sizes::NTuple{N, Int})
    unpack_name = symbol("unpack_" * name)
    unpack_name! = symbol("unpack_" * name * "!")
    pack_name = symbol("pack_" * name)
    @eval begin
        macro $unpack_name(ex)
            length($sizes) == length(ex.args[1].args) || throw(ArgumentError("Invalid number of output vars"))
            sizes_loc = $sizes
            exvec = quote end
            x_val = gensym("x")
            push!(exvec.args, :($x_val = $(ex.args[end])))
            length_cum = 1
            for (i, sym) in enumerate(ex.args[1].args)
                assert(typeof(sym) == Symbol)
                if $(sizes)[i] == 1
                    push!(exvec.args, :($(sym) = $(x_val)[$(length_cum)]))
                else
                    push!(exvec.args, :($(sym) = $(x_val)[$length_cum : $(length_cum + sizes_loc[i]-1)]))
                end
                length_cum += $(sizes)[i]
            end
            return esc(exvec)
        end

         macro $unpack_name!(ex)
            length($sizes) == length(ex.args[1].args) || throw(ArgumentError("Invalid number of output vars"))
            sizes_loc = $sizes
            exvec = quote end
            x_val = gensym("x")
            push!(exvec.args, :($x_val = $(ex.args[end])))
            length_cum = 1
            for (i, sym) in enumerate(ex.args[1].args)
                assert(typeof(sym) == Symbol)
                if $(sizes)[i] == 1
                    push!(exvec.args, :($(sym) = $(x_val)[$(length_cum)]))
                else
                    push!(exvec.args, :(@devec $(sym)[:] = $(x_val)[$length_cum : $(length_cum + sizes_loc[i]-1)]))
                end
                length_cum += $(sizes)[i]
            end
            return esc(exvec)
        end

        macro $pack_name(ex)
            length($sizes) == length(ex.args[2].args) || throw(ArgumentError("Invalid number of output vars"))
            sizes_loc = $sizes
            exvec = quote end
            length_cum = 1
            x = ex.args[1]
            for (i, sym) in enumerate(ex.args[2].args)
                assert(typeof(sym) == Symbol)
                if $(sizes)[i] == 1
                    push!(exvec.args, :($(x)[$(length_cum)] = $(sym)))
                else
                    push!(exvec.args, :($(x)[$length_cum : $(length_cum + sizes_loc[i]-1)] = $(sym)))
                end
                length_cum += $(sizes)[i]
            end
            return esc(exvec)
        end

    end
end
