module ConstLab

import Base.show

using NLsolve

export loadcase, driver, MatStatus, MatParameter, create_component_macro

const NON_CONV_MESSAGE = "constitutive iterations did not converge"

type NonConvergenceError <: Exception end
show(io::IO, nce::NonConvergenceError) = print(io, NON_CONV_MESSAGE)

abstract MatStatus
abstract MatParameter
abstract MatModel

include("driver.jl")
include("loadcase.jl")
include("utils.jl")

end # module
