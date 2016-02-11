module ConstLab

import Base.show

using NLsolve

export loadcase, driver, MatStatus, MatParameter, create_component_macros

const NON_CONV_MESSAGE = "constitutive iterations did not converge"

type NonConvergenceError <: Exception end
show(io::IO, nce::NonConvergenceError) = print(io, NON_CONV_MESSAGE)

const MAT_NON_CONV_MESSAGE = "material iterations did not converge"

type MaterialNonConvergenceError <: Exception end
show(io::IO, nce::NonConvergenceError) = print(io, MAT_NON_CONV_MESSAGE)

abstract MatStatus
abstract MatParameter
abstract MatModel

include("driver.jl")
include("loadcase.jl")
include("utils.jl")

end # module
