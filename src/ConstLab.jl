module ConstLab

import Base.show

using NLsolve
using Plots

#import Plots: plot, plot!
export loadcase, driver, MatStatus, MatParameter #, plot, plot!

const NON_CONV_MESSAGE = "constitutive iterations did not converge"

type NonConvergenceError <: Exception end
show(io::IO, nce::NonConvergenceError) = print(io, NON_CONV_MESSAGE)

abstract MatStatus
abstract MatParameter
abstract MatModel

include("driver.jl")
#include("plotting.jl")
include("loadcase.jl")

end # module
