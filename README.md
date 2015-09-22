# ConstLab.jl

`ConstLab.jl` is a small package for Julia to test and experiment with constitutive models.
It's main functionality is to call a user given material routine over a time range, aggregate the material response and return it back for analysis. The load applied to the material can be user generated or one of the predefined load cases can be used. To facilitate visualizing the results, some auxiliary plot functions are provided.

## Usage

This README contains a short guide to using `ConstLab.jl`.
It is divided in the following sections:

* **Material** - Details the way a material should be written to be compatible with the package.

* **Constitutive driver** - The main functionality of the package. This describes how to set
up an analysis that calls the material with a given load case.

* **Predefined load cases** - Shows the predefined load cases that can be generated.

* **Plotting** - Describes some of the plotting commands to simplify plotting the results from
the constitutive driver.

### Material

In order to define a new material, the following needs to be implemented:

* A material parameter type `<: MatParameter`. This type should contain material parameters
that are constant throughout the time steps. Typical parameters are elastic modulus
and Poisson's ratio. Algorithmic parameters should also go in this type, for example
the tolerances of a nonlinear solver in the material.

* A material status type ` <: MatStatus`. This should contain quantities
that needs to be saved from the previous time step. Typical quantities to store is the amount of
plastic strain and hardening. The result of running the simulation will contain a vector of the material status in each time step so anything that wants to be analyzed after the simulation should go here.

* A function stress with the specification
```julia
stress(ɛ::Vector, ∆t::Number, ms::MatStatus, mp::MatParameter)
    -> σ::Vector, ATS::Matrix, ms::MatStatus
```
`ε` is the strain for the current time step, `∆t` is the time increment from the last time step, `ms` is the `MatStatus` from the previous timestep and `mp` is the `MatParameter` for the material.

The function should return `σ, ATS, ms` where `σ` is the new stress, `ATS` is the
Algorithmic Tangent Stiffness (or an approximation of it), and `ms` is the updated `MatStatus`.

**Note**: You will need to import `stress` from `ConstLab.jl`so that your newly defined `stress` extends the one in `ConstLab.jl`.

### Constitutive Driver

The `driver` function takes material and load data, runs the analysis. Its specification is

```julia
driver(matstat::MatStatus,
       matpar::MatParameter,
       time_history::Vector,
       ε_history::Matrix,
       σ_history::Matrix,
       ε_control::Vector{Bool};
       [solver parameters]) -> εs::Matrix, σs::Matrix, matstats::Vector
```

Function arguments:

* `matstat`: the initial material status of the material.

* `matpar`: the material parameters of the material.

* `time_history`: the time steps to run the analysis on.

* `ε_history`: the strains for each time step. Should be a matrix of size
`(n_comp, nt)` where `n_comp` is the number
of components of the strain and `nt` is the number of time steps.

* `σ_history`: the target stresses for each time steps. Should be the same size as
`ε_history`.

* `ε_control`: which components are strain controlled. If there are components
that are not strain controlled, `driver` will iterate on these strain components
using the tangent returned from the material to fulfil that the resulting stresses
fulfil the ones given by `σ_history`.

Output:

* `εs`: the strains in iteration in each time step

* `σs`: the stresses in each time step

* `matstats`: vector of the the material status in each time step


#### Optional Solver parameters for `driver`.

The package used to solve for the prescribed stresses is [NLSolve.jl](https://github.com/EconForge/NLsolve.jl). The following can be passed to `driver` as optional arguments to control the solver parameters:

* `method`: Either `:trust_region` or `:newton`. For a description of these methods see `NLSolve.jl`.

* `xtol`: norm difference in `x` between two successive iterates under which
  convergence is declared. Default: `0.0`.

* `ftol`: infinite norm of residuals under which convergence is declared. Default: `1e-5`.

* `iterations`: maximum number of iterations. Default: `100`.

* `err_on_nonconv`: if an error should be raised if the equation does not converge in the
maximum number of iterations. Default: `true`.

* `warn_on_nonconv`: if a warning should should be printed if the equation does not converge in the
maximum number of iterations. Default: `true`. Only used if `err_on_nonconv = false`.

### Predefined load cases

`ConstLab.jl` can help you create input data to `driver` for some common loadcases. The specification for the function `loadcase` is

```julia
loadcase(case, ε_max, ts) -> ε_history, σ_history, ε_control
```

where `case` is one of the predefined load cases given below, `ε_max` is the maximum strain that should be achieved and `ts` is a range or vector containing the time steps of the analysis. The returned values `ε_history`, `σ_history` and `ε_control` are those needed in the `driver` function.

Allowed `case`'s (`11, 22, 33, 23, 13, 12`- Voigt notation assumed):

* `:uniaxial_strain`: full strain control,  ε11 varied.
* `:uniaxial_stress`: enforce σ22 = σ33 = 0, ε11 varied.
* `:biaxial_strain_plstrain`: full strain control, ε11 and ε22 varied.
* `:biaxial_strain_plstress`: enfore σ33 = 0, ε11 and ε22 varied.
* `:simpleshear`: full strain control, ε12 varied.

### Plotting

It is of course possible to manually extract data from the vector of material statuses
that `driver` returns and plot these with whatever plotting package that is desired.
This can however be cumbersome so `ConstLab.jl` therefore extends the `plot` command so that it can
directly take results that come from `driver`.

`ConstLab.jl` uses the [Plots.jl](https://github.com/tbreloff/Plots.jl) package for plotting. This means that you are free to set the backend for the plots as described in the `Plots.jl` manual.

The format for the extended plot command is `plot(x, y, matstats [; mod_x, mod_y])`.

The variable `matstats` should be the vector of `MatStats` returned from `driver`. `x` and `y` can either be a `Symbol`, `Vector` or `Matrix`. `mod_x` and `mod_y` are optional variables that modify `x` and `y`.

#### `x::Symbol`

This will extract the field `x` from each material status. Depending on the type of the field extracted `mod_x` and `mod_y` can do different things.

* `field::Number`: if no `mod_x` is given, the number is plotted, if `mod_x::Function` the function will be applied to the number and be plotted

* `field::Vector`: if `mod_x` is an integer that component of the vector is plotted, if `mod_x::Function` the function will be applied to the vector and be plotted.

* `field::Matrix`: if `mod_x` is an tuple of two integers, that index of the vector is plotted, if `  mod_x::Function` the function will be applied to the matrix and be plotted.

#### `x::Matrix`

* This will extract the column in `x` for the time step. The rules for `mod_x` then applies like for `field::Vector` above.

#### `x::Vector`

Plotted like a normal vector


#### Examples of plotting

* Below, `n_ε_p` is a vector in matstats, `n_κ` is a scalar in matstats. We want to plot the first copmponent of n_ε_p against n_κ:
```
p = plot(:n_ε_p, :n_κ; matstats, mod_x = 1)
```

* Below, `εs` and `σs` are the results from `driver`. `vm` is a function that computes the von Mises
stress. We want to plot the first component of `εs` against the von Mises stress:
```
p = plot(εs, σs, mod_x = 1, mod_y = vm)
```

## Examples

There are a few different examples in the `examples` directory. Note that some of the examples use more
packages than what is needed to run `ConstLab.jl`. For example, some material models in the examples use
the [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) package to compute Jacobians
to the residuals of the non linear equilibrium equations.

## Author

Kristoffer Carlsson (@KristofferC)