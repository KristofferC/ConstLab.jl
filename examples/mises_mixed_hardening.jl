using Parameters
using ForwardDiff
using NLsolve
using ConstLab
using Voigt
using Voigt.Unicode
using Devectorize

const VOIGT_SIZE = 9


create_component_macro("mises_mixed", (VOIGT_SIZE, VOIGT_SIZE, 1, 1))

const II = veye(VOIGT_SIZE) ⊗ veye(VOIGT_SIZE)
const Idev6 = eye(VOIGT_SIZE, VOIGT_SIZE) - 1/3 * II


@with_kw immutable MisesMixedHardMS
    ₙεₚ::Vector{Float64} = zeros(VOIGT_SIZE)
    ₙσdev::Vector{Float64} = zeros(VOIGT_SIZE)
    ₙαdev::Vector{Float64} = zeros(VOIGT_SIZE)
    ₙκ::Float64 = 0.0
    ₙμ::Float64 = 0.0
    loading::Bool = false
end

@with_kw immutable MisesMixedHardMP
    E::Float64
    ν::Float64
    Ee::Matrix{Float64} =   (G = E / 2(1 + ν);
                            K = E / 3(1 - 2ν);
                            Ee = 2 * G * Idev6 + K * II;
                            if VOIGT_SIZE == 6
                                Ee[4:6, 4:6] /= 2.0;
                            end;
                            Ee)
    σy::Float64
    H::Float64
    r::Float64
    κ∞::Float64
    α∞::Float64
    loading::Bool = false
end

function stress(ε, dt, matpar::MisesMixedHardMP, matstat::MisesMixedHardMS)
    @unpack matpar: E, Ee, ν, σy, H, r, κ∞, α∞
    @unpack matstat: ₙεₚ, ₙαdev, ₙκ, ₙμ

    ε = unsym(ε)
    σₜᵣ = Ee * (ε - ₙεₚ)
    σdevₜᵣ = dev(σₜᵣ)

    σ_red_e_tr = sqrt(3/2) * vnorm(dev(σₜᵣ - ₙαdev))

    Φ = σ_red_e_tr - σy - ₙκ

    if Φ < 0
        ms = MisesMixedHardMS(ₙεₚ, σdevₜᵣ, ₙαdev, ₙκ, ₙμ, false)
        return σₜᵣ, ms
    else

        # Takes the unknown state vector and σdev_tr
        R!(x, fx) = compute_residual!(fx, x, σdevₜᵣ, matpar, matstat)
   
        # Initial guess
        X = [σdevₜᵣ; ₙαdev; ₙκ; ₙμ]
        #=
        res = nlsolve(R!, X; iterations = 30, ftol = 1e-4, method=:newton, autodiff = true)
        if !NLsolve.converged(res)
            error("No convergence in material routine")
        end
        =#
        
        
        max_iters = 30
        iter = 0
        tol = 1e-4

        n_unknowns = 2*VOIGT_SIZE + 2
        fx = zeros(n_unknowns)
        J = zeros(n_unknowns, n_unknowns)
        R2!(fx, x) = compute_residual!(fx, x, σdevₜᵣ, matpar, matstat)
        dRdx! = jacobian(R2!, mutates = true, output_length = n_unknowns)
        while true
            iter += 1
            R2!(fx, X)
            #=
            if iter == 1
                println("$(maximum(abs(fx))) \t NaN")
            else
                println("$(maximum(abs(fx))) \t $(norm(dx))")
            end
            =#
            if maximum(abs(fx)) < tol
                break
            end
            
            if iter == max_iters
                 error("No convergence in material routine")
            end
            dRdx!(J, X)
            dx = J \ fx
            X = X - dx
        end
        
        
        #X = res.zero::Vector{Float64}
        ConstLab.@unpack_comp_mises_mixed (σdev, αdev, κ, μ) = X

        @devec σ = σₜᵣ - σdevₜᵣ + σdev
        σ_red_dev = dev(σ - αdev)
        σ_red_e = sqrt(3/2) * vnorm(σ_red_dev)
        εₚ = 3/2 * μ / σ_red_e * σ_red_dev
        εₚ += ₙεₚ

        ms = MisesMixedHardMS(εₚ, σdev, αdev, κ, μ, true)

        return σ, ms
    end
end



function compute_residual!(R, x, σdevₜᵣ, matpar, matstat)
    @unpack matpar: E, ν, σy, H, r, κ∞, α∞
    @unpack matstat: ₙεₚ, ₙαdev, ₙκ, ₙμ

    ConstLab.@unpack_mises_mixed (σdev, αdev, κ, μ) = x

    G = E / 2(1 + ν)

    @devec σ_red_dev = σdev - αdev
    σ_red_e = sqrt(3/2) * vnorm(σ_red_dev)
    σ_red_dev_hat = σ_red_dev / σ_red_e

    @devec R_σ = σdev - σdevₜᵣ + 3.*G.*μ .* σ_red_dev_hat
    R_κ = κ - ₙκ - r * H * μ * (1 - κ / κ∞)
    @devec R_α = αdev - ₙαdev - (1-r) .* H .* μ .* (σ_red_dev_hat - 1./α∞ .* αdev)
    R_Φ = σ_red_e - σy - κ
    
    ConstLab.@pack_mises_mixed R = (R_σ, R_α, R_κ, R_Φ)
 
    return
end

function ATS(ε, dt, matpar::MisesMixedHardMP, matstat::MisesMixedHardMS)
    @unpack matpar: Ee, E, ν
    @unpack matstat: ₙεₚ, ₙσdev, ₙαdev, ₙκ, ₙμ, loading
    X = [ₙσdev; ₙαdev; ₙκ; ₙμ]

    K = E / 3(1 - 2ν)
    ε = unsym(ε)


    if !loading
        return Ee
    else
        σₜᵣ = Ee * (ε - ₙεₚ)
        σdevₜᵣ = dev(σₜᵣ)

        R2!(fx, x) = compute_residual!(fx, x, σdevₜᵣ, matpar, matstat)
        dRdx! = jacobian(R2!, output_length = 2*VOIGT_SIZE + 2)
        
        dRdX = dRdx!(X)
        invJ = inv(dRdX)
        dσdevdε = invJ[1:VOIGT_SIZE, 1:VOIGT_SIZE] * Ee
        println(dσdevdε)
        ATS = K * II + dσdevdε
        println(ATS)
        println("----------")
        return ATS
    end
end
