using Parameters
using ForwardDiff
using NLsolve
using ConstLab
using Voigt
using Voigt.Unicode
using Devectorize

const VOIGT_SIZE = 9


create_component_macros("mises_mixed", (VOIGT_SIZE, 1, VOIGT_SIZE, 1))

const II = veye(VOIGT_SIZE) ⊗ veye(VOIGT_SIZE)
const Idev = eye(VOIGT_SIZE, VOIGT_SIZE) - 1/3 * II
vm(x) = sqrt(3/2) * vnorm(dev(x))


@with_kw immutable MisesMixedHardMS
    ₙεₚ::Vector{Float64} = zeros(VOIGT_SIZE)
    ₙσ::Vector{Float64} = zeros(VOIGT_SIZE)
    ₙα::Vector{Float64} = zeros(VOIGT_SIZE)
    ₙκ::Float64 = 0.0
    ₙμ::Float64 = 0.0
    loading::Bool = false
end

@with_kw immutable MisesMixedHardMP
    E::Float64
    ν::Float64
    Ee::Matrix{Float64} =   (G = E / 2(1 + ν);
                            K = E / 3(1 - 2ν);
                            Ee = 2 * G * Idev+ K * II;
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
    @unpack matstat: ₙεₚ, ₙα, ₙκ, ₙμ

    σₜᵣ = Ee * (ε - ₙεₚ)
  
    σ_red_e_tr = sqrt(3/2) * vnorm(dev(σₜᵣ - ₙα))

    Φ = σ_red_e_tr - σy - ₙκ

    if Φ < 0
        ms = MisesMixedHardMS(ₙεₚ, σₜᵣ, ₙα, ₙκ, ₙμ, false)
        return σₜᵣ, ms
    else
        # Initial guess
        X = [σₜᵣ; ₙκ; ₙα; ₙμ]
        
        #=
        R!(x, fx) = compute_residual!(fx, x, σₜᵣ, matpar, matstat)
        res = nlsolve(R!, X; iterations = 30, ftol = 1e-4, method=:newton, autodiff = true)
        if !NLsolve.converged(res)
            error("No convergence in material routine")
        end
        X = res.zero::Vector{Float64}
        =#
        
        
        max_iters = 30
        iter = 0
        tol = 1e-4

        n_unknowns = 2*VOIGT_SIZE + 2
        fx = zeros(n_unknowns)
        J = zeros(n_unknowns, n_unknowns)
        R2!(fx, x) = compute_residual!(fx, x, σₜᵣ, matpar, matstat)
        dRdx! = jacobian(R2!, mutates = true, output_length = n_unknowns)
       # println("|fx| \t \t |dx|")
        while true
            iter += 1
            R2!(fx, X)
            
            if iter == 1
          #      println("$(maximum(abs(fx))) \t NaN")
            else
           #     println("$(maximum(abs(fx))) \t $(norm(dx))")
            end
            
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
        
        ConstLab.@unpack_mises_mixed (σ, κ, α, μ) = X

        σ_red_dev = dev(σ - α)
        σ_red_e = sqrt(3/2) * vnorm(σ_red_dev)
        εₚ = 3/2 * μ / σ_red_e * σ_red_dev
        εₚ += ₙεₚ

        ms = MisesMixedHardMS(εₚ, σ, α, κ, μ, true)

        return σ, ms
    end
end

function compute_residual!(R, x, σₜᵣ, matpar, matstat)
    @unpack matpar: E, ν, σy, H, r, κ∞, α∞
    @unpack matstat: ₙεₚ, ₙα, ₙκ, ₙμ

    ConstLab.@unpack_mises_mixed (σ, κ, α, μ) = x

    G = E / 2(1 + ν)

    σ_red_dev = dev(σ - α)
    σ_red_e = vm(σ - α)
    σ_red_dev_hat = σ_red_dev / σ_red_e

    @devec R_σ = σ - σₜᵣ + 3.*G.*μ .* σ_red_dev_hat
    R_κ = κ - ₙκ - r * H * μ * (1 - κ / κ∞)
    @devec R_α = α - ₙα - (1-r) .* H .* μ .* (σ_red_dev_hat - 1./α∞ .* α)
    R_Φ = σ_red_e - σy - κ
    
    ConstLab.@pack_mises_mixed R = (R_σ, R_κ, R_α, R_Φ)
 
    return
end

function compute_jacobian(x, matpar, matstat)
    @unpack mp: E, Ee, ν, σy, H, r, κ∞, α∞
    @unpack matstat: ₙεₚ, ₙσ, ₙα, ₙκ, ₙμ

    zerov = zeros(VOIGT_SIZE)
    zeroc = zeros(1, VOIGT_SIZE)
    
    σ = ₙσ
    α = ₙα
    κ = ₙκ
    μ = ₙμ

    G = E / 2(1 + ν)

    σ_red_dev = dev(σ - α)
    σ_red_e = vm(σ - α)
    σ_red_dev_hat = σ_red_dev / σ_red_e

    I = veye(9,9)
    ν = 3/2 * σ_red_dev / σ_red_e
    Q = Idev - 3 / (2(σ_red_e)^2) * (σ_red_dev ⊗ σ_red_dev)
    N = 3 / (2σ_red_e) * Q
    Nα = -N

    dRσdσ = I + 2*G*μ*N
    dRσdκ = zerov
    dRσdα = 2*G*μ*Nα
    dRσdμ = 2*G*ν

    dRκdσ = zeroc
    dRκdκ = 1 + μ*r*H / κ∞
    dRκdα = zeroc
    dRκdμ = -r*H*(1- κ/κ∞)

    dRαdσ = -μ*(1 - r) * H * 2/3 * N
    dRαdκ = zerov
    dRαdα = I - μ*(1-r) * H * (2/3 * Nα - I/α∞)
    dRαdμ = -(1-r)*H*(2/3 * ν - α/α∞)

    dRΦdσ = ν'
    dRΦdκ = -1
    dRΦdα = -ν'
    dRΦdμ = 0
   
    J = [dRσdσ dRσdκ dRσdα dRσdμ;
        dRκdσ dRκdκ dRκdα dRκdμ;
        dRαdσ dRαdκ dRαdα dRαdμ;
        dRΦdσ dRΦdκ dRΦdα dRΦdμ]
    
    return J
end

function ATS(ε, dt, matpar::MisesMixedHardMP, matstat::MisesMixedHardMS)
    @unpack matpar: Ee, E, ν
    @unpack matstat: ₙεₚ, ₙσ, ₙα, ₙκ, ₙμ, loading
    X = [ₙσ; ₙκ; ₙα; ₙμ]

    K = E / 3(1 - 2ν)
   
    if !loading
        return Ee
    else
        σₜᵣ = Ee * (ε - ₙεₚ)

        R2!(fx, x) = compute_residual!(fx, x, σₜᵣ, matpar, matstat)
        dRdx! = jacobian(R2!, output_length = 2*VOIGT_SIZE + 2)
        dRdX = dRdx!(X)

        dσdε = inv(dRdX)[1:VOIGT_SIZE, 1:VOIGT_SIZE] * Ee
        return dσdε
    end
end
