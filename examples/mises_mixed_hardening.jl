using Parameters
using ForwardDiff
using NLsolve
using ConstLab
using Voigt
using Voigt.Unicode

using Devectorize

@with_kw immutable MisesMixedHardMS <: MatStatus
    ₙεₚ::Vector{Float64} = zeros(6)
    ₙσdev::Vector{Float64} = zeros(6)
    ₙαdev::Vector{Float64} = zeros(6)
    ₙκ::Float64 = 0.0
    ₙμ::Float64 = 0.0
end

@with_kw immutable MisesMixedHardMP <: MatParameter
    E::Float64
    ν::Float64
    σy::Float64
    H::Float64
    r::Float64
    κ∞::Float64
    α∞::Float64
end

const II = veye(6) ⊗ veye(6)
const Idev6 = eye(6,6) - 1/3 * II

function stress(ε, dt, matpar::MisesMixedHardMP, matstat::MisesMixedHardMS)
    @unpack matpar: E, ν, σy, H, r, κ∞, α∞
    @unpack matstat: ₙεₚ, ₙαdev, ₙκ, ₙμ

    G = E / 2(1 + ν)
    K = E / 3(1 - 2ν)
    Ee = 2 * G * Idev6 + K * II
    Ee[4:6, 4:6] /= 2.0
    σₜᵣ = Ee * (ε - ₙεₚ)
    σdevₜᵣ = dev(σₜᵣ)

    σ_red_e_tr = sqrt(3/2) * vnorm(dev(σₜᵣ - ₙαdev))

    Φ = σ_red_e_tr - σy - ₙκ

    if Φ < 0
        ms = MisesMixedHardMS(ₙεₚ, σdevₜᵣ, ₙαdev, ₙκ, ₙμ)
        return σₜᵣ, ms
    else

        # Takes the unknown state vector and σdev_para
        R(x) = compute_residual(x, σdevₜᵣ, matpar, matstat)
        dRdx = jacobian(R)

        # Initial guess
        x0 = [σdevₜᵣ; ₙαdev; ₙκ; ₙμ]
        res = nlsolve(not_in_place(R, dRdx), x0; iterations = 30, ftol = 1e-7)
        if !NLsolve.converged(res)
            error("No convergence in material routine")
        end

        X = res.zero::Vector{Float64}

        σdev = X[1:6]
        αdev = X[7:12]
        κ = X[13]
        μ = X[14]

        @devec σ = σₜᵣ - σdevₜᵣ + σdev
        σ_red_dev = dev(σ - αdev)
        σ_red_e = sqrt(3/2) * vnorm(σ_red_dev)
        εₚ = ₙεₚ + 3/2 * μ / σ_red_e * σ_red_dev

        ms = MisesMixedHardMS(εₚ, σdev, αdev, κ, μ)


       return σ, ms
    end
end

function ATS(ε, dt, matpar::MisesMixedHardMP, matstat::MisesMixedHardMS)
    @unpack matpar: E, ν, σy, H, r, κ∞, α∞
    @unpack matstat: ₙεₚ, ₙσdev, ₙαdev, ₙκ, ₙμ

    X = [ₙσdev; ₙαdev; ₙκ; ₙμ]

    G = E / 2(1 + ν)
    K = E / 3(1 - 2ν)
    Ee = 2 * G * Idev6 + K * II
    Ee[4:6, 4:6] /= 2.0

    σ_red_e = sqrt(3/2) * vnorm(ₙσdev - dev(ₙαdev))

    Φ = σ_red_e - σy - ₙκ
    if Φ < 0
        return Ee
    else
       function R_ε(ε)
            σ = Ee * (ε - ₙεₚ)
            σdev = dev(σ)
            compute_residual(X, σdev, matpar, matstat)
        end

        σₜᵣ = Ee * (ε - ₙεₚ)
        σdevₜᵣ = dev(σₜᵣ)

        R_x(x) = compute_residual(x, σdevₜᵣ, matpar, matstat)

        dRdX = jacobian(R_x, X)
        dRdε = jacobian(R_ε, ε)
        dXdε = - dRdX \ dRdε
        dσdevdε = dXdε[1:6, 1:6]

        ATS = K * II + dσdevdε
        #ATS[:, 4:6] /= 2.0

        return ATS
    end
end


function compute_residual(x, σdevₜᵣ, matpar, matstat)
    @unpack matpar: E, ν, σy, H, r, κ∞, α∞
    @unpack matstat: ₙεₚ, ₙαdev, ₙκ, ₙμ

    σdev = x[1:6]
    αdev = x[7:12]
    κ = x[13]
    μ = x[14]

    G = E / 2(1 + ν)

    @devec σ_red_dev = σdev - αdev
    σ_red_e = sqrt(3/2) * vnorm(σ_red_dev)
    σ_red_dev_hat = σ_red_dev / σ_red_e

    @devec R_σ = σdev - σdevₜᵣ + 3.*G.*μ .* σ_red_dev_hat
    R_κ = κ - ₙκ - r * H * μ * (1 - κ / κ∞)
    @devec R_α = αdev - ₙαdev - (1-r) .* H .* μ .* (σ_red_dev_hat - 1./α∞ .* αdev)
    R_Φ = σ_red_e - σy - κ

    return [R_σ; R_α; R_κ; R_Φ]
end
