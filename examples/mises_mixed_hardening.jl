using Parameters
using ForwardDiff
using NLsolve
using ConstLab
using Voigt
using Voigt.Unicode
using Devectorize

@with_kw immutable MisesMixedHardMS
    ₙεₚ::Vector{Float64} = zeros(6)
    ₙσdev::Vector{Float64} = zeros(6)
    ₙαdev::Vector{Float64} = zeros(6)
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
                            Ee[4:6, 4:6] /= 2.0;
                            Ee)
    σy::Float64
    H::Float64
    r::Float64
    κ∞::Float64
    α∞::Float64
    loading::Bool = false
end

create_component_macro("mises_mixed", (6, 6, 1, 1))

const II = veye(6) ⊗ veye(6)
const Idev6 = eye(6,6) - 1/3 * II

function stress(ε, dt, matpar::MisesMixedHardMP, matstat::MisesMixedHardMS)
    @unpack matpar: E, Ee, ν, σy, H, r, κ∞, α∞
    @unpack matstat: ₙεₚ, ₙαdev, ₙκ, ₙμ

    σₜᵣ = Ee * (ε - ₙεₚ)
    σdevₜᵣ = dev(σₜᵣ)

    σ_red_e_tr = sqrt(3/2) * vnorm(dev(σₜᵣ - ₙαdev))

    Φ = σ_red_e_tr - σy - ₙκ

    if Φ < 0
        ms = MisesMixedHardMS(ₙεₚ, σdevₜᵣ, ₙαdev, ₙκ, ₙμ, false)
        return σₜᵣ, ms
    else

        # Takes the unknown state vector and σdev_tr
        R(x) = compute_residual(x, σdevₜᵣ, matpar, matstat)
        dRdx = jacobian(R)

        # Initial guess
        x0 = [σdevₜᵣ; ₙαdev; ₙκ; ₙμ]
        res = nlsolve(not_in_place(R, dRdx), x0; iterations = 30, ftol = 1e-7)
        if !NLsolve.converged(res)
            error("No convergence in material routine")
        end

        X = res.zero::Vector{Float64}
        ConstLab.@unpack_comp_mises_mixed (σdev, αdev, κ, μ) = X

        @devec σ = σₜᵣ - σdevₜᵣ + σdev
        σ_red_dev = dev(σ - αdev)
        σ_red_e = sqrt(3/2) * vnorm(σ_red_dev)
        εₚ = ₙεₚ + 3/2 * μ / σ_red_e * σ_red_dev

        ms = MisesMixedHardMS(εₚ, σdev, αdev, κ, μ, true)

       return σ, ms
    end
end



function compute_residual(x, σdevₜᵣ, matpar, matstat)
    @unpack matpar: E, ν, σy, H, r, κ∞, α∞
    @unpack matstat: ₙεₚ, ₙαdev, ₙκ, ₙμ

    ConstLab.@unpack_comp_mises_mixed (σdev, αdev, κ, μ) = x

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

function ATS(ε, dt, matpar::MisesMixedHardMP, matstat::MisesMixedHardMS)
    @unpack matpar: Ee
    @unpack matstat: ₙεₚ, ₙσdev, ₙαdev, ₙκ, ₙμ, loading
    X = [ₙσdev; ₙαdev; ₙκ; ₙμ]

    K = E / 3(1 - 2ν)

    if !loading
        return Ee
    else

        σₜᵣ = Ee * (ε - ₙεₚ)
        σdevₜᵣ = dev(σₜᵣ)

        R_x(x) = compute_residual(x, σdevₜᵣ, matpar, matstat)

        dRdX = jacobian(R_x, X)
        dσdevdε = dRdX[1:6, 1:6] \ Ee
        ATS = K * II + dσdevdε
        return ATS
    end
end
