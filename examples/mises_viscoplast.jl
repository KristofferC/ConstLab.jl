using Parameters
using ForwardDiff
using NLsolve
using Devectorize

using Voigt
using Voigt.Unicode

using ConstLab

create_component_macros("mises_visc", (1, 1, 1))

const II = veye(6) ⊗ veye(6)
const Idev = eye(6,6) - 1/3 * II

@with_kw immutable ViscoPlastMS
    ₙεₚ::Vector{Float64} = zeros(6)
    ₙσ::Vector{Float64} = zeros(6)
    ₙκ::Float64 = 0.0
    ₙμ::Float64 = 0.0
    loading::Bool = false
end

@with_kw immutable ViscoPlastMP
    E::Float64
    ν::Float64
    Ee::Matrix{Float64} =   (G = E / 2(1 + ν);
                            K = E / 3(1 - 2ν);
                            Ee = 2 * G * Idev+ K * II;
                            Ee[4:6, 4:6] /= 2.0;
                            Ee)
    σy::Float64
    H::Float64
    n::Float64
    σc::Float64
    t★::Float64
end

function stress(ε, ∆t, matpar::ViscoPlastMP, matstat::ViscoPlastMS)
    @unpack matpar: E, Ee, ν, σy, H, n, σc, t★
    @unpack matstat: ₙεₚ, ₙσ, ₙκ, ₙμ

    σₜᵣ = Ee * (ε - ₙεₚ)
    σdevₜᵣ = dev(σₜᵣ)
    sₜᵣ = sqrt(3/2) * vnorm(σdevₜᵣ)
    Φ = sₜᵣ - σy - ₙκ

    if Φ < 0
        return σₜᵣ, ViscoPlastMS(ₙεₚ, σₜᵣ, ₙκ, ₙμ, false)
    else

        R!(x, fx) = compute_residual!(fx, x, sₜᵣ, ∆t, matpar, matstat)
        x0 = [sₜᵣ, ₙκ, ₙμ]
        
        res = nlsolve(R!, x0; iterations = 20, ftol=1e-5, method=:newton, autodiff = true)
        if !converged(res)
            error("Material did not converge")
        end
        X = res.zero::Vector{Float64}
        ConstLab.@unpack_mises_visc (s, κ, μ) =  X

        G = E / 2(1 + ν)
        @devec σ_dev = 1 ./ (1 + 3 .* G .* μ ./ s) .* σdevₜᵣ
        @devec σ = σₜᵣ - σdevₜᵣ + σ_dev
        @devec εₚ = ₙεₚ + 3 ./ (2 .* s) .* μ .* σ_dev

        return σ, ViscoPlastMS(εₚ, σ, κ, μ, true)
    end
end

function compute_residual!(fx, x, sₜᵣ, ∆t, matpar, matstat)
    @unpack matpar: E, ν, σy, H, n, σc, t★
    @unpack matstat: ₙεₚ, ₙσ, ₙκ, ₙμ

    ConstLab.@unpack_mises_visc (s, κ, μ) = x

    Φ = max(0.0, s - σy - κ)
    G = E / 2(1 + ν)
    R_s = s - sₜᵣ + 3G*μ
    R_κ = κ - ₙκ - H * μ
    Γ′ = (Φ / σc)^n
    Γ′′ =  n / σc * (Φ / σc)^(n-1)
    R_μ = Γ′ - t★ / ∆t * μ

    R_s′ = R_s/3G
    R_κ′ = R_κ/H
    R_μ′ = R_μ/Γ′′
    ConstLab.@pack_mises_visc fx = (R_s′, R_κ′, R_μ′) 
    return
end

function compute_jacobian(x, ∆t, matpar, matstat)
    @unpack matpar: E, ν, σy, H, n, σc, t★
    
    ConstLab.@unpack_mises_visc (s, κ, μ) = x

    G = E / 2(1 + ν)
    Φ = max(zero(eltype(x)), s - σy - κ)
    Γ′′ =  n / σc * (Φ / σc)^(n-1)

    J = zeros(3,3)
    J[1,1] = 1/3G
    J[1,3] = 1
    J[2,2] = 1/H
    J[2,3] = -1
    J[3,1] = 1
    J[3,2] = -1
    J[3,3] = -1/Γ′′ *t★/∆t
    return J
end

function ATS(ε, ∆t, matpar::ViscoPlastMP, matstat::ViscoPlastMS)
    @unpack matpar: E, Ee, ν, σy, H, n, σc, t★
    @unpack matstat: ₙεₚ, ₙσ, ₙκ, ₙμ, loading

    if !loading
        return Ee
    else
        # ATS
        σ, μ, κ = ₙσ, ₙμ, ₙκ
        σdev = dev(σ)
        s = sqrt(3/2) * vnorm(σdev)
        Φ = s - σy - κ
        G = E / 2(1 + ν)
        
        σdevdev = σdev ⊗ σdev
        Γ′′ =  n / σc * (Φ / σc)^(n-1)
        h_at = 3 * G + H + 1 / Γ′′ * t★ / ∆t
        b = (3 * G * μ / s) / (1 + 3 * G * μ / s)
        @devec Q = Idev - (3./(2.*s.^2) .* (σdevdev))
        @devec ATS = Ee + (- 2 .* G .* b .* Q - 9 .* G.^2 ./ (h_at .* s.^2) .* (σdevdev))
        @devec ATS[4:6, 4:6] ./= 2.0
        return ATS
    end
end
