using Parameters
using ForwardDiff
using NLsolve
using ConstLab
using Devectorize

using Voigt
using Voigt.Unicode

const II = veye(6) ⊗ veye(6)
const Idev = eye(6,6) - 1/3 * II

@with_kw immutable CrystPlastMS
    n_ε_p::Vector{Float64}
    # One per slip system:
    n_κ::Vector{Float64}
    n_τ::Vector{Float64}
    n_μ::Vector{Float64}
end

function CrystPlastMS(nslip)
    n_ε_p = zeros(6)
    n_κ = zeros(nslip)
    n_τ = zeros(nslip)
    n_μ = zeros(nslip)
    CrystPlastMS(n_ε_p, n_κ, n_τ, n_μ)
end


@with_kw immutable CrystPlastMP
    E::Float64
    Ee::Matrix{Float64}
    ν::Float64
    σy::Float64
    n::Float64
    H::Float64
    q::Float64
    D::Float64
    tstar::Float64
    angles::Vector{Float64}
    sxm_sym::Vector{Vector{Float64}}
    nslip::Int
end


function CrystPlastMP(E, ν, σy, n, H, q, D, tstar, angles)
    sxm_sym = Vector{Vector{Float64}}()
    nslip = length(angles)
    for α = 1:nslip
        t = deg2rad(angles[α])
        s = [cos(t), sin(t), 0.0]
        m = [cos(t + pi/2), sin(t + pi/2), 0.0]
        sxm = voigtsym(s * m')
        push!(sxm_sym, sxm)
    end
    G = E / 2(1 + ν);
    K = E / 3(1 - 2ν);
    Ee = 2 * G * Idev+ K * II;
    Ee[4:6, 4:6] /= 2.0;
    Ee
    CrystPlastMP(E, Ee, ν, σy, n, H, q, D, tstar, angles, sxm_sym, nslip)
end

function stress(ε, dt, matpar::CrystPlastMP, matstat::CrystPlastMS)
    @unpack matstat: n_ε_p, n_κ, n_τ, n_μ
    @unpack matpar: Ee, ν, σy, sxm_sym, nslip

    σ_tr = Ee * (ε - n_ε_p)

    x0 = [σ_tr; n_κ; n_μ]
    
    R!(x, fx) = compute_residual!(fx, x, σ_tr, dt, matpar, matstat)
    res = nlsolve(R!, x0; iterations = 30, ftol = 1e-4, autodiff = true)
    if !NLsolve.converged(res)
        error("No convergence in material routine")
    end
    X = res.zero::Vector{Float64}
    ConstLab.@unpack_cp (σ, κ, μ) = X
    ε_p = copy(n_ε_p)
    τ = zeros(nslip)
    for α = 1:nslip
        τ[α] = σ : sxm_sym[α]
        ε_p += μ[α] * sxm_sym[α] * sign(n_τ[α])
    end

    return σ, CrystPlastMS(ε_p, κ, τ, μ)
end


function compute_residual!(fx, x, σ_tr, dt, matpar, matstat)
    @unpack matstat: n_ε_p, n_κ, n_τ, n_μ
    @unpack matpar: Ee, ν, σy, sxm_sym, nslip, H, q, tstar, D, n
    
    ConstLab.@unpack_cp (σ, κ, μ) = x

    R_Φ = zeros(eltype(σ), nslip)
    R_κ = zeros(eltype(σ), nslip)
    Φ = zeros(eltype(σ), nslip)
    τ = zeros(eltype(σ), nslip)

    for α=1:nslip
        R_κ[α] = κ[α] - n_κ[α]
        for β = 1:nslip
            R_κ -= H * μ[β] * (q + (1 - q) * Int(α == β))
        end
        τ[α] = σ : sxm_sym[α]
        Φ[α] = abs(τ[α]) - (κ[α] + σy)
        R_Φ[α] = μ[α] * tstar - dt * max(0, Φ[α]/D)^n
    end

    R_σ = σ - σ_tr
    ep = zeros(eltype(μ), 6)
    for α = 1:nslip
        ep += μ[α] .* sxm_sym[α] .* sign(τ[α])
    end

    R_σ += Ee * ep
    
    ConstLab.@pack_cp fx = (R_σ, R_κ, R_Φ)
    return
end
