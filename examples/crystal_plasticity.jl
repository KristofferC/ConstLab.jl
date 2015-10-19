using Parameters
using ForwardDiff
using NLsolve
using ConstLab
using Devectorize

using Voigt
using Voigt.Unicode

import ConstLab.stress

@with_kw immutable CrystPlastMS  <: MatStatus
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


@with_kw immutable CrystPlastMP  <: MatParameter
    E::Float64
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
    CrystPlastMP(E, ν, σy, n, H, q, D, tstar, angles, sxm_sym, nslip)
end

const II = veye(6) ⊗ veye(6)
const Idev6 = eye(6,6) - 1/3 * II

function stress(ε, dt, matpar::CrystPlastMP, matstat::CrystPlastMS)

    @unpack matstat: n_ε_p, n_κ, n_τ, n_μ
    @unpack matpar: E, ν, σy, sxm_sym, nslip

    G = E / 2(1 + ν)
    K = E / 3(1 - 2ν)
    Ee = 2 * G * Idev6 + K * II
    Ee[4:6, 4:6] /= 2.0
    σ_tr = Ee * (ε - n_ε_p)

    function res_wrapper(x)
        σ = x[1:6]
        κ = x[7:7+nslip-1]
        μ = x[7+nslip:end]
        R_σ, R_κ, R_Φ = compute_residual(σ, κ, μ, σ_tr, dt, matpar, matstat)
        [R_σ; R_κ; R_Φ]
    end

    x0 = [σ_tr; n_κ; n_μ]

    jac = jacobian(res_wrapper)
    res = nlsolve(not_in_place(res_wrapper, jac), x0; iterations = 30, store_trace = true, ftol = 1e-6)

    if !NLsolve.converged(res)
        error("No convergence in material routine")
    end

    X = res.zero
    σ_X = X[1:6]
    κ_X = X[7:7+nslip-1]
    μ_X = X[7+nslip:end]
    ε_p = copy(n_ε_p)
    τ_X = zeros(eltype(σ_X), nslip)
    for α = 1:nslip
        τ_X[α] = σ_X : sxm_sym[α]
        ε_p += μ_X[α] * sxm_sym[α] * sign(n_τ[α])
    end

    ms = CrystPlastMS(ε_p, κ_X, τ_X, μ_X)
    return σ_X, zeros(6,6), ms
end


function compute_residual(σ, κ, μ, σ_tr, dt, matpar, matstat)

    @unpack_CrystPlastMP matpar
    @unpack_CrystPlastMS matstat

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
    G = E / 2(1 + ν)
    K = E / 3(1 - 2ν)
    Ee = 2 * G * Idev6 + K * II

    R_σ += Ee * ep

    return R_σ, R_κ, R_Φ
end
