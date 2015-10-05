using Parameters
using ForwardDiff
using NLsolve
using ConstLab
using Voigt
using Voigt.Unicode

using Devectorize

import ConstLab.stress

@with_kw immutable MisesMixedHardMS <: MatStatus
    n_ε_p::Vector{Float64} = zeros(6)
    n_α_dev::Vector{Float64} = zeros(6)
    n_κ::Float64 = 0.0
    n_μ::Float64 = 0.0
end

@with_kw immutable MisesMixedHardMP <: MatParameter
    E::Float64
    ν::Float64
    σy::Float64
    H::Float64
    r::Float64
    κ_∞::Float64
    α_∞::Float64
end

const II = veye(6) ⊗ veye(6)
const Idev6 = eye(6,6) - 1/3 * II

function stress(ε, dt, matpar::MisesMixedHardMP, matstat::MisesMixedHardMS)

    @unpack matpar: E, ν, σy, H, r, κ_∞, α_∞
    @unpack matstat: n_ε_p, n_α_dev, n_κ, n_μ

    G = E / 2(1 + ν)
    K = E / 3(1 - 2ν)
    Ee = 2 * G * Idev6 + K * II
    Ee[4:6, 4:6] /= 2.0
    σ_tr = Ee * (ε - n_ε_p)
    σ_dev_tr = dev(σ_tr)

    σ_red_e_tr = sqrt(3/2) * vnorm(dev(σ_tr - n_α_dev))

    Φ = σ_red_e_tr - σy - n_κ

    if Φ < 0
        ms = MisesMixedHardMS(n_ε_p, n_α_dev, n_κ, n_μ)
        return σ_tr, Ee, ms
    else

        # Takes the unknown state vector and σ_dev_para
        function res_wrapper(x, σ_dev_pr)
            _σ_dev = x[1:6]
            _α_dev = x[7:12]
            _κ = x[13]
            _μ = x[14]
            R_σ, R_α, R_κ, R_Φ = compute_residual(_σ_dev, _α_dev, _κ, _μ, σ_dev_pr, matpar, matstat)
            R = [R_σ; R_α; R_κ; R_Φ]
        end

        res_wrapper(x) = res_wrapper(x, σ_dev_tr)

        jac = jacobian(res_wrapper)

        # Initial guess
        x0 = [σ_dev_tr; n_α_dev; n_κ; n_μ]
        res = nlsolve(not_in_place(res_wrapper, jac), x0; iterations = 30, ftol = 1e-7)
        if !NLsolve.converged(res)
            error("No convergence in material routine")
        end

        X = res.zero

        σ_dev_X = X[1:6]
        α_dev_X = X[7:12]
        κ_X = X[13]
        μ_X = X[14]
        @devec σ_X = σ_tr - σ_dev_tr + σ_dev_X
        σ_red_dev_X = dev(σ_X - α_dev_X)
        σ_red_e_X = sqrt(3/2) * vnorm(σ_red_dev_X)
        ε_p = n_ε_p + 3/2 * μ_X ./ σ_red_e_X * σ_red_dev_X

        ms = MisesMixedHardMS(ε_p, α_dev_X, κ_X, μ_X)

        function dRdε_wrapper(ε_para)
            σ_ATS = Ee * (ε_para - n_ε_p)
            σ_dev_ATS = dev(σ_ATS)
            res_wrapper(X, σ_dev_ATS)
        end


        J = jac(X)
        dRdε_f = jacobian(dRdε_wrapper)
        dRdε = dRdε_f(ε)
        dXdε = -J \ dRdε
        dσdev_dε = dXdε[1:6, 1:6]

        ATS = K * (veye(6) * veye(6)') + dσdev_dε

       return σ_X, ATS, ms
    end
end

function compute_residual(σ_dev, α_dev, κ, μ, σ_dev_tr, matpar, matstat)

    @unpack matpar: E, ν, σy, H, r, κ_∞, α_∞
    @unpack matstat: n_ε_p, n_α_dev, n_κ, n_μ

    G = E / 2(1 + ν)

    @devec σ_red_dev = σ_dev - α_dev
    σ_red_e = sqrt(3/2) * vnorm(σ_red_dev)
    σ_red_dev_hat = σ_red_dev / σ_red_e

    @devec R_σ = σ_dev - σ_dev_tr + 3.*G.*μ .* σ_red_dev_hat
    R_κ = κ - n_κ - r * H * μ * (1 - κ / κ_∞)
    @devec R_α = α_dev - n_α_dev - (1-r) .* H .* μ .* (σ_red_dev_hat - 1./α_∞ .* α_dev)
    R_Φ = σ_red_e - σy - κ

    return R_σ, R_α, R_κ, R_Φ
end
