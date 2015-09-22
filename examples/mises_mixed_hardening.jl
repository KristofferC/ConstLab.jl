using Parameters
using ForwardDiff
using NLsolve
using ConstLab

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

const Id = Float64[1, 1, 1, 0, 0, 0, 0, 0, 0]

const Idev = 1/3 * Float64[  2    -1    -1     0     0     0     0     0     0;
                            -1     2    -1     0     0     0     0     0     0;
                            -1    -1     2     0     0     0     0     0     0;
                             0     0     0     3     0     0     0     0     0;
                             0     0     0     0     3     0     0     0     0;
                             0     0     0     0     0     3     0     0     0;
                             0     0     0     0     0     0     3     0     0;
                             0     0     0     0     0     0     0     3     0;
                             0     0     0     0     0     0     0     0     3];

function to_9comp(v::Vector, stress = false)
    length(v) == 6 || throw(ArgumentError("Wrong length"))
    if stress
        f = 1.0
    else
        f = 2.0
    end
    v9 = [v; v[4]; v[5]; v[6]]
    v9[4:end] /= f
    v9
end

function to_6comp(v::Vector, stress = false)
    length(v) == 9 || throw(ArgumentError("Wrong length"))
    if stress
        f = 1.0
    else
        f = 2.0
    end
    v6 = v[1:6]
    v6[4:end] *= f
    v6
end

function to_6comp(m::Matrix)
    size(m) == (9,9) || throw(ArgumentError("Wrong size"))
    m6 = m[1:6, 1:6]
    m6[4:6, :] /= 2.0
    m6
end

function stress(ε, dt, matpar::MisesMixedHardMP, matstat::MisesMixedHardMS)

    @unpack_MisesMixedHardMP matpar
    @unpack_MisesMixedHardMS matstat
    ε = to_9comp(ε)
    n_α_dev = to_9comp(n_α_dev, true)
    n_ε_p = to_9comp(n_ε_p)

    G = E / 2(1 + ν)
    K = E / 3(1 - 2ν)
    Ee = 2 * G * Idev + K * (Id * Id')
    σ_tr = Ee * (ε - n_ε_p)
    σ_dev_tr = Idev * σ_tr
    σ_red_tr = σ_tr - n_α_dev
    σ_red_dev_tr = Idev * σ_red_tr

    σ_red_e_tr = sqrt(3/2) * norm(σ_red_dev_tr)

    Φ = σ_red_e_tr - σy - n_κ

    if Φ < 0
        ms = MisesMixedHardMS(to_6comp(n_ε_p), to_6comp(n_α_dev, true),
                                       n_κ, n_μ)
        return σ_tr[1:6], to_6comp(Ee), ms
    else

        # Initial guesses
        σ_dev0 = σ_dev_tr
        α_dev0 = n_α_dev
        κ0 = n_κ
        μ0 = n_μ

        # Takes the unknown state vector and σ_dev_para
        function res_wrapper(x, σ_dev_pr)
            T = promote_type(eltype(x), eltype(σ_dev_pr))
            R = zeros(T, length(x))
            _σ_dev = x[1:9]
            _α_dev = x[10:18]
            _κ = x[19]
            _μ = x[20]
            R_σ, R_α, R_κ, R_Φ = compute_residual(_σ_dev, _α_dev, _κ, _μ, σ_dev_pr, matpar, matstat)
            R[1:9] = R_σ
            R[10:18] = R_α
            R[19] = R_κ
            R[20] = R_Φ
            R
        end

        res_wrapper(x) = res_wrapper(x, σ_dev_tr)


        jac = jacobian(res_wrapper)

        x0 = [σ_dev0; α_dev0; κ0; μ0]

        res = nlsolve(not_in_place(res_wrapper, jac), x0; iterations = 30, store_trace = true, ftol = 1e-7)
        if !NLsolve.converged(res)
            error("No convergence in material routine")
        end

        X = res.zero

        σ_dev_X = X[1:9]
        α_dev_X = X[10:18]
        κ_X = X[19]
        μ_X = X[20]
        σ_X = σ_tr - σ_dev_tr + σ_dev_X
        σ_red_X = σ_X - α_dev_X
        σ_red_dev_X = Idev * σ_red_X
        σ_red_e_X = sqrt(3/2) * norm(σ_red_dev_X)
        ε_p = n_ε_p + 3/2 * μ_X / σ_red_e_X * σ_red_dev_X
        ε_p = to_6comp(ε_p)
        α_dev_X = to_6comp(α_dev_X, true)

        ms = MisesMixedHardMS(ε_p, α_dev_X, κ_X, μ_X)

        function dRdε_wrapper(ε_para)
            σ_ATS = Ee * (ε_para - n_ε_p)
            σ_dev_ATS = Idev * σ_ATS
            res_wrapper(X, σ_dev_ATS)
        end


        J = jac(X)
        dRdε_f = jacobian(dRdε_wrapper)
        dRdε = dRdε_f(ε)
        dXdε = -J \ dRdε
        dσdev_dε = dXdε[1:9, 1:9]

        ATS = K * (Id * Id') + dσdev_dε

       return σ_X[1:6], to_6comp(ATS), ms
    end
end

function compute_residual(σ_dev, α_dev, κ, μ, σ_dev_tr, matpar, matstat)

    @unpack_MisesMixedHardMP matpar
    @unpack_MisesMixedHardMS matstat

    G = E / 2(1 + ν)

    σ_red_dev = σ_dev - α_dev
    σ_red_e = sqrt(3/2) * norm(σ_red_dev)
    σ_red_dev_hat = σ_red_dev / σ_red_e

    R_σ = σ_dev - σ_dev_tr + 3.*G.*μ .* σ_red_dev_hat
    R_κ = κ - n_κ - r * H * μ * (1 - κ / κ_∞)
    R_α = α_dev - to_9comp(n_α_dev, true) - (1-r) .* H .* μ .* (σ_red_dev_hat - 1./α_∞ .* α_dev)
    R_Φ = σ_red_e - σy - κ

    return R_σ, R_α, R_κ, R_Φ
end
