using Parameters
using ForwardDiff
using NLsolve

import ConstLab.stress
import Base.show

@with_kw immutable ViscoPlastMS <: MatStatus
    n_ε_p::Vector{Float64} = zeros(6)
    n_κ::Float64 = 0.0
    n_μ::Float64 = 0.0
end

@with_kw immutable ViscoPlastMP <: MatParameter
    E::Float64
    ν::Float64
    σy::Float64
    H::Float64
    n::Float64
    σc::Float64
    tstar::Float64
end

const Id = Float64[1, 1, 1, 0, 0, 0]

const Idev6 = 1/3 * Float64[ 2    -1    -1     0     0    0;
                            -1     2    -1     0     0     0 ;
                            -1    -1     2     0     0     0 ;
                             0     0     0     3     0     0 ;
                             0     0     0     0     3     0 ;
                             0     0     0     0     0     3 ];

const Ihalf = diagm([1, 1, 1, 0.5, 0.5, 0.5]);

const diff_cache = ForwardDiffCache()

norm_voigt(σ) = sqrt(σ[1]^2 + σ[2]^2 + σ[3]^2 +
                     2σ[4]^2 + 2σ[5]^2 + 2σ[6]^2)

function stress(ε, dt, matpar::ViscoPlastMP, matstat::ViscoPlastMS)
    @unpack_ViscoPlastMP matpar
    @unpack_ViscoPlastMS matstat

    G = E / 2(1 + ν)
    K = E / 3(1 - 2ν)
    Ee = 2 * G * Ihalf * Idev6 + K * (Id * Id')

    σ_tr = Ee * (ε - n_ε_p)
    σ_dev_tr = Idev6 * σ_tr
    s_tr = sqrt(3/2) * norm_voigt(σ_dev_tr)

    Φ = s_tr - σy - n_κ
    if Φ < 0
        return σ_tr, Ee, ViscoPlastMS(n_ε_p, n_κ, n_μ)
    else
        function res_wrapper(x)
            s_res = x[1]
            κ_res = x[2]
            μ_res = x[3]
            R_s, R_κ, R_μ = compute_residual(s_res, κ_res, μ_res, s_tr, dt, matpar, matstat)
            [R_s; R_κ; R_μ]
        end

        # Initial guesses
        x0 = [s_tr;
              n_κ;
              n_μ]

        jac = jacobian(res_wrapper)

        res = nlsolve(not_in_place(res_wrapper, jac), x0; iterations = 20, ftol=1e-5, store_trace = true)
        X = res.zero::Vector{Float64}

        s = X[1]
        κ = X[2]
        μ = X[3]


        σ_dev = 1 / (1 + 3G * μ / s) * σ_dev_tr
        σ = σ_tr - σ_dev_tr + σ_dev
        ε_p = n_ε_p + 3 / 2s * μ * σ_dev

        # ATS
        G_primprim =  n / σc * (Φ / σc)^(n-1)
        h_at = 3 * G + H + 1 / G_primprim * tstar / dt
        b = (3 * G * μ / s) / (1 + 3 * G * μ / s)
        Q = Idev6*Ihalf  - (3/(2*s^2) * (σ_dev * σ_dev'))
        ATS = Ee + (- 2 * G * b * Q - 9 * G^2 / (h_at * s^2) * (σ_dev * σ_dev'))

       return σ, ATS, ViscoPlastMS(ε_p, κ, μ)
    end
end

function compute_residual(s, κ, μ, s_tr, dt, matpar, matstat)
    @unpack_ViscoPlastMP matpar
    @unpack_ViscoPlastMS matstat

    G = E / 2(1 + ν)
    Φ = s - σy - κ
    G_primprim =  n / σc * (max(0, Φ) / σc)^(n-1);

    R_s = s - s_tr + 3G*μ

    R_κ = κ - n_κ - H * μ

    G_prim = (max(0, Φ) / σc)^n;
    R_μ = G_prim - tstar / dt * μ

    return R_s / 3G, R_κ / H, R_μ / G_primprim
end