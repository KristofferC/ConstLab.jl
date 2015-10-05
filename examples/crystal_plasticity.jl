using Parameters
using ForwardDiff
using NLsolve
using ConstLab

import ConstLab.stress

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
    angles_rad = angles * pi / 180.0
    for α = 1:nslip
        t = angles_rad[α]
        s = [cos(t), sin(t), 0.0]
        m = [cos(t + pi/2), sin(t + pi/2), 0.0]
        sxm = sym(s ⊗ m)

        push!(sxm_sym)
    end
    CrystPlastMP(E, ν, σy, n, H, q, D, tstar, angles, sxm_sym, nslip)
end

const Idev = 1/3 * Float64[  2    -1    -1     0     0     0     0     0     0;
                            -1     2    -1     0     0     0     0     0     0;
                            -1    -1     2     0     0     0     0     0     0;
                             0     0     0     3     0     0     0     0     0;
                             0     0     0     0     3     0     0     0     0;
                             0     0     0     0     0     3     0     0     0;
                             0     0     0     0     0     0     3     0     0;
                             0     0     0     0     0     0     0     3     0;
                             0     0     0     0     0     0     0     0     3];

const Id = Float64[1,1,1,0,0,0,0,0,0]
""
function stress(ε, dt, matpar::CrystPlastMP, matstat::CrystPlastMS)

    @unpack_CrystPlastMP matpar
    @unpack_CrystPlastMS matstat

    G = E / 2(1 + ν)
    K = E / 3(1 - 2ν)
    Ee = 2 * G * Idev + K * (Id * Id')
    σ_tr = Ee * (ε - n_ε_p)

    # Initial guesses
    σ0 = σ_tr
    κ0 = n_κ
    μ0 = n_μ

    function res_wrapper(x)
        σ = x[1:9]
        κ = x[10:10+nslip-1]
        μ = x[10+nslip:end]
        R_σ, R_κ, R_Φ = compute_residual(σ, κ, μ, σ_tr, dt, matpar, matstat)
        [R_σ; R_κ; R_Φ]
    end

    function num_grad(x)
        h = 1e-7
        R = res_wrapper(x)
        J = zeros(length(x), length(x))
        for i in 1:length(x)
            x[i] += h
            Rh = res_wrapper(x)
            J[:,i] = (Rh - R) / h
            x[i] -= h
        end
        return J
    end

    x0 = [σ0;
         κ0;
         μ0]

    jac = jacobian(res_wrapper)

    res = nlsolve(not_in_place(res_wrapper, jac), x0; iterations = 30, store_trace = true, ftol = 1e-6)

     if !NLsolve.converged(res)
            error("No convergence in material routine")
        end

    X = res.zero
    σ_X = X[1:9]
    κ_X = X[10:10+nslip-1]
    μ_X = X[10+nslip:end]
    ε_p = n_ε_p
    τ_X = zeros(nslip)
    for α = 1:nslip
        τ_X[α] = dot(σ_X, sxm_sym[α])
        ε_p += μ_X[α] * sxm_sym[α] * sign(n_τ[α])
    end

    ms = CrystPlastMS(ε_p, κ_X, τ_X, μ_X)
    return σ_X, ms

end


function compute_residual(σ, κ, μ, σ_tr, dt, matpar, matstat)

    @unpack_CrystPlastMP matpar
    @unpack_CrystPlastMS matstat


    R_σ = zeros(eltype(σ), 9)
    R_Φ = zeros(eltype(σ), nslip)
    R_κ = zeros(eltype(σ), nslip)
    Φ = zeros(eltype(σ), nslip)
    τ = zeros(eltype(σ), nslip)

    for α=1:nslip
        R_κ[α] = n_κ[α] - κ[α]
        for β = 1:nslip
            R_κ -= H * μ[β] * (q + (1 - q) * Int(α == β))
        end
        τ[α] = dot(σ, sxm_sym[α])
        Φ[α] = abs(τ[α]) - (κ[α] + σy)
        R_Φ[α] = μ[α] * tstar - dt * max(0, Φ[α]/D)^n
    end

    R_σ[:] = σ - σ_tr
    ep = zeros(9)
    for α = 1:nslip
        ep += μ[α] * sxm_sym[α] * sign(τ[α])
    end
    G = E / 2(1 + ν)
    K = E / 3(1 - 2ν)
    Ee = 2 * G * Idev + K * (Id * Id')
    R_σ += Ee * ep

    return R_σ, R_κ, R_Φ
end

end
