"""
    GaussianLikelihood(obs, data, prior, name)

Alias for `SimulatorLikelihood(Normal, obs, data, prior, name)`. Represents
a univariate Guassian likelihood.
"""
GaussianLikelihood(
    obs,
    data,
    prior=NamedProductPrior(σ=Exponential(1.0)),
    name=nameof(obs)
) = SimulatorLikelihood(Normal, obs, data, prior, name)

"""
    IsotropicGaussianLikelihood(obs, data, prior, name)

Alias for `SimulatorLikelihood(IsoNormal, obs, data, prior, name)`. Represents
a multivaraite i.i.d Guassian likelihood with homoscedastic noise.
"""
IsotropicGaussianLikelihood(
    obs,
    data,
    prior=NamedProductPrior(σ=Exponential(1.0)),
    name=nameof(obs)
) = SimulatorLikelihood(IsoNormal, obs, data, prior, name)

"""
    DiagonalGaussianLikelihood(obs, data, prior, name)

Alias for `SimulatorLikelihood(DiagNormal, obs, data, prior, name)`. Represents
a multivaraite i.i.d Guassian likelihood with heteroscedastic noise.
"""
DiagonalGaussianLikelihood(
    obs,
    data,
    prior=NamedProductPrior(σ=filldist(Exponential(1.0), prod(size(obs)))),
    name=nameof(obs)
) = SimulatorLikelihood(DiagNormal, obs, data, prior, name)


function predictive_distribution(lik::SimulatorLikelihood{Normal}, σ)
    μ = getvalue(lik.obs)[1]
    return Normal(μ, σ)
end

function predictive_distribution(lik::SimulatorLikelihood{<:MvNormal}, σ)
    μ = vec(getvalue(lik.obs))
    Σ = cov(lik, σ)
    return MvNormal(μ, Σ)
end

Statistics.cov(lik::SimulatorLikelihood{IsoNormal}, σ::Number) = Diagonal(σ^2*ones(prod(size(lik.data))))
Statistics.cov(lik::SimulatorLikelihood{IsoNormal}, σ::AbstractVector) = cov(lik, σ[1])
Statistics.cov(lik::SimulatorLikelihood{DiagNormal}, σ::AbstractVector) = Diagonal(σ.^2)

"""
    FixedCov

Marker distribution type for [`FixedGaussianLikelihood`](@ref): a multivariate
Gaussian whose covariance is known and fixed rather than parameterised by a prior.
"""
struct FixedCov end

"""
    FixedGaussianLikelihood(obs, data, cov, name=nameof(obs))

A multivariate Gaussian likelihood `N(G(θ), Σ)` with a **known, fixed** covariance
`Σ = cov`. Unlike the other Gaussian likelihoods it has no noise parameters: `Σ`
is treated as fixed data, so it contributes nothing to the joint prior, and ensemble
methods read it directly as the observation covariance. Intended for summary-statistic
/ synthetic-likelihood inference where `Σ` is estimated once (e.g. by bootstrap) and
then held fixed. The covariance is stored in the `prior` field but hidden from the
joint prior via `getprior`.
"""
FixedGaussianLikelihood(obs, data, cov, name=nameof(obs)) =
    SimulatorLikelihood(FixedCov, obs, data, cov, name)

# No noise parameters — keep the stored covariance out of the joint prior.
getprior(::SimulatorLikelihood{FixedCov}) = nothing

predictive_distribution(lik::SimulatorLikelihood{FixedCov}) =
    MvNormal(vec(getvalue(lik.obs)), lik.prior)
