"""
    ABCSMC{NF, DistFn} <: EnsembleInferenceAlgorithm

Approximate Bayesian Computation - Sequential Monte Carlo
"""


@kwdef struct ABCSMC{NF, DistFn} <: EnsembleInferenceAlgorithm
    distance::DistFn
    maxiters::Int = 5
    ε_quantile::NF = 0.5
    batch_size::Int = 64
    max_proposals_per_round::Int = 100_000
    kernel_scale::NF = 2.0
    prior_approx::GaussianApproximationMethod = LaplaceMethod()
end

# ABC doesn't use an observation covariance; bypass the default obs_cov build.
function init(inference_prob::SimulatorInferenceProblem,
              alg::ABCSMC,
              ensalg::Union{Nothing,EnsembleAlgorithm}=EnsembleThreads(),
              solve_args...;
              obs_cov_func = (_...) -> zeros(0, 0),
              kwargs...)
    return invoke(init,
        Tuple{SimulatorInferenceProblem, EnsembleInferenceAlgorithm,
              Union{Nothing,EnsembleAlgorithm}, Vararg},
        inference_prob, alg, ensalg, solve_args...;
        obs_cov_func, kwargs...)
end

mutable struct ABCSMCState{
        NF,
        ensType <: AbstractMatrix{NF},
        meanType <: AbstractVector{NF},
        covType <: AbstractMatrix{NF},
    } <: EnsembleState
    ens::ensType
    obs_mean::meanType
    obs_cov::covType
    weights::Vector{NF}
    distances::Vector{NF}
    ε::NF # threshold for next round
    prior::MvNormal
    iter::Int
    rng::AbstractRNG
end

isiterative(::ABCSMC) = true

get_ensemble(state::ABCSMCState) = state.ens

get_obs_mean(state::ABCSMCState) = state.obs_mean

get_obs_cov(state::ABCSMCState) = state.obs_cov

hasconverged(alg::ABCSMC, state::ABCSMCState) = state.iter >= alg.maxiters

function initialstate(alg::ABCSMC,
                      prior::AbstractSimulatorPrior,
                      ens::AbstractMatrix,
                      obs_mean::AbstractVector,
                      obs_cov::AbstractMatrix;
                      rng::AbstractRNG=Random.default_rng())
    NF = eltype(ens)
    N = size(ens, 2)
    unconstrained_prior = gaussian_approx(alg.prior_approx, prior; rng)
    ABCSMCState(
        ens,
        convert(Vector{NF}, obs_mean),
        convert(Matrix{NF}, obs_cov),
        fill(one(NF)/N, N),
        fill(NF(Inf), N),
        NF(Inf),
        unconstrained_prior,
        0,
        rng
    )
end

function ensemblestep!(solver::EnsembleSolver{<:ABCSMC})
    state = solver.state
    alg = solver.alg
    rng = state.rng
    prev_ens = copy(state.ens)
    weights = state.weights
    NF = eltype(prev_ens)

    θ_size, N = size(prev_ens)

    solver.verbose && @info "ABC-SMC round $(state.iter) starting" N

    if state.iter == 1
        out = ensemble_forward(solver)
        dists = [alg.distance(view(out.pred, :, i), state.obs_mean)
                 for i in axes(out.pred, 2)]
        state.distances = dists
        state.ε = threshold(alg, dists)

        solver.verbose && @info "ABC-SMC round 1 done" ε=state.ε min_d=minimum(dists) med_d=median(dists)
        return out
    end

    kernel = build_kernel(prev_ens, weights, alg.kernel_scale)

    proposal_dist = Categorical(weights ./ sum(weights))

    accepted_θ = Vector{Vector{NF}}(undef, N)
    accepted_d = Vector{NF}(undef, N)
    proposals  = Matrix{NF}(undef, θ_size, alg.batch_size)
    noise  = Matrix{NF}(undef, θ_size, alg.batch_size)

    acc = EnsembleAccumulator()
    n_accepted = 0
    n_proposals = 0
    while n_accepted < N && n_proposals < alg.max_proposals_per_round
        b = min(alg.batch_size, alg.max_proposals_per_round - n_proposals)
        idxs = rand(rng, proposal_dist, b)
        randn!(rng, noise)
        @views proposals[:, 1:b] .= prev_ens[:, idxs] .+ kernel.L * noise[:, 1:b]
        state.ens = proposals[:, 1:b]

        out = ensemble_forward(solver)

        batch_accepted = Int[]
        @views for i in 1:b
            d = alg.distance(out.pred[:, i], state.obs_mean)
            if (d < state.ε)
                n_accepted += 1
                accepted_θ[n_accepted] = proposals[:, i]
                accepted_d[n_accepted] = d
                push!(batch_accepted, i)
                n_accepted >= N && break
            end
        end
        push_accepted!(acc, out, batch_accepted)
        n_proposals += b
        solver.verbose && @info "ABC-SMC round $(state.iter)" accepted=n_accepted proposals=n_proposals ε=state.ε
    end

    n_accepted == 0 && error("ABC-SMC: no particles accepted in round $(state.iter) " *
    "(ε=$(state.ε), proposals=$n_proposals). Loosen ε_quantile " *
    "or raise max_proposals_per_round.")

    if n_accepted < N
        @warn "ABC-SMC round $(state.iter): only $n_accepted/$N accepted after $n_proposals proposals; ensemble shrinking"
    end

    dists = accepted_d[1:n_accepted]
    state.ens = reduce(hcat, accepted_θ[1:n_accepted])
    state.weights = new_weights(state.ens, prev_ens, weights, kernel, state.prior)
    state.ε = threshold(alg, dists)

    push!(solver.logprior, [logpdf(state.prior, view(state.ens, :, i)) for i in 1:n_accepted])

    solver.verbose && @info "ABC-SMC round $(state.iter) done" ε=state.ε min_d=minimum(dists) med_d=median(dists)

    return (; pred=acc.pred, observables=acc.observables)
end


finalize!(::EnsembleSolver{<:ABCSMC}) = nothing

function build_kernel(ens, weights, scale)
    θ_size = size(ens, 1)
    μ_w = ens * weights
    centered = ens .- μ_w
    Σ = scale * Symmetric(centered * Diagonal(weights) * centered')
    Σ = Matrix(Σ) + sqrt(eps(eltype(Σ))) * I
    Σ_chol = cholesky(Symmetric(Σ))
    L = Σ_chol.L
    log_kern_const = -0.5 * (logdet(Σ_chol) + θ_size * log(2π))
    return (; L=L, log_kern_const=log_kern_const, Σ_chol=Σ_chol)
end

function new_weights(new_ens, prev_ens, prev_weights, kernel, prior)
    N_prev = size(prev_ens, 2)
    N_new = size(new_ens, 2)
    NF = eltype(new_ens)
    log_wprev = log.(prev_weights)
    log_w     = Vector{NF}(undef, N_new)
    for i in 1:N_new
        θ′ = view(new_ens, :, i)
        log_π = logpdf(prior, θ′)
        log_kernels = Vector{NF}(undef, N_prev)
        for j in 1:N_prev
            diff = θ′ .- view(prev_ens, :, j)
            log_kernels[j] = kernel.log_kern_const - 0.5 * dot(diff, kernel.Σ_chol \ diff)
        end
        log_w[i] = log_π - logsumexp(log_wprev .+ log_kernels)
    end
    return exp.(log_w .- logsumexp(log_w))
end

function threshold(alg::ABCSMC, dists)
    finite = filter(isfinite, dists)
    isempty(finite) && return eltype(dists)(Inf)
    return quantile(finite, alg.ε_quantile)
end
