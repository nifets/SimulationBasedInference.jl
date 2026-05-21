
function CommonSolve.solve(
    prob::SimulatorInferenceProblem,
    emcee::typeof(AffineInvariantMCMC.sample);
    storage::SimulationData=SimulationArrayStorage(),
    num_samples = 1000,
    num_chains = 100,
    thinning = 1,
    rng::Random.AbstractRNG=Random.default_rng(),
    solve_kwargs...
)
    # make log density function;
    f_raw = logdensityfunc(prob, storage)
    call_count = Ref(0)
    total_calls = num_chains * num_samples
    f = function(ζ)
        call_count[] += 1
        c = call_count[]
        if c % max(1, total_calls ÷ 20) == 0 || c == 1
            @info "Emcee forward eval $c / $total_calls"
        end
        f_raw(ζ)
    end
    # sample prior and apply transform
    prior_samples = sample(rng, prob.prior, num_chains)
    ζ₀ = reduce(hcat, map(SBI.bijector(prob), prior_samples))
    @info "Emcee starting" num_chains num_samples total_calls
    samples, logprobs = emcee(f, num_chains, ζ₀, num_samples, thinning; rng)
    param_names = labels(prob.u0)
    chains = Chains(transpose(samples), param_names)
    return SimulatorInferenceSolution(prob, emcee, storage, chains)
end
