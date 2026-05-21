module SimulationBasedInferenceEmceeExt

using SimulationBasedInference
import SimulationBasedInference: logdensityfunc
import MCMCChains: Chains

import AffineInvariantMCMC
import CommonSolve
import Random

include("emcee.jl")

end