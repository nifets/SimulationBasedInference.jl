module SimulationBasedInferenceFluxExt

# The `Emulators` submodule this extension glued Flux into was removed from core
# SimulationBasedInference, leaving the old FluxModel/Emulator code dangling and
# breaking precompilation whenever Flux is loaded. Neutered to an empty module so
# the (Flux-triggered) extension precompiles; restore real glue here if/when an
# emulator API returns.

end
