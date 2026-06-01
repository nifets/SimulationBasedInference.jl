using JLD2: jldopen

"""
    SimulationData{inputType, outputType}

Base type representing storage for simulation data, i.e. input/output pairs.
"""
abstract type SimulationData{inputType, outputType} end

"""
    store!(::SimulationData, x, y)

Stores the input/output pair x/y in the given forward map storage container.
"""
function store! end

function getinputs end
function getoutputs end
function getmetadata end

getinputs(s::SimulationData, i) = getinputs(s)[i]
getoutputs(s::SimulationData, i) = getoutputs(s)[i]
getmetadata(s::SimulationData, i) = getmetadata(s)[i]

Base.length(storage::SimulationData) = length(getinputs(storage))
Base.lastindex(storage::SimulationData) = length(storage)
Base.firstindex(storage::SimulationData) = 1
Base.getindex(s::SimulationData, i) = (getinputs(s, i), getoutputs(s, i), getmetadata(s, i))
Base.iterate(s::SimulationData, state=1) =
    state <= length(s) ? (s[state], state + 1) : nothing

"""
    SimulationArrayStorage <: SimulationData

Simple implementation of `SimulationData` that stores all results in generically
typed `Vector`s.
"""
struct SimulationArrayStorage{inputType, outputType, metaType} <: SimulationData{inputType, outputType}
    inputs::Vector{inputType}
    outputs::Vector{outputType}
    metadata::Vector{metaType}
end

SimulationArrayStorage(;
    input_type::Type = Any,
    output_type::Type = Any,
    metadata_type::Type = Any
) = SimulationArrayStorage(input_type[], output_type[], metadata_type[])

getinputs(storage::SimulationArrayStorage) = storage.inputs
getoutputs(storage::SimulationArrayStorage) = storage.outputs
getmetadata(storage::SimulationArrayStorage) = storage.metadata

function store!(storage::SimulationArrayStorage, x, y; attr...)
    push!(storage.inputs, x)
    push!(storage.outputs, y)
    push!(storage.metadata, (; attr...))
end

function clear!(storage::SimulationArrayStorage)
    resize!(storage.inputs, 0)
    resize!(storage.outputs, 0)
    resize!(storage.metadata, 0)
end

"""
    SimulationFileStorage <: SimulationData

Implementation of `SimulationData` that stores all results to file, using JLD2.
"""
mutable struct SimulationFileStorage <: SimulationData{Any, Any}
    path::String
    count::Int
end

function SimulationFileStorage(path)
    n = isfile(path) ? jldopen(f -> haskey(f, "input") ? length(keys(f["input"])) : 0, path, "r") : 0
    SimulationFileStorage(String(path), n)
end

Base.length(s::SimulationFileStorage) = s.count

function store!(s::SimulationFileStorage, x, y; attr...)
    i = s.count + 1
    jldopen(s.path, "a+") do f
        f["input/$i"]  = x
        f["output/$i"] = y
        f["meta/$i"]   = (; attr...)
    end
    s.count = i
    return s
end

getinputs(s::SimulationFileStorage)   = jldopen(f -> [f["input/$i"]  for i in 1:s.count], s.path, "r")
getoutputs(s::SimulationFileStorage)  = jldopen(f -> [f["output/$i"] for i in 1:s.count], s.path, "r")
getmetadata(s::SimulationFileStorage) = jldopen(f -> [f["meta/$i"]   for i in 1:s.count], s.path, "r")
