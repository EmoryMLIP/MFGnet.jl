# # Exploratory Analysis
#
# Template for exploratory data analysis and experimentation in Julia

# ## Setup

using LinearAlgebra
using Statistics
using Random

# Set random seed for reproducibility
Random.seed!(42)

# ## Load Data

# Load or generate your data here
data = rand(100, 5)

# ## Exploratory Analysis

# ### Summary Statistics

println("Data shape: ", size(data))
println("Data type: ", eltype(data))
println("\nColumn means:")
println(mean(data, dims=1))

# ### Visualization (if Plots.jl available)

# using Plots
# plot(data[:, 1], label="Column 1")

# ## Experiments

# Try different approaches here

function experiment_1(x)
    return sum(x.^2)
end

function experiment_2(x)
    return sqrt(sum(x.^2))
end

# Compare results
result1 = experiment_1(data[:, 1])
result2 = experiment_2(data[:, 1])

println("Experiment 1: ", result1)
println("Experiment 2: ", result2)

# ## Performance Benchmarking

using BenchmarkTools

@btime experiment_1($(data[:, 1]))
@btime experiment_2($(data[:, 1]))

# ## Notes and Observations

# Record your findings here:
# - Finding 1: ...
# - Finding 2: ...
# - TODO: ...
