"""
Custom exception types for MFGnet.jl

This module defines structured exception types for better error handling and debugging.
"""

# Export all custom exceptions
export MFGnetException, DimensionMismatchError, InvalidParameterError,
       OptimizationFailure, NumericalInstability

"""
Base type for all MFGnet exceptions
"""
abstract type MFGnetException <: Exception end

"""
    DimensionMismatchError <: MFGnetException

Thrown when array dimensions are incompatible.

# Fields
- `expected::String`: Description of expected dimensions
- `got::String`: Description of actual dimensions
- `context::String`: Where the error occurred
"""
struct DimensionMismatchError <: MFGnetException
    expected::String
    got::String
    context::String
end

function Base.showerror(io::IO, e::DimensionMismatchError)
    print(io, "DimensionMismatchError in $(e.context): ")
    print(io, "expected $(e.expected), got $(e.got)")
end

"""
    InvalidParameterError <: MFGnetException

Thrown when function parameters are invalid.

# Fields
- `parameter::String`: Name of the invalid parameter
- `reason::String`: Why the parameter is invalid
- `context::String`: Where the error occurred
"""
struct InvalidParameterError <: MFGnetException
    parameter::String
    reason::String
    context::String
end

function Base.showerror(io::IO, e::InvalidParameterError)
    print(io, "InvalidParameterError in $(e.context): ")
    print(io, "parameter '$(e.parameter)' is invalid: $(e.reason)")
end

"""
    OptimizationFailure <: MFGnetException

Thrown when optimization algorithm fails to converge.

# Fields
- `algorithm::String`: Name of the optimization algorithm
- `iterations::Int`: Number of iterations performed
- `achieved_tol::Float64`: Tolerance achieved
- `target_tol::Float64`: Target tolerance
- `reason::String`: Additional information about failure
"""
struct OptimizationFailure <: MFGnetException
    algorithm::String
    iterations::Int
    achieved_tol::Float64
    target_tol::Float64
    reason::String
end

OptimizationFailure(algorithm, iterations, achieved_tol, target_tol) =
    OptimizationFailure(algorithm, iterations, achieved_tol, target_tol, "")

function Base.showerror(io::IO, e::OptimizationFailure)
    print(io, "$(e.algorithm) failed to converge after $(e.iterations) iterations ")
    print(io, "(achieved: $(e.achieved_tol), target: $(e.target_tol))")
    if !isempty(e.reason)
        print(io, "\nReason: $(e.reason)")
    end
end

"""
    NumericalInstability <: MFGnetException

Thrown when numerical computations produce invalid results (NaN, Inf).

# Fields
- `operation::String`: Description of the operation that failed
- `values::Vector`: Sample of problematic values
- `context::String`: Where the error occurred
"""
struct NumericalInstability <: MFGnetException
    operation::String
    values::Vector
    context::String
end

function Base.showerror(io::IO, e::NumericalInstability)
    print(io, "NumericalInstability in $(e.context): ")
    print(io, "$(e.operation) produced invalid values: $(e.values[1:min(5, length(e.values))])")
    if length(e.values) > 5
        print(io, "... ($(length(e.values)) total)")
    end
end
