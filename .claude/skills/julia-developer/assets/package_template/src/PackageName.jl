module PackageName

# Imports
using LinearAlgebra

# Exports
export myfunction

"""
    myfunction(x::AbstractArray)

Brief description of what this function does.

# Arguments
- `x::AbstractArray`: Input array

# Returns
- `result`: Description of output

# Examples
```jldoctest
julia> myfunction([1, 2, 3])
6
```
"""
function myfunction(x::AbstractArray)
    return sum(x)
end

end # module PackageName
