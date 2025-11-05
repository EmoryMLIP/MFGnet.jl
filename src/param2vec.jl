"""
    vec2param!(Θvec, Θparam)

Unflatten vector into nested parameter structure (in-place)

Recursively fills Θparam with values from flat vector Θvec.
Works with nested tuples and arrays.

# Example
```julia
Θ = ((K1, b1), (K2, b2))  # Nested parameter structure
v = param2vec(Θ)          # Flatten to vector
vec2param!(v, Θ)          # Restore structure (in-place)
```
"""
function vec2param!(Θvec,Θparam::AbstractArray)
    Θparam .= reshape(Θvec, size(Θparam))
    return Θparam
end

function vec2param!(Θvec,Θparm::Tuple)
    cnt = 0
    for k=1:length(Θparm)
        nk = lengthvec(Θparm[k])
        vec2param!(Θvec[cnt+1:cnt+nk], Θparm[k])
        cnt+=nk
    end
    return Θparm
end

"""
    lengthvec(Θparm)

Count total number of parameters in nested structure

Recursively traverses tuples to compute total parameter count
"""
lengthvec(Θparm::AbstractArray) = length(Θparm)

function lengthvec(Θparm::Tuple)
    cnt = 0;
    for k=1:length(Θparm)
        cnt+= lengthvec(Θparm[k])
    end
    return cnt
end

# Extract element type from nested structure
getParmsType(Θ::AbstractArray) = typeof(Θ[1])
getParmsType(Θ::Tuple)        = getParmsType(Θ[1])

"""
    param2vec(Θparm)

Flatten nested parameter structure into vector

Converts nested tuples of arrays into single flat vector for optimization.
Inverse of vec2param!.

# Example
```julia
Θ = ((K1, b1), (K2, b2))  # Nested parameters
v = param2vec(Θ)          # v is a flat vector
```
"""
function param2vec(Θparm::Tuple)
    cnt = lengthvec(Θparm)
    R = getParmsType(Θparm)
    Θvec = zeros(R, cnt)
    return param2vec!(Θparm,Θvec)
end

param2vec(Θparm::AbstractArray) = vec(Θparm)
function param2vec!(Θparm,Θvec)
    cnt = 0
    for k=1:length(Θparm)
        nk = lengthvec(Θparm[k])
        Θvec[cnt+1:cnt+nk] .= param2vec(Θparm[k])
        cnt+=nk
    end
    return Θvec
end
