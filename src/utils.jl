export evalObj, evalObjAndGrad

"""
    append(A, B)

Concatenate tuples or mixed tuple/non-tuple arguments

Handles all combinations: (tuple,tuple), (tuple,scalar), (scalar,tuple), (scalar,scalar)
"""
function append(A::Tuple,B)
	return (A...,B)
end

function append(A,B)
	return (A,B)
end

function append(A,B::Tuple)
	return(A,B...)
end

function append(A::Tuple,B::Tuple)
	return (A..., B...)
end

"""
    myMap(f, Θ)

Recursively apply function f to nested tuple/array structures

# Examples
```julia
myMap(x -> 2x, (1, (2, 3)))  # Returns (2, (4, 6))
myMap(x -> x.^2, [1 2; 3 4]) # Returns [1 4; 9 16]
```

Used for parameter transformations in nested structures
"""
function myMap(f::Function,Θ::AbstractArray)
    return f(Θ)
end


function myMap(f::Function,Θ::Tuple)
    # Use tuple mapping for type stability instead of Array{Any}
    return map(x -> myMap(f, x), Θ)
end

"""
    evalObjAndGrad(J, Θ, parms, ps)

Evaluate objective function and compute gradients using Zygote AD

# Arguments
- `J`: Objective function (typically MeanFieldGame)
- `Θ::Vector`: Flattened parameter vector
- `parms`: Parameter structure (nested tuples)
- `ps`: Zygote Params object containing trainable parameters

# Returns
- `Jc`: Objective function value
- `dJ`: Gradient vector (same shape as Θ)

# Implementation
Uses Zygote.pullback for reverse-mode AD, then flattens gradient structure to vector
"""
function evalObjAndGrad(J,Θ::Vector,parms,ps)
    # Unflatten vector Θ into nested parameter structure
    parms = vec2param!(Θ,parms)

    # Compute objective and gradient via Zygote pullback
    Jc,back = Zygote.pullback(() -> J(parms), ps)
	gc = back(Zygote.sensitivity(Jc))  # gc = gradient collection (dict-like)

    # Flatten gradient structure back to vector
    dJ = Θ .* 0.0
    cnt = 0;
    for p in ps
		if !isnothing(gc[p])
        	gp = vec(gc[p])  # gp = gradient part for this parameter
			dJ[cnt+1:cnt+length(gp)] = gp
			cnt +=length(gp)
		else
			println("grad was nothing")
		end
    end
    return Jc,dJ
end

"""
    evalObj(J, Θ, parms, ps)

Evaluate objective function without computing gradients

# Arguments
- `J`: Objective function (typically MeanFieldGame)
- `Θ::Vector`: Flattened parameter vector
- `parms`: Parameter structure (nested tuples)
- `ps`: Zygote Params object (unused, for API compatibility)

# Returns
- `Jc`: Objective function value
"""
function evalObj(J,Θ::Vector,parms,ps)
    # Unflatten vector Θ into nested parameter structure
    parms = vec2param!(Θ,parms)
    Jc = J(parms)
    return Jc
end
