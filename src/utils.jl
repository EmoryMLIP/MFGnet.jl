export evalObj, evalObjAndGrad

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

function myMap(f::Function,Θ::AbstractArray)
    return f(Θ)
end


function myMap(f::Function,Θ::Tuple)
    fΘ = Array{Any}(undef,length(Θ))
    for k=1:length(Θ)
        fΘ[k] = myMap(f,Θ[k])
    end
    return tuple(fΘ...)
end

function evalObjAndGrad(J,Θ::Vector,parms,ps)
    # put theta into vectors
    parms = vec2param!(Θ,parms)
	# Jc = J(parms)
    # back = gradient(()->J(parms), ps)
    Jc,back = Zygote.pullback(() -> J(parms), ps)
	gc = back(Zygote.sensitivity(Jc))
	
    dJ = Θ .* 0.0
    cnt = 0;
    for p in ps
		if gc[p]!=nothing 
        	gp = vec(gc[p])
			dJ[cnt+1:cnt+length(gp)] = gp
			cnt +=length(gp)
		else
			println("grad was nothing")
		end
    end
    return Jc,dJ
end

function evalObj(J,Θ::Vector,parms,ps)
    # put theta into vectors
    parms = vec2param!(Θ,parms)
    Jc = J(parms)
    return Jc
end
