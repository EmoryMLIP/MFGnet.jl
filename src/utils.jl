export evalObj, evalObjAndGrad


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
    Jc,back = Zygote.Tracker.forward(()->J(parms), ps)
    gc = back(1)

    dJ = Θ .* 0.0
    cnt = 0;
    for p in ps
        gp = vec(gc[p].data)
        dJ[cnt+1:cnt+length(gp)] = gp
        cnt +=length(gp)
    end
    return Jc.data,dJ
end

function evalObj(J,Θ::Vector,parms,ps)
    # put theta into vectors
    parms = vec2param!(Θ,parms)
    Jc = J(parms)
    return Jc.data
end
