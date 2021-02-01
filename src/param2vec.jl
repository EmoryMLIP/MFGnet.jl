

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

lengthvec(Θparm::AbstractArray) = length(Θparm)

function lengthvec(Θparm::Tuple)
    cnt = 0;
    for k=1:length(Θparm)
        cnt+= lengthvec(Θparm[k])
    end
    return cnt
end

getParmsType(Θ::AbstractArray) = typeof(Θ[1])
getParmsType(Θ::Tuple)        = getParmsType(Θ[1])

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
