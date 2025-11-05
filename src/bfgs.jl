"""
    armijo(f, fk, dfk, xk, pk; t=1.0, maxIter=10, c1=1e-4, b=0.5)

Backtracking Armijo line search

Finds step size t satisfying: f(xk + t*pk) ≤ fk + t*c1*⟨dfk, pk⟩

# Arguments
- `f`: Objective function (returns value and gradient)
- `fk`: Current function value f(xk)
- `dfk`: Current gradient ∇f(xk)
- `xk`: Current iterate
- `pk`: Search direction
- `t`: Initial step size (default: 1.0)
- `maxIter`: Maximum backtracking iterations
- `c1`: Armijo parameter (sufficient decrease)
- `b`: Backtracking factor (multiplicative reduction)

# Returns
- `t`: Step size satisfying Armijo condition
- `LS`: Number of line search iterations (-1 if failed)
"""
function armijo(f::Function,fk,dfk,xk,pk;t=1.0, maxIter=10, c1=1e-4,b=0.5)
    LS = 1

    while LS<=maxIter
        if f(xk+t*pk)[1] <= fk + t*c1*dot(dfk,pk)
            break
        end
        t *= b
        LS += 1
    end
    if LS>maxIter
    	LS= -1
    	t = 0.0
    end
    return t,LS
end


"""
    bfgs(f, fdf, x; H, maxIter=20, atol=1e-8, out=0, storeInterm=false, lineSearch, cb)

BFGS quasi-Newton method for unconstrained optimization: min_x f(x)

# Arguments
- `f`: Objective function (for line search only)
- `fdf`: Function returning (value, gradient)
- `x::Vector`: Initial iterate
- `H`: Initial inverse Hessian approximation (default: identity)
- `maxIter`: Maximum iterations
- `atol`: Absolute tolerance on ||∇f||
- `out`: Verbosity level (0=final, 1=per-iter, -1=silent)
- `storeInterm`: Store intermediate iterates
- `lineSearch`: Line search function
- `cb`: Callback function called each iteration

# Returns
- `x`: Final iterate
- `flag`: Convergence flag (0=success, -1=maxiter, -3=linesearch fail)
- `his`: Optimization history [f, ||∇f||, LS iters] per iteration
- `X`: Intermediate iterates (if storeInterm=true)
- `H`: Final inverse Hessian approximation
"""
function bfgs(f::Function,fdf::Function,x::Vector;H=Matrix(1.0I,length(x),length(x)), maxIter=20,atol=1e-8,out::Int=0,storeInterm::Bool=false,
	lineSearch::Function=(f,fk,dfk,xk,pk,ak)->armijo(f,fk,dfk,xk,pk,maxIter=30,t=ak),cb::Function=()->())

    his = zeros(maxIter,3)  # his = history: [f_val, grad_norm, linesearch_iters]
    # I   = speye(length(x))
    X   = (storeInterm) ? zeros(length(x),maxIter) : []
    fk,dfk  = fdf(x)

    i = 1; flag = -1; a0 = 1.0
    while i<=maxIter

        his[i,1:2] = [fk norm(dfk)]
        if storeInterm; X[:,i] = x; end;
        if norm(dfk)<atol
            his  = his[1:i,:]
            flag = 0
            break
        end

        # get search direction
        pk    = - H*dfk
        # line search
        ak,his[i,3] = lineSearch(f,fk,dfk,x,pk,a0)
        if his[i,3]==1
            a0 =5*ak
        else
            a0= 2*ak
        end
        cb(i)
        if out>0
             Printf.@printf( "iter=%4d\t|f|=%1.2e\t|df|=%1.2e\tLS=%d\tmuLS=%1.2e\n", i, his[i,1] ,his[i,2] ,his[i,3],ak)
        end
        if his[i,3]==-1
             flag = -3
             his  = his[1:i,:]
             break;
        end
        x    += ak*pk
        fk,dfnew    = fdf(x)
        sk    = ak*pk
        yk    = dfnew - dfk
        if dot(yk,sk)>0 # ensure that approximate Hessians remain positive definite
        	H     = (I - (sk*yk')/dot(sk,yk)) * H * (I - (yk*sk')/dot(sk,yk)) + (sk*sk')/dot(yk,sk)
        else
            if out>0
                println("bfgs detected negative curvature. Resetting Hessian")
                a0=1.0
            end
            H = Matrix(1.0I,length(x),length(x))
        end
        dfk  = dfnew
        i+=1
    end
    i = min(maxIter,i)

    if out>=0
        if flag==-1
            Printf.@printf("bfgs iterated maxIter (=%d) times but reached only atol of %1.2e instead of tol=%1.2e",i,his[i,2],atol)
        elseif flag==-3
            Printf.@printf("bfgs stopped of a line search fail at iteration %d.",i)
        elseif out>1
            Printf.@printf("bfgs achieved desired atol of %1.2e at iteration %d.",atol,i)
        end
    end

    if storeInterm; X = X[:,1:i]; end
    return x,flag,his,X,H
end
