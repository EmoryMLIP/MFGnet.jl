"""
    function jInv.Vis.viewImage2D
    visualizes 2D image on mesh
    Input:
    I - image data
    M - 2 dimensional AbstractTensorMesh
    kwargs get piped to pcolormesh
"""
function viewImage2D(I::Array,M::AbstractTensorMesh;kwargs...)
	if M.dim!=2
		error("viewImage2D supports only 2-dimensional images")
	end
	I = reshape(I,tuple(M.n...))
	x1,x2 = getCellCenteredAxes(M)
	ph =  heatmap(x1,x2,I';kwargs...)
    yaxis!((M.domain[3],M.domain[4]))
    xaxis!((M.domain[1],M.domain[2]))
    return ph
end



function montageArray(Img::Array{R};ncol=Int(ceil(sqrt(size(Img,3)))),kwargs...) where R <: Real
    (m1,m2,m3) = size(Img)
    nrow = Int(ceil(m3/ncol))
    C = zeros(R,m1*nrow, m2*ncol)
    for k1=1:nrow
        for k2=1:ncol
            if k1+(k2-1)*nrow > m3
                break
            else
                C[(k1-1)*m1+1:(k1*m1), (k2-1)*m2+1:(k2*m2)] = Img[:,:,k1+(k2-1)*nrow]
            end
        end
    end
    Mn = getRegularMesh([0.5; ncol+.5; .5; nrow+.5],[size(C,1); size(C,2)])
    ph = viewImage2D(C, Mn ;kwargs...)
    xticks!(1:ncol)
    ylab = Array{String}(undef,ncol)
    for k=1:ncol
        ylab[k] = @sprintf("%d",1+(k-1)*ncol)
    end
    yticks!(1:nrow,ylab)
    return ph
end

"""
function plotGrid
    plots nodal and cell-centered grids
    Requirement: PyPlot must be installed.
    Usage:
    plotGrid(M::AbstractTensorMesh) - plots nodal grid M
    plotGrid(x,M)                   - plots grid defined by x (nodal or cc)
    plotGrid is based on PyPlot.plot and PyPlot.plot3D. All keywords from those functions
    are available and forwarded to those methods.
"""
function plotGrid(y,M;spacing=[1,1,1],color="blue",kwargs...)

    if length(y)==M.dim*prod(M.n)
        nn = M.n
    elseif length(y)==M.dim*prod(M.n .+ 1)
        nn = M.n .+ 1
    else
        error("plotGrid - unknown grid type, length(y)=$(length(y)), n=$(M.n), dim=$(M.dim)")
    end
    Y = reshape(y,(prod(nn),M.dim))
    yi(d) = reshape(Y[:,d],tuple(nn...))
    if M.dim==2
        J1 = 1:spacing[1]:nn[1]; y1 = reshape(yi(1),tuple(nn...));
        J2 = 1:spacing[2]:nn[2]; y2 = reshape(yi(2),tuple(nn...));
        p1 = plot!(y1[:,J2],y2[:,J2]; color=color,legend=nothing,  kwargs...);
        p2 = plot!(y1[J1,:]',y2[J1,:]';color=color,legend=nothing,  kwargs...);
    elseif M.dim==3
        pt1 = p3!(yi(1),yi(2),yi(3),nn,[1 2 3],spacing; color=color,legend=nothing, kwargs...)
        pt2 = p3!(yi(1),yi(2),yi(3),nn,[2,1,3],spacing; color=color,legend=nothing, kwargs...);
        pt3 = p3!(yi(1),yi(2),yi(3),nn,[3,1,2],spacing; color=color,legend=nothing, kwargs...);
    else
        error("plotGrid - cannot handle $(M.dim)-dimensional grids")
    end
end

"""
function KDEGaussian
    evaluates Kernel Density Estimator  using Gaussian as underlying distribution
    used for plotting rho(x,t) for intermediate ts
"""


function KDEGaussian(x,data;c=1,normalize=true)
    # Kernel Density Estimator for higher-dimensional problems using Gaussian Distribution
    # evaluates f̂(x) = 1/n * Σ_{i=1}^n (K(x-data[:,n])),
    # where K(x) = 1/(2pi^(d/2)*det(H)) * exp(-0.5*x'*H^(-1)*x)

    # inputs:
    #   - x: vectors of size (1,n) to be evaluated
    #   - data: points to create density estimator (size(data) = (1,nData))
    #   - c: constant to multiply (normalized covariance)
    # outputs:
    #   - y: KDE evaluated at x

    # create standard Gaussian
    # r(x) = 1/(2pi^(d/2)*det(H)) * exp(-0.5*x'*H^(-1)*x)

    # assumes data is of size (d x n), d = dimension
    H = cov(data')
    # H = H./maximum(eigvals(H)) # normalize
    if normalize==true
        H = H./maximum(eigvals(H))
    end
    H = H*c # constant

    d = size(x,1) # dimension of data

    r(x,H)  = (1.0/((2*pi)^(d/2)*sqrt(det(H)))) * (exp.(-0.5*(sum((H\x).*x, dims=1))))

    kernel(x) = r((x.-data),H)
    fHat(x)   = mean(kernel(x))

    n = size(x,2)
    fHatVec = zeros(1,n)
    for i=1:n
        fHatVec[i] = fHat(x[:,i])
    end

    return fHatVec

end
