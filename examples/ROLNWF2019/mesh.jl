"""
Simple mesh generation utilities to replace jInv.Mesh dependency

Provides basic mesh generation functions used in the experiments.
"""

"""
    RegularMesh

Simple regular mesh structure with domain and number of cells
"""
struct RegularMesh{T<:Real}
    domain::Vector{T}  # [xmin, xmax, ymin, ymax]
    n::Vector{Int}     # [nx, ny] number of cells
    h::Vector{T}       # [hx, hy] cell sizes
    dim::Int           # spatial dimension (always 2 for this implementation)
end

"""
    getRegularMesh(domain, n)

Create a regular mesh on a rectangular domain.

# Arguments
- `domain`: [xmin, xmax, ymin, ymax] - domain bounds (Vector or Matrix)
- `n`: [nx, ny] - number of cells in each dimension (Vector, can have floats which will be converted to Int)

# Returns
- `RegularMesh` object
"""
function getRegularMesh(domain::Union{Vector{T},Matrix{T}}, n::Union{Vector,Matrix}) where T<:Real
    # Convert domain to vector if it's a matrix
    domain_vec = vec(domain)
    @assert length(domain_vec) == 4 "domain must be [xmin, xmax, ymin, ymax]"

    # Convert n to Int vector
    n_vec = Int.(vec(n))
    @assert length(n_vec) == 2 "n must be [nx, ny]"
    @assert all(n_vec .> 0) "Number of cells must be positive"

    hx = (domain_vec[2] - domain_vec[1]) / n_vec[1]
    hy = (domain_vec[4] - domain_vec[3]) / n_vec[2]
    h = [hx, hy]
    dim = 2

    return RegularMesh(domain_vec, n_vec, h, dim)
end

"""
    getCellCenteredGrid(M::RegularMesh)

Get cell-centered grid points from a regular mesh.

# Arguments
- `M`: RegularMesh object

# Returns
- Matrix of size (2, nx*ny) with cell center coordinates [x; y]
"""
function getCellCenteredGrid(M::RegularMesh{T}) where T<:Real
    nx, ny = M.n
    hx, hy = M.h
    xmin, xmax, ymin, ymax = M.domain

    # Cell centers
    x = range(xmin + hx/2, xmax - hx/2, length=nx)
    y = range(ymin + hy/2, ymax - hy/2, length=ny)

    # Create grid
    X = repeat(x, 1, ny)
    Y = repeat(y', nx, 1)

    return [vec(X)'; vec(Y)']
end

"""
    getCellCenteredAxes(M::RegularMesh)

Get cell-centered axes for plotting.

# Arguments
- `M`: RegularMesh object

# Returns
- Tuple (x, y) where x and y are the cell center coordinates along each axis
"""
function getCellCenteredAxes(M::RegularMesh{T}) where T<:Real
    nx, ny = M.n
    hx, hy = M.h
    xmin, xmax, ymin, ymax = M.domain

    x = range(xmin + hx/2, xmax - hx/2, length=nx)
    y = range(ymin + hy/2, ymax - hy/2, length=ny)

    return (x, y)
end

"""
    getFaceGrids(M::RegularMesh)

Get face-centered grid points (staggered grid) from a regular mesh.

# Arguments
- `M`: RegularMesh object

# Returns
- Tuple (X1, X2) where X1 contains x-face centers and X2 contains y-face centers
"""
function getFaceGrids(M::RegularMesh{T}) where T<:Real
    nx, ny = M.n
    hx, hy = M.h
    xmin, xmax, ymin, ymax = M.domain

    # X-faces (vertical faces): (nx+1) × ny points
    x1 = range(xmin, xmax, length=nx+1)
    y1 = range(ymin + hy/2, ymax - hy/2, length=ny)
    X1 = repeat(x1, 1, ny)
    Y1 = repeat(y1', nx+1, 1)

    # Y-faces (horizontal faces): nx × (ny+1) points
    x2 = range(xmin + hx/2, xmax - hx/2, length=nx)
    y2 = range(ymin, ymax, length=ny+1)
    X2 = repeat(x2, 1, ny+1)
    Y2 = repeat(y2', nx, 1)

    return ([vec(X1)'; vec(Y1)'], [vec(X2)'; vec(Y2)'])
end
