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
end

"""
    getRegularMesh(domain, n)

Create a regular mesh on a rectangular domain.

# Arguments
- `domain`: [xmin, xmax, ymin, ymax] - domain bounds
- `n`: [nx, ny] - number of cells in each dimension

# Returns
- `RegularMesh` object
"""
function getRegularMesh(domain::Vector{T}, n::Vector{Int}) where T<:Real
    @assert length(domain) == 4 "domain must be [xmin, xmax, ymin, ymax]"
    @assert length(n) == 2 "n must be [nx, ny]"
    @assert all(n .> 0) "Number of cells must be positive"

    hx = (domain[2] - domain[1]) / n[1]
    hy = (domain[4] - domain[3]) / n[2]
    h = [hx, hy]

    return RegularMesh(domain, n, h)
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
