"""
Simple mesh utilities for regular grids
Replaces jInv.Mesh dependency with basic Julia implementations
"""

export getRegularMesh, getCellCenteredGrid, getCellCenteredAxes

"""
    RegularMesh

Simple regular mesh structure for d-dimensional rectangular domains.

Fields:
- `domain::Vector`: Domain bounds [x1min, x1max, x2min, x2max, ..., xdmin, xdmax]
- `n::Vector{Int}`: Number of cells in each dimension
- `dim::Int`: Spatial dimension
"""
struct RegularMesh{R<:Real}
    domain::Vector{R}
    n::Vector{Int}
    dim::Int
end

"""
    getRegularMesh(domain, n)

Create a regular mesh on a rectangular domain.

# Arguments
- `domain::Vector`: Domain bounds [x1min, x1max, x2min, x2max, ..., xdmin, xdmax]
- `n::Vector{Int}`: Number of cells in each dimension [n1, n2, ..., nd]

# Returns
- `RegularMesh`: A regular mesh structure

# Example
```julia
# 2D domain [-1, 1] × [-1, 1] with 64×64 cells
M = getRegularMesh([-1.0, 1.0, -1.0, 1.0], [64, 64])
```
"""
function getRegularMesh(domain::Vector{R}, n::Vector{Int}) where R<:Real
    d = length(n)
    @assert length(domain) == 2*d "Domain must have 2d entries for d-dimensional mesh"
    return RegularMesh{R}(domain, n, d)
end

# Allow Float64 domain with Int vector for n
getRegularMesh(domain::Vector{R}, n::Vector) where R<:Real = getRegularMesh(domain, Int.(n))

# Allow Matrix (row vector) for domain
getRegularMesh(domain::Matrix{R}, n::Vector) where R<:Real = getRegularMesh(vec(domain), Int.(n))

"""
    getCellCenteredGrid(M::RegularMesh)

Get the cell-centered grid points for a regular mesh.

# Arguments
- `M::RegularMesh`: The mesh structure

# Returns
- `Matrix`: N × d matrix where each row is a grid point (N = total number of cells, d = dimension)

# Example
```julia
M = getRegularMesh([-1.0, 1.0, -1.0, 1.0], [4, 4])
X = getCellCenteredGrid(M)  # Returns 16×2 matrix
```
"""
function getCellCenteredGrid(M::RegularMesh{R}) where R<:Real
    d = length(M.n)

    # Create 1D cell-centered grids for each dimension
    grids_1d = Vector{Vector{R}}(undef, d)
    for i in 1:d
        xmin = M.domain[2*i-1]
        xmax = M.domain[2*i]
        ni = M.n[i]
        h = (xmax - xmin) / ni
        # Cell centers: xmin + h/2, xmin + 3h/2, ..., xmax - h/2
        grids_1d[i] = range(xmin + h/2, xmax - h/2, length=ni) |> collect
    end

    # Create meshgrid - return N×d matrix (each row is a point)
    if d == 1
        # 1D case - return column vector
        return reshape(grids_1d[1], :, 1)
    elseif d == 2
        # 2D case
        x = grids_1d[1]
        y = grids_1d[2]
        nx, ny = length(x), length(y)

        # Create (nx*ny)×2 matrix
        # Order: iterate over y for each x (column-major for reshaping)
        X = zeros(R, nx*ny, 2)
        idx = 1
        for j in 1:ny
            for i in 1:nx
                X[idx, 1] = x[i]
                X[idx, 2] = y[j]
                idx += 1
            end
        end
        return X
    else
        # General d-dimensional case
        # Create all combinations using Cartesian product
        ntotal = prod(M.n)
        X = zeros(R, ntotal, d)

        # Use CartesianIndices for general dimension
        cart_indices = CartesianIndices(Tuple(M.n))
        for (linear_idx, cart_idx) in enumerate(cart_indices)
            for dim in 1:d
                X[linear_idx, dim] = grids_1d[dim][cart_idx[dim]]
            end
        end
        return X
    end
end

"""
    getCellCenteredAxes(M::RegularMesh)

Get 1D arrays of cell-centered coordinates for each dimension.

# Arguments
- `M::RegularMesh`: The mesh structure

# Returns
- `Tuple`: Tuple of vectors (x1, x2, ..., xd) where each vector contains cell centers for that dimension

# Example
```julia
M = getRegularMesh([-1.0, 1.0, -1.0, 1.0], [4, 4])
x, y = getCellCenteredAxes(M)  # Returns two vectors of cell centers
```
"""
function getCellCenteredAxes(M::RegularMesh{R}) where R<:Real
    d = length(M.n)

    axes = Vector{Vector{R}}(undef, d)
    for i in 1:d
        xmin = M.domain[2*i-1]
        xmax = M.domain[2*i]
        ni = M.n[i]
        h = (xmax - xmin) / ni
        axes[i] = range(xmin + h/2, xmax - h/2, length=ni) |> collect
    end

    return Tuple(axes)
end
