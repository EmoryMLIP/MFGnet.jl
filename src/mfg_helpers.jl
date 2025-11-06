# Helper functions for MFG computations

"""
    spatial_dim(U)

Extract spatial dimension from augmented state vector U.

U has structure [x₁,...,xd; logdet; costL; costF; costHJ] where the last 4
components are scalar auxiliary variables.
"""
spatial_dim(U::AbstractArray) = size(U, 1) - 4

"""
    spatial_positions(U)

Extract spatial positions from augmented state vector.

Returns a view (no copying) of the first d rows of U, where d is the spatial dimension.
"""
spatial_positions(U::AbstractArray) = @view U[1:end-4, :]

"""
    symmetrize(A::AbstractMatrix)

Symmetrize matrix A by computing (A + A')/2.

Used for quadratic potential terms in PotentialNN to ensure Hermitian structure.
"""
symmetrize(A::AbstractMatrix) = (A + A') / 2

"""
    affine_transform(K, b, S)

Compute affine transformation K*S + b with proper broadcasting.

Used in neural network layers for the pre-activation computation.
"""
@inline affine_transform(K::AbstractMatrix, b::AbstractVector, S::AbstractArray) = K*S .+ b
