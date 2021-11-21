% function [A1,A2,A3,D1,D2,D3] = getLinearOperators(omega,m,noFluxBC)
% 
% build linear operators for finite volume discretization
%
% Inputs:
%    omega     - description of 3D domain 
%    m         - number of grid cells
%    noFluxBC  - flag to set no flux boundary conditions, default=0
% 
% Outputs:
%   A1,A2,A3 - staggered-to-cell-center average matrices
%   D1,D2,D3 - staggered-to-cell-center difference matrices

function [A1,A2,A3,D1,D2,D3] = getLinearOperators(omega,m,noFluxBC)

if not(exist('noFluxBC','var')) || isempty(noFluxBC)
    noFluxBC=0;
end


av1 = spdiags(ones(m(1),1)*[1/2 1/2],0:1,m(1),m(1)+1);
av2 = spdiags(ones(m(2),1)*[1/2 1/2],0:1,m(2),m(2)+1);
av3 = spdiags(ones(m(3),1)*[1/2 1/2],0:1,m(3),m(3)+1);
if noFluxBC
    av1 = av1(:,2:end-1);
    av2 = av2(:,2:end-1);
end

A1 = kron(speye(m(3)), kron(speye(m(2)),av1)); 
A2 = kron(speye(m(3)), kron(av2, speye(m(1))));
A3 = kron(av3, kron(speye(m(2)),speye(m(1))));

h = (omega(2:2:end)-omega(1:2:end))./m;
d1 = spdiags(ones(m(1),1)*[-1 1],0:1,m(1),m(1)+1)/h(1);
d2 = spdiags(ones(m(2),1)*[-1 1],0:1,m(2),m(2)+1)/h(2);
d3 = spdiags(ones(m(3),1)*[-1 1],0:1,m(3),m(3)+1)/h(3);
if noFluxBC
    d1 = d1(:,2:end-1);
    d2 = d2(:,2:end-1);
end
D1 = kron(speye(m(3)), kron(speye(m(2)),d1)); 
D2 = kron(speye(m(3)), kron(d2, speye(m(1))));
D3 = kron(d3, kron(speye(m(2)),speye(m(1))));