function [F,dF,d2F] = Fp(rho,Q,h,dt)

if nargin==0
    runMinimalExample;
    return
end
dF = []; d2F = [];

F = dt*prod(h)*rho'* Q;

if nargout>1
    dF = dt*prod(h)* Q';
end

if nargout>2
    d2F = sparse(numel(rho),numel(rho));
end


function runMinimalExample

n    = [16 16];
h    = 1./n;
rho = rand(n); rho = rho/(prod(h)*sum(rho(:)));
Q = ones(numel(rho),1);

[G,dG,d2G] = feval(mfilename,rho(:),Q,h);

f = @(rho) Gkl(rho,Q,h);
checkDerivative(f,rho(:))
