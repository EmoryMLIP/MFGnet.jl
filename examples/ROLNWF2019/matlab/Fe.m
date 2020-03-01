function [F,dF,d2F] = Fe(rho,h,dt)

if nargin==0
    runMinimalExample;
    return
end
dF = []; d2F = [];

F = dt*prod(h)*rho'*log(rho);

if nargout>1
    dF = dt*prod(h)* (1 + log(rho))';
end

if nargout>2
    d2F = sdiag(dt*prod(h)./rho);
end



function runMinimalExample

n    = [16 16];
h    = 1./n;
rho = rand(n); rho = rho/(prod(h)*sum(rho(:)));

[F,dF,d2F] = feval(mfilename,rho(:),h);

f = @(rho) Fe(rho,h);
checkDerivative(f,rho(:))
