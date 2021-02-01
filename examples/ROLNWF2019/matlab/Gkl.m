function [G,dG,d2G] = Gkl(rho,rho1,h)

if nargin==0
    runMinimalExample;
    return
end
dG = []; d2G = [];

G = prod(h)*rho'* (log(rho) - log(rho1));

if nargout>1
    dG = prod(h)* (1 + log(rho) - log(rho1))';
end

if nargout>2
    d2G = sdiag(prod(h)./rho);
end



function runMinimalExample

n    = [16 16];
h    = 1./n;
rho1 = rand(n); rho1 = rho1/(prod(h)*sum(rho1(:)));
rho = rand(n); rho = rho/(prod(h)*sum(rho(:)));

[G,dG,d2G] = feval(mfilename,rho(:),rho1(:),h);

f = @(rho) Gkl(rho,rho1(:),h);
checkDerivative(f,rho(:))
