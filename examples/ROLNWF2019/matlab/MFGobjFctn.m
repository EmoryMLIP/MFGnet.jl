% =========================================================================
% function [Jc,para,dJ,H] = MFGobjFctn(x, rho0, rho1, omega, m,N, A1,A2,A3)
%
% objective function for mean field games and control problems
%
% J(x) = alpha(1)* L(x) + alpha(2)*Fe(x) + alpha(3)*Fp(x) + alpha(4)*Gkl(x)
%
% where 
%     x  - (m1,m2,rho) contains the momenta and density
%     L  - transport costs, int_0^T int_Omega |v|^2 rho dx dt
%     Fe - entropy interaction term, int_0^T int_Omega rho*log(rho) dx dt
%     Fp - potential interaction term, int_0^T int_Omega Q(x,t)*rho dx dt
%     Gkl- KL terminal costs
%
%
% Inputs:
%   x       contains momentum (m1, m2) and time (rho) 
%   rho0    initial density (template image)
%   rho1    final density (target image) 
%   Q       potential term (can be empty if alpha(3)==0
%   omega   spatial domain of images
%   m       number of discretization points for image 
%   N       number of time steps (nodal) 
%   A1      averaging matrix for m1 (nodal to cell-centered) 
%   A2      averaging matrix for m2 (nodal to cell-centered)    
%   A3      averaging matrix for rho (nodal to cell-centered)
%   alpha   weights for indivdual terms
%
% Outputs:
%  Jc       current function value J(vc)
%  para     struct {Jc, rho}, for plots
%  dJ       gradient of J
%  H        approximation to Hessian of J
%
% see also MPLDDMMobjFctn, OMTobjFctn 
% =========================================================================
function [Jc,para,dJ,H] = MFGobjFctn(x, rho0, rho1,Q, omega, m,N, A1,A2,A3,alpha)

if nargin==0
    runOMT;
    return
end

if not(exist('alpha','var')) || isempty(alpha)
    alpha = [1 1 1 1];
end

% unpack x -> m1,m2,rho
n   = numel(x);
nm1 =size(A1,2);
nm2 = size(A2,2);
nrho = m(1)*m(2)*(N-1);

% initialize
Fec = 0.0; dFe = zeros(1,n); d2Fe = sparse(n,n);
Fpc = 0.0; dFp = zeros(1,n); d2Fp = sparse(n,n);
Rc  = 0.0;

m1 = x(1:nm1);
m2 = x(nm1+(1:nm2));
rho = reshape([rho0(:); x(nm1+nm2+(1:nrho))],numel(rho0),[]);

dt = 1/(N-1);
h  = (omega(2:2:end)-omega(1:2:end))./m;

e  = ones(m(1)*m(2)*(N-1),1);
% transport cost
Lc = 0.5* dt*prod(h)* e'* ( ( A1*(m1.^2) + A2*(m2.^2)) .* (A3*(1./rho(:))) );
% terminal cost
[Gc,dG,d2G] = Gkl(rho(:,end),rho1(:),h);

if alpha(2)>0
    % interaction costs
    [Fec,dFe,d2Fe] = Fe(rho(:),h,dt);
    dFe = [zeros(1,nm1+nm2) dFe(prod(m)+1:end)];
    d2Fe = blkdiag(sparse(nm1+nm2,nm1+nm2),d2Fe(prod(m)+1:end,prod(m)+1:end));
end
if alpha(3)>0
    % potential costs
    [Fpc,dFp,d2Fp] = Fp(rho(:),Q,h,dt);
    dFp = [zeros(1,nm1+nm2) dFp(prod(m)+1:end)];
    d2Fp = blkdiag(sparse(nm1+nm2,nm1+nm2),d2Fp(prod(m)+1:end,prod(m)+1:end));
end

% compute log-barrier
if any(rho(:)<0)
    Rc = Inf;
end
Jc = alpha(1)*Lc + alpha(2)*Fec + alpha(3)*Fpc +  alpha(4)*Gc + Rc;

para = struct('Jc',Jc,'Lc',alpha(1)*Lc,'Gc',alpha(4)*Gc,'Fec',alpha(2)*Fec,'Fpc',alpha(3)*Fpc,'rho',rho(:),'alpha',alpha);
if nargout>2
    tt =  A3*(1./rho(:));
    dLm = 2*([m1;m2].*([A1'*tt;A2'*tt]));
    dLrho = -diag(sparse(1./(rho(:).^2)))*A3'*[A1,A2]*([m1;m2].^2);
    dL = 0.5*dt*prod(h)*[dLm(:); dLrho(numel(rho0)+(1:nrho))]';    
    dG = [zeros(1,nm1+nm2+m(1)*m(2)*(N-2)) dG];
    dJ = alpha(1)*dL + alpha(2)*dFe + alpha(3)*dFp + alpha(4)*dG;
end
if nargout>3
    d2Lm   = 2* diag(sparse([A1'*tt;A2'*tt]));
    C1   = 2* ((m1.*A1')*A3) * diag(sparse(-1./rho(:).^2));
    C2   = 2* ((m2.*A2')*A3) * diag(sparse(-1./rho(:).^2));
    C = [C1;C2];
    d2Lrho = 2*diag(sparse(A3'*[A1,A2]*([m1;m2].^2)))*diag(sparse(1./rho(:).^3));
    d2Lrho = d2Lrho(numel(rho0)+(1:nrho),numel(rho0)+(1:nrho));
    C = C(:,numel(rho0)+(1:nrho));
    d2L = dt*prod(h)* [d2Lm C; C' d2Lrho];
    
    d2G = blkdiag(sparse(nm1+nm2+m(1)*m(2)*(N-2),nm1+nm2+m(1)*m(2)*(N-2)),d2G);
    
    H = 0.5*alpha(1)*d2L + alpha(2)*d2Fe + alpha(3)*d2Fp + alpha(4)*d2G;
end
