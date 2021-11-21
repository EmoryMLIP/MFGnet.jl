% function q = viewQuiver(V,omega,m,varargin)
%
% view cell-centered vector field using quiver

function q = viewQuiver(vc,omega,m,varargin)

if nargin==0
    runMinimalExample;
    return
end
    

for k=1:2:length(varargin)    % overwrites default parameter
  eval([varargin{k},'=varargin{',int2str(k+1),'};']);
end

h = (omega(2:2:end)-omega(1:2:end))./m;
x1 = omega(1)+h(1)/2:h(1):omega(2)-h(1)/2;
x2 = omega(3)+h(2)/2:h(2):omega(4)-h(2)/2;
[x,y] = ndgrid(x1,x2);
vc = reshape(vc,[],2);

q = quiver(x,y,reshape(vc(:,1),m),reshape(vc(:,2),m));

function runMinimalExample

omega = [-1 1 -1 1];
m = [24 32];
xc = reshape(getCellCenteredGrid(omega,m),[],2);
vc = [xc(:,1); xc(:,2)];
q = viewQuiver(vc,omega,m);