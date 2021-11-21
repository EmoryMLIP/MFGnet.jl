function char = computeCharacteristics(v1,v2,omega,m,tspan,y0)

np = numel(y0)/2;
v1f = @(t,x) linearInter(reshape(v1,m(1),m(2),[]),[omega 0 1],[x(:);t*ones(np,1)]);
v2f = @(t,x) linearInter(reshape(v2,m(1),m(2),[]),[omega 0 1],[x(:);t*ones(np,1)]);
odefun = @(t,x) [v1f(t,x);v2f(t,x)];

[t,y] = ode45(odefun,tspan,y0(:));
char = reshape(y',np,[],numel(tspan));
char = permute(char,[2 1 3]);
