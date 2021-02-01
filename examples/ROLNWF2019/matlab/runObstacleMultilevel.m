% Solve 2D crowd motion problem and store results. 

% clear; close;
clear;
data = load('ObstacleProblem2D.mat');
omegaData = double(data.domain)'; 
omega = omegaData;
mOrig = double(data.n)';

mTarget = [128 128 64];
nLevel  = 4;
doRecompute = true;
alpha = [1 0.01 1 5];

for k=nLevel:-1:1
    m = mTarget(1:2) * 2^(-k+1);
    N = mTarget(3) * 2^(-k+1);
    h = (omega(2:2:end)-omega(1:2:end))./m;
    dt = 1/(N-1);

    resFile = sprintf('results/runObstacleMultilevel-%d-%d-%d.mat',m,N);
    
    rho0t = linearInter(reshape(data.rho0x,mOrig),omegaData,getCellCenteredGrid(omega,m));
    rho1t = linearInter(reshape(data.rho1x,mOrig),omegaData,getCellCenteredGrid(omega,m));
    Qx    = linearInter(reshape(data.Qx,mOrig),omegaData,getCellCenteredGrid(omega,m));

    rho0x = max(rho0t,1e-7);
    rho1x = max(rho1t,1e-7);
    rho0 = reshape(rho0x,m);
    rho1 = reshape(rho1x,m);
    
    % build linear operators and constraints
    [A1,A2,A3,D1,D2,D3] = getLinearOperators([omega 0 1], [m N-1],1);
    D3left = D3(:,1:prod(m));
    D3in = D3(:,prod(m)+1:end);
    D = dt*prod(h)*[D1 D2 D3in];
    q = - dt*prod(h)*(D3left*rho0(:));
    
    fctn = @(x) MFGobjFctn(x, rho0, rho1,repmat(Qx(:),N,1), omega, m,N, A1,A2,A3,alpha);

    xStop = pcg(D'*D,D'*q)+1e-2;
    lamStop = zeros(prod(m)*(N-1),1);
    if k==nLevel
        x0 = xStop;
        lam = lamStop;
    else
        % prolongate from previous level
        x1n = omega(1)+h(1): h(1) : omega(2)-h(1);
        x2n = omega(3)+h(2): h(2) : omega(4)-h(2);
        x3n = dt : dt : 1;
        x1c = omega(1)+h(1)/2 : h(1) : omega(2);
        x2c = omega(3)+h(2)/2 : h(2) : omega(4);
        x3c = dt/2 : dt : 1;
        
        [x1t,x2t,x3t] = ndgrid(x1n,x2c,x3c);
        xf1 = [x1t(:) x2t(:) x3t(:)];
        [x1t,x2t,x3t] = ndgrid(x1c,x2n,x3c);
        xf2 = [x1t(:) x2t(:) x3t(:)];
        [x1t,x2t,x3t] = ndgrid(x1c,x2c,x3n);
        xf3 = [x1t(:) x2t(:) x3t(:)];
        
        xc = getCellCenteredGrid([omega 0 1],[m N-1]);
        
        m1  = linearInter(m1Opt,[omega 0 1]+[1 -1 0 0 0 0]*h(1),xf1(:));
        m2  = linearInter(m2Opt,[omega 0 1]+[0 0 1 -1  0 0]*h(2),xf2(:));
        rho = linearInter(rhoOpt,[omega -dt 1+dt], xf3(:));
        lam = linearInter(lamOpt,[omega 0 1],xc);
        
        x0 = [m1(:); m2(:); rho(:)];
    end
    
    if doRecompute || not(exist(resFile,'file'))
        [xOpt,lamOpt,His] = PrimalDualNewton(fctn,x0,D,q,'lam',lam,'lamStop',lamStop,'xStop',xStop,...
            'directSolve',1,'atolPrimal',1e-8,'maxIter',200,'LSreduction',0);
        para = His.para;
        His.para.rho = [];
        save(resFile,'xOpt','lamOpt','His','omega','m','N','alpha')
    else
        load(resFile)
        [~,para] = fctn(xOpt);
    end
    m1Opt  = reshape(xOpt(1:size(A1,2)),[],m(2),N-1);
    m2Opt  = reshape(xOpt(numel(m1Opt)+(1:size(A2,2))),m(1),[],N-1);
    rhoOpt = reshape([rho0(:); xOpt(end-prod(m)*(N-1)+1:end);],m(1),m(2),[]);
    lamOpt = reshape(lamOpt,m(1),m(2),[]);
end


%%

m1Opt  = reshape(xOpt(1:size(A1,2)),[],m(2),N-1);
m2Opt  = reshape(xOpt(numel(m1Opt)+(1:size(A2,2))),m(1),[],N-1);
rhoOpt = reshape([rho0(:); xOpt(end-prod(m)*(N-1)+1:end);],m(1),m(2),[]);
%% compute velocities


x1n = linspace(omega(1),omega(2),m(1)+1);
x2n = linspace(omega(3),omega(4),m(2)+1);
x3n = linspace(0,1,N);
x1c = omega(1)+h(1)/2 : h(1) : omega(2);
x2c = omega(3)+h(2)/2 : h(2) : omega(4);
x3c = dt/2 : dt : 1;

[x1t,x2t,x3t] = ndgrid(x1n,x2c,x3c);
xf1 = [x1t(:) x2t(:) x3t(:)];
[x1t,x2t,x3t] = ndgrid(x1c,x2n,x3c);
xf2 = [x1t(:) x2t(:) x3t(:)];


rhoOptf1 = reshape(linearInter(rhoOpt,[omega,-dt/2,1+dt/2],xf1(:)),m(1)+1,m(2),[]);
rhoOptf2 = reshape(linearInter(rhoOpt,[omega,-dt/2,1+dt/2],xf2(:)),m(1),m(2)+1,[]);

v1Opt = zeros(m(1)+1,m(2),N-1);
v1Opt(2:end-1,:,:) = m1Opt ./ rhoOptf1(2:end-1,:,:);

v2Opt = zeros(m(1),m(2)+1,N-1);
v2Opt(:,2:end-1,:) = m2Opt ./ rhoOptf2(:,2:end-1,:);
save(resFile,'v2Opt','v1Opt','rhoOpt','rho0','rho1','-append')
return
%% show velocities
m1center = A1*m1Opt(:); 
m2center = A2*m2Opt(:); 
rhocenter = A3*rhoOpt(:);
v1center = reshape(m1center./rhocenter,[],N-1);
v2center = reshape(m2center./rhocenter,[],N-1); 

fig = figure(); clf;
fig.Name = 'Optimal Transport as MFG';
for k=1:N-1
    subplot(3,N-1,k)
    viewQuiver([v1center(:,k); v2center(:,k)],omega,m)    ;
    axis equal tight
    title(sprintf('v(:,%d)',k));
end
%%
subplot(3,N-1, N:2*(N-1))
imgmontage(rhoOpt, [omega 0 1], [m N],'framesx',1);
title(sprintf('density'))
%%
subplot(3,N-1,2*N:3*(N-1))
imgmontage(lamOpt, [omega 0 1], [m N-1],'framesx',1);
title(sprintf('potential'))
colorbar

return
%% show characteristics
jRes = load('Obstacle-BFGSHighSamples-d-2-iter500.mat')
y0 = jRes.charFwd(1:2,:,1)';
charOT = computeCharacteristics(v1center,v2center,omega,m,linspace(0,1,20),y0(:));
save(resFile,'charOT','-append')

figure(1); clf;
subplot(1,3,1);
viewImage2Dsc(rho0,omega,m);
hold on;
plot(squeeze(charOT(1,:,1)),squeeze(charOT(2,:,1)),'.r','markersize',20);
plot(squeeze(charOT(1,:,end)),squeeze(charOT(2,:,end)),'.r','markersize',20);
for k=1:size(charOT,2)
    plot(squeeze(charOT(1,k,:)),squeeze(charOT(2,k,:)),'-r','linewidth',2);
%      plot(squeeze(jRes.charFwd(1,k,:)),squeeze(jRes.charFwd(2,k,:)),'--w','linewidth',2);
end
%  plot(squeeze(jRes.charFwd(1,:,end)),squeeze(jRes.charFwd(2,:,end)),'.w','markersize',20);
hold off
title('rho0 + characteristics')

subplot(1,3,2);
viewImage2Dsc(rhoOpt(end-prod(m)+1:end),omega,m);
caxis([min(rho1(:)) max(rho1(:))])
colorbar

subplot(1,3,3);
viewImage2Dsc(jRes.rho0z,omega,[128 128]);
title('rho1')
caxis([min(rho1(:)) max(rho1(:))])
colorbar