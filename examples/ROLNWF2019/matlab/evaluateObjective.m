% Evaluate mean field objective using the controls obtained with the
% Eulerian and Lagrangian method


clear;

% resFiles = dir('omtResults/runGaussianMixtureMFGOT-32-*.mat'); example = 'OT-Eulerian';
resFiles = dir('omtResults/OMT-Multilevel-noHJB-nt-8-d-2-*.mat'); example = 'OT-Lagrangian';

% resFiles = dir('omtResults/runObstacleMultilevel-*.mat');;example = 'Obstacle-Eulerian';
% resFile = 'runObstacleMultilevel-64-64-32.mat'; example = 'Obstacle-Eulerian';
% resFiles = dir('obstacleResults/Obstacle-Multilevel-d-2-*.mat'); example = 'Obstacle-Lagrangian';
% resFile = 'Obstacle-Multilevel-d-2-level-2-iter-500.mat'; example = 'Obstacle-Lagrangian';


% % setting for finite volume method
m = [256 256];
N = 512;

if not(isempty(strfind(example,'Obstacle')))
    omega = [-3 3 -4 4];
    data = load('ObstacleProblem2D.mat');
    data.rho0x = reshape(data.rho0x,double(data.n)');
    data.rho1x = reshape(data.rho1x,double(data.n)');
    data.Qx = reshape(data.Qx,double(data.n)');
    rho0x = linearInter(data.rho0x,data.domain,getCellCenteredGrid(omega,m));
    rho1x = linearInter(data.rho1x,data.domain,getCellCenteredGrid(omega,m));
    Q = linearInter(data.Qx,data.domain,getCellCenteredGrid(omega,m));
    
else
    omega =[ -5 5 -5 5];
    data = load('OMTProblem2D.mat');
    data.rho0x = reshape(data.rho0x,double(data.n)');
    data.rho1x = reshape(data.rho1x,double(data.n)');
    rho0x = linearInter(data.rho0x,data.domain,getCellCenteredGrid(omega,m));
    rho1x = linearInter(data.rho1x,data.domain,getCellCenteredGrid(omega,m));
end
%%

for k=1:numel(resFiles)
    resFile = fullfile('omtResults',resFiles(k).name)
    switch example
        case 'OT-Lagrangian'
            % OT - MFGnet results
            res = load(resFile);
            omegaV = double(res.domain)';
            mV = double(res.n)';
            hV = (omegaV(2:2:end)-omegaV(1:2:end))./mV;
            NV = size(res.V1,3);
            dtV = 1/(NV-1);
            alpha = [2 0 0 5];
            getV1  = @(x,t)  linearInter(res.V1,[omegaV 0 1] + [-1/2,+1/2,0,0,0,0]*hV(1),[x(:); t*ones(prod(m+[1,0]),1)]);
            getV2  = @(x,t)  linearInter(res.V2,[omegaV 0 1] + [0,0,-1/2,+1/2,0,0]*hV(2),[x(:); t*ones(prod(m+[0,1]),1)]);
            
        case 'OT-Eulerian'
            % OT results
            res = load(resFile);
            omegaV = res.omega;
            mV = res.m;
            hV = (omegaV(2:2:end)-omegaV(1:2:end))./mV;
            NV = res.N;
            dtV = 1/(NV-1);
            v1OptPad = res.v1Opt(:,:,[1,1:end,end]); % pad in time since original V1 is cc.
            v2OptPad = res.v2Opt(:,:,[1,1:end,end]);
            alpha = [2 0 0 5];
            getV1  = @(x,t)  linearInter(v1OptPad,[omegaV -dtV 1+dtV] + [-1/2,+1/2,0,0,0,0]*hV(1),[x(:); t*ones(prod(m+[1,0]),1)]);
            getV2  = @(x,t)  linearInter(v2OptPad,[omegaV -dtV 1+dtV] + [0,0,-1/2,+1/2,0,0]*hV(2),[x(:); t*ones(prod(m+[0,1]),1)]);
        case 'Obstacle-Lagrangian'
            % Obstacle - MFGnet results
            res = load(resFile);
            omegaV = double(res.domain)';
            mV = double(res.n)';
            hV = (omegaV(2:2:end)-omegaV(1:2:end))./mV;
            NV = size(res.V1,3);
            dtV = 1/(NV-1);
            alpha = [1 0.01 1 5];
            getV1  = @(x,t)  linearInter(res.V1,[omegaV 0 1] + [-1/2,+1/2,0,0,0,0]*hV(1),[x(:); t*ones(prod(m+[1,0]),1)]);
            getV2  = @(x,t)  linearInter(res.V2,[omegaV 0 1] + [0,0,-1/2,+1/2,0,0]*hV(2),[x(:); t*ones(prod(m+[0,1]),1)]);
            
            
        case 'Obstacle-Eulerian'
            % % Obstacle results
            res = load(resFile);
            omegaV = res.omega;
            mV = res.m;
            hV = (omegaV(2:2:end)-omegaV(1:2:end))./mV;
            NV = res.N;
            dtV = 1/(NV-1);
            v1OptPad = res.v1Opt(:,:,[1,1:end,end]); % pad in time since original V1 is cc.
            v2OptPad = res.v2Opt(:,:,[1,1:end,end]);
            alpha = [1 0.01 1 5];
            getV1  = @(x,t)  linearInter(v1OptPad,[omegaV -dtV 1+dtV] + [-1/2,+1/2,0,0,0,0]*hV(1),[x(:); t*ones(prod(m+[1,0]),1)]);
            getV2  = @(x,t)  linearInter(v2OptPad,[omegaV -dtV 1+dtV] + [0,0,-1/2,+1/2,0,0]*hV(2),[x(:); t*ones(prod(m+[0,1]),1)]);
        otherwise
            error('result file not found %s',resFile);
    end
    
    h = (omega(2:2:end)-omega(1:2:end))./m;
    dt = 1/N;
    
    %% build grids and function handles for interpolation
    x1n = linspace(omega(1),omega(2),m(1)+1);
    x2n = linspace(omega(3),omega(4),m(2)+1);
    x1c = omega(1)+h(1)/2 : h(1) : omega(2);
    x2c = omega(3)+h(2)/2 : h(2) : omega(4);
    
    [x1t,x2t] = ndgrid(x1c,x2c);
    xc = [x1t(:) x2t(:)];
    [x1t,x2t] = ndgrid(x1n,x2c);
    xf1 = [x1t(:) x2t(:)];
    [x1t,x2t] = ndgrid(x1c,x2n);
    xf2 = [x1t(:) x2t(:)];
    
    getRho = @(rho,x) nnInter(reshape(rho,m),omega,x(:));
    
    % testRho = norm(getRho(rho0x,xc) - rho0x)
    % testV1  = norm(getV1(xf1,0) -reshape( V1(:,:,1),[],1))
    % testV2  = norm(getV2(xf2,0) -reshape( V2(:,:,1),[],1))
    
    %% build linear operators
    av1 = spdiags(ones(m(1),1)*[1/2 1/2],0:1,m(1),m(1)+1);
    av2 = spdiags(ones(m(2),1)*[1/2 1/2],0:1,m(2),m(2)+1);
    
    A1 = kron(speye(m(2)),av1);
    A2 = kron(av2, speye(m(1)));
    
    d1 = spdiags(ones(m(1),1)*[-1 1],0:1,m(1),m(1)+1)/h(1);
    d2 = spdiags(ones(m(2),1)*[-1 1],0:1,m(2),m(2)+1)/h(2);
    D1 = kron(speye(m(2)),d1);
    D2 = kron(d2, speye(m(1)));
    %% forward in time
    rhoFwd = zeros(prod(m),N+1);
    rhoFwd(:,1) = rho0x;
    LcFwd = 0.0;
    FeFwd = 0.0;
    FpFwd = 0.0;
    w = [dt/2; dt*ones(N-1,1);dt/2];  % quadrature weights for trapezoidal rule
    for k = 1:N
        
        tk = (k-1)*dt;
        v1 = getV1(xf1,tk);
        v2 = getV2(xf2,tk);
        rhov1 = 0*v1;
        rhov2 = 0*v2;
        
        % flux in x1 direction
        idp = find(v1>=0);
        rhop = getRho(rhoFwd(:,k), xf1(idp,:)-[h(1)/2,0]); %  rho from left
        rhov1(idp) = v1(idp).*rhop;
        idn = find(v1<0);
        rhon = getRho(rhoFwd(:,k), xf1(idn,:)+[h(1)/2,0]);  %  rho from right
        rhov1(idn) = v1(idn).*rhon;
        
        % flux in x2 direction
        idp = find(v2>=0);
        rhop = getRho(rhoFwd(:,k), xf2(idp,:)-[0,h(2)/2]); %  rho from below
        rhov2(idp) = v2(idp).*rhop;
        idn = find(v2<0);
        rhon = getRho(rhoFwd(:,k), xf2(idn,:)+[0,h(2)/2]); % rho from above
        rhov2(idn) = v2(idn).*rhon;
        
        rhoFwd(:,k+1) = rhoFwd(:,k)-dt*(D1*rhov1 + D2*rhov2);
        
        % update running costs
        LcFwd = LcFwd + w(k) * 0.5* prod(h)* sum((A1*(v1.^2) + A2*(v2.^2)) .*rhoFwd(:,k),1);
        if alpha(2)>0
            FeFwd = FeFwd + w(k)*prod(h)*rhoFwd(:,k)'*log(rhoFwd(:,k));
        end
        if alpha(3)>0
            FpFwd = FpFwd + w(k)*prod(h)*Q'*rhoFwd(:,k);
        end
        
        %     viewImage2Dsc(rhoFwd(:,k+1),omega,m);
        %     caxis([min(rho1x) max(rho1x)])
        %        colorbar
    end
    
    LcFwd = LcFwd + w(end) * prod(h)* sum((A1*(v1.^2) + A2*(v2.^2)) .*rhoFwd(:,end),1);
    if alpha(2)>0
        FeFwd = FeFwd + w(end)*prod(h)*rhoFwd(:,end)'*log(rhoFwd(:,end));
    end
    if alpha(3)>0
        FpFwd = FpFwd + w(end)*prod(h)*Q'*rhoFwd(:,end);
    end
    GcFwd = prod(h)*rhoFwd(:,end)'*log(rhoFwd(:,end)./rho1x);
    fprintf('fwd:\t Lc=%1.2e\t Fe=%1.2e\tFp=%1.2e\tGc=%1.2e\tJc=%1.4e\n',alpha(1)*LcFwd,alpha(2)*FeFwd,alpha(3)*FpFwd,alpha(4)*GcFwd,alpha*[LcFwd;FeFwd;FpFwd;GcFwd])
    
    %% backward in time
    rhoBwd = zeros(prod(m),N+1);
    rhoBwd(:,end) = rho1x;
    LcBwd =0;
    FeBwd = 0;
    FpBwd = 0;
    for k = N:-1:1
        tk = k*dt;
        v1 = -getV1(xf1,tk);
        v2 = -getV2(xf2,tk);
        rhov1 = 0*v1;
        rhov2 = 0*v2;
        
        % flux in x1 direction
        idp = find(v1>=0);
        rhop = getRho(rhoBwd(:,k+1), xf1(idp,:)-[h(1)/2,0]); %  rho from left
        rhov1(idp) = v1(idp).*rhop;
        idn = find(v1<0);
        rhon = getRho(rhoBwd(:,k+1), xf1(idn,:)+[h(1)/2,0]);  %  rho from right
        rhov1(idn) = v1(idn).*rhon;
        
        % flux in x2 direction
        idp = find(v2>=0);
        rhop = getRho(rhoBwd(:,k+1), xf2(idp,:)-[0,h(2)/2]); %  rho from below
        rhov2(idp) = v2(idp).*rhop;
        idn = find(v2<0);
        rhon = getRho(rhoBwd(:,k+1), xf2(idn,:)+[0,h(2)/2]); % rho from above
        rhov2(idn) = v2(idn).*rhon;
        
        rhoBwd(:,k) = rhoBwd(:,k+1)-dt*(D1*rhov1 + D2*rhov2);
        
        % update running costs
        LcBwd = LcBwd + w(k) * 0.5* prod(h)* sum((A1*(v1.^2) + A2*(v2.^2)) .*rhoFwd(:,k),1);
        if alpha(2)>0
            FeBwd = FeBwd + w(k)*prod(h)*rhoBwd(:,k)'*log(rhoBwd(:,k));
        end
        if alpha(3)>0
            FpBwd = FpBwd + w(k)*prod(h)*Q'*rhoBwd(:,k);
        end
        
        %    viewImage2Dsc(rhoBwd(:,k+1),omega,m);
        %    caxis([min(rho1x) max(rho1x)])
        %    colorbar
    end
    LcBwd = LcBwd + w(end) * prod(h)* sum((A1*(v1.^2) + A2*(v2.^2)) .*rhoBwd(:,1),1);
    if alpha(2)>0
        FeBwd = FeBwd + w(end)*prod(h)*rhoBwd(:,end)'*log(rhoBwd(:,end));
    end
    if alpha(3)>0
        FpBwd = FpBwd + w(end)*prod(h)*Q'*rhoBwd(:,end);
    end
    GcBwd = prod(h)*rhoBwd(:,1)'*log(rhoBwd(:,1)./rho0x);
    fprintf('bwd:\t Lc=%1.2e\t Fe=%1.2e\tFp=%1.2e\tGc=%1.2e\tJc=%1.2e\n',alpha(1)*LcBwd,alpha(2)*FeBwd,alpha(3)*FpBwd,alpha(4)*GcBwd,alpha*[LcBwd;FeBwd;FpBwd;GcBwd])

%%
rho0z = rhoFwd(:,end);
rho1y = rhoBwd(:,1);

resContinuity = struct('rho0z',rho0z,'rho1y',rho1y,'costBwd',[alpha(1)*LcBwd,alpha(2)*FeBwd,alpha(3)*FpBwd,alpha(4)*GcBwd],...
    'costFwd',[alpha(1)*LcFwd,alpha(2)*FeFwd,alpha(3)*FpFwd,alpha(4)*GcFwd]);

save(resFile,'resContinuity','-append')

end

return
%%
fig = figure(); clf;
fig.Name = resFile;
subplot(2,3,1);
viewImage2Dsc(rhoFwd(:,1),omega,m);
title('rho0, given')
colorbar

subplot(2,3,2);
imgmontage(rhoFwd,[omega 0 1],[m N+1]);
caxis([min(rho1x(:)) max(rho1x(:))])
title('rhoFwd')
subplot(2,3,3);
viewImage2Dsc(rhoFwd(:,end),omega,m);
colorbar
title('rho1, estimated')
colorbar
caxis([min(rho1x(:)) max(rho1x(:))])

subplot(2,3,4);
viewImage2Dsc(rhoBwd(:,1),omega,m);
title('rho0, estimated')
colorbar
caxis([min(rho0x(:)) max(rho0x(:))])
subplot(2,3,5)
imgmontage(rhoBwd,[omega 0 1],[m N+1]);
caxis([min(rho1x(:)) max(rho1x(:))]);
title('rhoBwd')

subplot(2,3,6);
viewImage2Dsc(rhoBwd(:,end),omega,m);
colorbar
title('rho1, given')
colorbar


