%==============================================================================
%
% solve min_x fctn(x) s.t. D*x=q
%
% compute saddle point of Lagrangian L(x,lam) = fctn(x) + lam'*(D*x-q)
%
% Input:
%   fctn        function handle
%   xc          initial guess
%   D          - constraint matrix
%   q          - rhs of constraint
%   varargin    optional parameter, see below
%
% Output:
%   (xc,lam)    saddle point (current iterate)
%   His         iteration history
%
% see also GaussNewton.m
%==============================================================================

function [xc,lam,His] = PrimalDualNewton(fctn,xc,D,q,varargin)

if nargin ==0, % help and minimal example
  help(mfilename); return;
end;

% parameter initialization -----------------------------------------------
maxIter      = 10;              % maximum number of iterations
atolPrimal   = 1e-8;            % for stopping, objective function
atolDual     = 1e-2;            %   - " -     , current value
vecNorm      = @norm;           % norm to be used for dJ and dy
solver       = [];              % linear solver
xStop        = xc;              % used for stopping in multi-level framework
Jstop        = [];              %
Plots        = @(iter,para) []; % for plots;
directSolve  = 1;
lam          = zeros(size(D,1),1);
lamStop      = zeros(size(D,1),1);
gmresRestart = 10;
gmresTol     = 1e-2;
gmresMaxIter = 100;
mu           = 0;
Ain          = [];
bin          = [];
for k=1:2:length(varargin)    % overwrites default parameter
  eval([varargin{k},'=varargin{',int2str(k+1),'};']);
end

if ~isa(Plots,'function_handle') && (Plots == 0 || strcmp(Plots,'off'))
  Plots        = @(iter,para) []; % for plots;
end

if isempty(xStop), xStop  = xc; end; % vStop used for stopping only
% -- end parameter setup   ----------------------------------------------

% some output
% FAIRmessage = @(str) fprintf('%% %s  [ %s ]  % s\n',...
%   char(ones(1,10)*'-'),str,char(ones(1,60-length(str))*'-'));
FAIRmessage([mfilename '(LR 2017/06/01)']);
fprintf('[ maxIter=%s / atolPrimal=%s / atolDual=%s / length(yc)=%d ]\n',...
  num2str(maxIter),num2str(atolPrimal),num2str(atolDual),length(xc));

% -- initialize  ---------------------------------------------------------
STOP = zeros(3,1);
if isempty(Jstop) || isempty(xStop)
    % evaluate objective function for stopping values and plots
    [Jstop,paraStop,dJStop] = fctn(xStop); Jstop = abs(Jstop) + (Jstop == 0);
    Plots('stop',paraStop);
    resPriStop  = D*xStop - q;
    resDualStop = dJStop + lamStop'*D;
end

% evaluate objective function for starting values and plots
[Jc,para,dJ,H] = fctn(xc);
if (mu>0) && not(isempty(Ain)) && not(isempty(bin))
    gc = Ain*xc-bin;
    if any(gc(:)<0)
        error('xc must be feasible')
    end
    Pc =  -mu*sum(log(gc));
    dP = (mu./gc')*Ain;
    d2P = Ain'*diag(sparse(1./gc.^2))*Ain;
    Jc = Jc + Pc;
    dJ = dJ + dP;
    H = H + d2P;
end
resPri  = D*xc - q;
resDual = dJ + lam'*D;

Plots('start',para);
iter = 0; xOld = 0*xc; Jold = Jc; x0 = xc;

hisStr    = {'iter','J','Jold-J','|resPri|','|resDual|','|dx|','LS','gmresIter','relres','L','Fe','Fp','Gc'};
his        = zeros(maxIter+2,13);
his(1,1:5) = [-1,Jstop,Jstop-Jc,vecNorm(resPriStop),vecNorm(resDualStop)];
his(1,10:13) = [para.Lc, para.Fec, para.Fpc, para.Gc];


nnzH = -1;
if isnumeric(H); nnzH = nnz(H); end;
his(2,:)   = [0,Jc,Jstop-Jc,vecNorm(resPri),vecNorm(resDual),vecNorm(x0-xStop),0,0,0.0 para.Lc, para.Fec, para.Fpc, para.Gc];

% some output
fprintf('%4s %-12s %-12s %-12s %-12s %-12s  %-12s  %-12s  %-12s  %-12s  %-12s  %-12s  %-12s\n%s\n',...
  hisStr{:},char(ones(1,111)*'-'));
dispHis = @(var) ...
  fprintf('%4d %-12.4e %-12.3e %-12.3e %-12.3e %-12.3e  %-12d %-12d   %-12.3e  %-12.3e  %-12.3e  %-12.3e  %-12.3e\n',var);
dispHis(his(1,:));
dispHis(his(2,:));
% -- end initialization   ------------------------------------------------


%-- start the iteration --------------------------------------------------
while 1
  % check stopping rules
  STOP(1) = (vecNorm(resPri) <= atolPrimal);
  STOP(2) = (vecNorm(resDual) <= atolDual);
  STOP(3) = (iter >= maxIter);
  if all(STOP(1:2)) || any(STOP(3)), break;  end;

  iter = iter + 1;
  % solve the KKT system
  r1 = -resDual(:);
  r2 = -resPri(:);
  
  % solve saddle-point problem: 
  %
  % | H  D'| | dx   | =  |-dJ-D'*lam|
  % | D  0 | | dlam |    |-r2|
  %
  % where, H is the Hessian, D a space-time divergence, and r1 and r2 the
  % residuals computed in the objective function. If the size of this
  % linear system is small, we can build the Schur complement S=-D*(H\D')
  % and solve first for dlam and then for dx. If the system is large, we
  % can use 
  if directSolve
%     tic;
%     S = -D*(H\D'); % Schur complement
%     dlam = S\(r2-D*(H\r1)); % Neeha's calculation
%     dx = H\(r1 - D'*dlam);
%     timeSchur = toc;
    % for testing, compare with solution from saddle point  problem
    tic;
    KKT  = [H D'; D sparse(size(D,1),size(D,1))]; 
    dd   = KKT\[r1;r2];
    dx   = dd(1:numel(r1));
    dlam = dd(numel(r1)+1:end);
    timeKKT = toc;
%     errdx= norm(dx-dxt)/norm(dx);
%     errdl = norm(dlam-dlamt)/norm(dlam);
%     fprintf('timeSchur: %1.2f\t timeKKT:%1.2f\t ratio:%1.4f\n',timeSchur,timeKKT,timeKKT/timeSchur);
%     fprintf('err(dx) = %1.2e\terr(dlam) = %1.2e\n',errdx,errdl);
    gmresIter = -1;
    relres = norm(KKT*dd-[r1;r2])/norm([r1;r2]);;
  else
    % build Schur preconditioner
    
    M = @(x) SchurPrecond(H,D,x);
    KKT = [H D'; D sparse(size(D,1),size(D,1))]; 
    [dd,flag,relres,gmresIter,resvec]  = gmres(KKT,[r1;r2],gmresRestart,gmresTol,gmresMaxIter,M);
    dx = dd(1:numel(r1));
    dlam = dd(numel(r1)+1:end);
  end
  
  % perform line-search to make sure residuals decrease
  LSmaxIter = 10;
  t = 1.0; LS = 0;
  for LSiter =1:LSmaxIter
      xt   = xc  + t*dx;
      lamt = lam + t*dlam;
      
      [Jt,para,dJt] = fctn(xt);
      if (mu>0) && not(isempty(Ain)) && not(isempty(bin))
          gt = Ain*xt-bin;
          if any(gt(:)<0)
              Jt = Inf;
          else
              Jt = Jt -mu*sum(log(gt));
              dJt = dJt + (mu./gt')*Ain;
          end
      end
      resPrit  = D*xt - q;
      resDualt = dJt + lamt'*D;
      
      
      LS = ~isinf(Jt) & (norm([resPrit(:);resDualt(:)]) < norm([resPri(:);resDual(:)]));
      if LS, break; end
      t = t/2;          % reduce t
  end
  if LS==0
      warning('LS fail');
%       keyboard
      break;
  end
  
  % save old values and update
  xOld = xc; Jold = Jc; xc = xt; lam = lamt;
  [Jc,para,dJ,H] = fctn(xc); % evalute objective function
  if (mu>0) && not(isempty(Ain)) && not(isempty(bin))
      gc = Ain*xc-bin;
      if any(gc(:)<0)
          error('xc must be feasible')
      end
      Pc =  -mu*sum(log(gc));
      dP = (mu./gc')*Ain;
      d2P = mu*Ain'*diag(sparse(1./gc.^2))*Ain;
      Jc = Jc + Pc;
      dJ = dJ + dP;
      H = H + d2P;
  end

  resPri  = D*xc - q;
  resDual = dJ + lam'*D;
  
%   clf; plot(para.T,para.d,'r',para.T,para.g,'g')
%   pause

  % some output
  his(iter+2,:) = [iter,Jc,Jold-Jc,vecNorm(resPri),vecNorm(resDual),vecNorm(xc-xOld),LSiter,gmresIter(1),relres, para.Lc, para.Fec, para.Fpc, para.Gc];
  dispHis(his(iter+2,:));
  Plots(iter,para);
% pause
end;%while; % end of iteration loop
%-------------------------------------------------------------------------
Plots(iter,para);

% clean up
His.str = hisStr;
His.his = his(1:iter+2,:);
His.para = para;
fprintf('STOPPING:\n');
fprintf('%d[ %-12s=%16.8e <= %-20s=%16.8e]\n',STOP(1),...
  '|resPrimal|',vecnorm(resPri),'atolPrimal',atolPrimal);
fprintf('%d[ %-12s=%16.8e <= %-20s=%16.8e]\n',STOP(2),...
  '|resDual|',vecnorm(resDual),'atolDual',atolDual);
fprintf('%d[ %-12s=  %-14d >= %-20s=  %-14d]\n',STOP(3),...
  'iter',iter,'maxIter',maxIter);

FAIRmessage([mfilename,' : done !']);

%==============================================================================

function x = SchurPrecond(H,D,r)
r1 = r(1:size(H,1));
r2 = r(numel(r1)+1:end);
S = @(x) D*(H\(D'*x)); % Schur complement
dlam = pcg(S,-(r2-D*(H\r1)),1e-2,1000); % Neeha's calculation
dx = H\(r1 - D'*dlam);
x = [dx;dlam];
