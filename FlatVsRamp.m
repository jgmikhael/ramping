% Fig 3 in Mikhael, Kim, Uchida, & Gershman.
% Written 10Nov20 by JGM.

clear; close all; clc

%% --------------------------------------------------------------------- %%

% parameters
gamma = .98;                 	% discount factor
alpha = .1;                   	% learning rate
n = 100;                     	% arbitrary (see: CS)
numIter = 2000;              	% number of iterations for learning
T = round(.8*n);               	% reward time
CS = round([.1 .1]*n)+1;      	% trial onset
weber = .15;                	% Weber fraction
T0 = 5;                         % total trial duration (s)

%% --------------------------------------------------------------------- %%

% visualize GCaMP impulse response function

load('kernel_GCaMP6m_UnexpR.mat')
% dn_x represents 10s, and length(dn_x) = 1001, so each time bin is 10ms

% complete the tail part (see Kim et al. (2020), right above 'Note on the
% shape of the fitted value function')
bTail = dn_x/1000>4 & dn_x/1000<=5;
dn_y(bTail) = dn_y(bTail).*(5.0-dn_x(bTail)/1000);

% cut out pre-impulse
dn_x = dn_x(501:end);
dn_y = dn_y(501:end);
lirf = length(dn_x);                    % length of IRF before sparsifying

% sparsify the IRF so that each bin corresponds to a state
spars = 1:(round(1000*T0/n)/10):lirf;  	% (ms/state)/(ms/bin) = bins/state
dnx = dn_x(spars);                      % time (ms) of each state
irf = dn_y(spars);                      % IRF for each state after the impulse

figure(101)
hold on
plot(dn_x/100,dn_y)                     % original (s)
plot(dnx/100,irf,'r--')                 % sparsified (s)
plot([0 0],[-.05 .4],'k--','LineWidth',1)
legend('Original','Sparsified','Box','Off')
ylim(1.1*[min(dn_y) max(dn_y)])
xlabel('Time (s)')
ylabel('GCaMP Activity')
box off

%% --------------------------------------------------------------------- %%

% kernel widths
S = .1+zeros(1,n);                      % SD of small uncertainty kernel
L = 3+zeros(1,n);                       % SD of large uncertainty kernel

% true value
t = 1:n;                                % time
r = zeros(n,1); r(T) = 1;               % reward schedule
oT = [1:CS(2)-1 T+1:n];                 % times outside the trial
V = gamma.^(T-t)'; V(oT) = 0;           % true value

% kernels = (time, kernel mean)
web = weber*(t-CS(1)); web(1:CS(1)-1) = 0;
[xs, xl, xw] = deal(zeros(n,n-1));
for y = 1:n
    xs(:,y) = normpdf(t,y,S(y))';       % small kernel
    xl(:,y) = normpdf(t,y,L(y))';       % large kernel
    xw(:,y) = normpdf(t,y,web(y))';     % Weber's law
end
xs(:,oT)=0; xl(:,oT)=0; xw(:,oT)=0;     % leave out times outside trial
xs=xs./sum(xs); xl=xl./sum(xl); xw=xw./sum(xw);         % make prob dist's
xs(isnan(xs))=0; xl(isnan(xl))=0; xw(isnan(xw))=0;      % nan's to zeros

%% --------------------------------------------------------------------- %%

% initialize
[w, Vh, delta] = deal(zeros(n,2));     	% weights, estimated value, RPE

% learning without feedback
for iter = 1:numIter
    for y = CS(1):T+1
        Vh(y,1) = w(:,1)'*xw(:,y);
        Vh(y+1,1) = w(:,1)'*xw(:,y+1);
        delta(y,1) = r(y) + gamma*Vh(y+1,1) - Vh(y,1);
        w(:,1) = w(:,1) + alpha*delta(y,1).*xw(:,y);
    end
end

% learning with feedback
beta = alpha*(exp((log(gamma))^2*(L.^2-S.^2)'/2)-1);
for iter = 1:numIter
    for y = CS(2):T+1
        Vh(y,2) = w(:,2)'*xs(:,y);
        Vh(y+1,2) = w(:,2)'*xl(:,y+1);
        delta(y,2) = r(y) + gamma*Vh(y+1,2) - Vh(y,2);
        w(:,2) = w(:,2) + (alpha*delta(y,2)-beta(y)*w(:,2)).*xs(:,y);
    end
end

%% --------------------------------------------------------------------- %%

lbls = {'CS','R';'Click','Goal'};
for e = 1:2
    figure(102); subplot(2,2,e)
    plot(t,delta(:,e),'k')
    yticks(0:.05:.1)
    % ylim([min(delta(:)) max(delta(:))])

    figure(e)
    dx = delta(:,e); 
    [~,v1] = max(dx);       % normalize dx jump by state size
    dx(v1) = dx(v1)*T0/n;
    if e == 1; dc = dx;   	% convolution with IRFs 
    else; dc = conv(dx,irf); end
    plot(dc,'k')
end

TT = [round(.2*n) T];
limX = CS-[3 6];
DAlabels = {'DA Spikes','[DA]'};
for e = 1:4
    if e < 3; figure(102); subplot(2,2,e)
    else figure(e-2)
        ylabel(DAlabels{e-2},'FontSize',26)
    end
   	xlabel('Time','FontSize',26)
  	box off
    ee = 2-rem(e,2);
    xlim([limX(ee)-3 1.3*TT(ee)])
    xticks([CS(ee)-1,TT(ee)])
    xticklabels({lbls{ee,:}})
    yticks([])
end
