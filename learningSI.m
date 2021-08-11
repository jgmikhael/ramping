% Fig S2 in Mikhael, Kim, Uchida, & Gershman.
% Updated 4Apr21 by JGM.

clear; close all; clc

%%--------------------------------------------------------------------- %%

% define parameters

gamma = .98;            % discount factor
CSUS = 7.6;             % time (s) between CS and US in standard task
T0 = 10;                % total trial duration (s)
n = 100;                % number of states (keep as multiple of T0)
%                         time per state: T0/n = (15000 ms)/(150 states)
%                                              = 100 ms/state
CSs = 1;                % CS time (s), arbitrary
CS = CSs*(n/T0);        % CS state
T = (CSUS+CSs)*(n/T0);  % reward (US) state: (s)(states/s)

weber = .15;            % Weber fraction

t = linspace(T0/n,T0,n);% true time
ti = 1:n;              	% list of states

% true value
r = zeros(n,1); r(T) = 1;      	% reward schedule
oT = [1:CS-1 T+1:n];           	% times outside the trial
V = gamma.^(T-ti)'; V(oT) = 0;	% true value

%%--------------------------------------------------------------------- %%

% color scheme
col = [34 32 32             % black
    127 129 131             % light gray
    ]/255;

%%--------------------------------------------------------------------- %%

% visualize GCaMP impulse response function

load('kernel_GCaMP6m_UnexpR.mat')
% dn_x represents 10s, and length(dn_x) = 1001, so each time bin is 10ms

% cut out pre-impulse
dn_x = dn_x(501:end);
dn_y = dn_y(501:end);
lirf = length(dn_x);                % length of IRF before sparsifying

% sparsify the IRF so that each bin corresponds to a state
spars = 1:((1000*T0/n)/10):lirf;    % (ms/state)/(ms/bin) = bins/state
irf = dn_y(spars);                  % IRF for each state after the impulse


%%-------------- Characterize S and L for Bright and Dark ------------- %%

ttl1 = {'Bright','Dark'};
pureBright = weber;

% dark condition

% characterize relationship between brightness and kernels
b2 = 3;         % controls the point of intersection for two regimes
c = 1;          % controls smoothness of transition between two regimes

% create S
e1 = 5;
e2 = weber*(ti-b2);                             % large kernel SD, DARK
S(2,:) = ((e1+e2)+sqrt((e1-e2).^2+4*c))./2-e1;  % combination of e1 and e2


% characterize (L-S)
z = 1.5*(pureBright-.1)./(1+exp(c*(t-b2)));

L(2,:) = S(2,:)+z;

% bright condition
S(1,:) = S(2,1)+zeros(1,n);
L(1,:) = L(2,1)+zeros(1,n);

%%--------------------------------------------------------------------- %%

beta = exp((log(gamma))^2*(L.^2-S.^2)'/2)-1;
d = beta.*V;

for e = 1:2
    dcx = conv(d(:,e),irf);
    dc(:,e) = dcx(1:length(d));
end

%%--------------------------------------------------------------------- %%

figure(101)

subplot(2,2,1)
plot(ti,L(1,:)-S(1,:))
ylabel('L-S')
title('Bright')

subplot(2,2,3)
hold on
plot(ti,S(1,:))
plot(ti,L(1,:))
ylabel('S and L')

subplot(2,2,2)
plot(ti,L(2,:)-S(2,:))
ylabel('L-S')
title('Dark')

subplot(2,2,4)
hold on
plot(ti,S(2,:))
plot(ti,L(2,:))
ylabel('S and L')

for e = 1:4
    subplot(2,2,e)
    xlabel('Time')
end

fig = figure(1);

for e = 1:2
    
    subplot(4,1,1)
    plot(ti,L(e,:)-S(e,:),'Color',col(e,:))
    hold on
    ylabel('{\boldmath${l-s}$}','interpreter','latex')
    legend('Bright','Dark','Location','Northeast','box','off')
    
    subplot(4,1,2)
    plot(ti,beta(:,e),'Color',col(e,:))
    hold on
    ylabel('{\boldmath${\beta}$}','interpreter','latex')
    
    subplot(4,1,3) % predicted RPE
    plot(ti,d(:,e),'Color',col(e,:))
    hold on
    ylabel('RPE')
    
    subplot(4,1,4) % predicted RPE+IRF
    plot(ti,dc(:,e),'Color',col(e,:))
    hold on
    ylabel('DA')
    xlabel('Time')
    
end

for e = 1:4
    subplot(4,1,e)
    yticks([])
    xticks(0:50:100)
    box off
end
