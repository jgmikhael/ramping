% Fig S1 in Mikhael, Kim, Uchida, & Gershman.
% Written 14Nov20 by JGM.

clear; close all; clc

%%--------------------------------------------------------------------- %%

% set relevant params
T0 = 15;                % total trial duration (s)
n = 150;                % number of states (keep as multiple of T0)

% load IRF
load('kernel_GCaMP6m_UnexpR.mat')
% dn_x represents 10s, and length(dn_x) = 1001, so each time bin is 10ms

% cut out pre-impulse
dn_x = dn_x(501:end);
dn_y = dn_y(501:end);
lirf = length(dn_x);                % length of IRF before sparsifying

% sparsify the IRF so that each bin corresponds to a state
spars = 1:((1000*T0/n)/10):lirf;    % (ms/state)/(ms/bin) = bins/state
dnx = dn_x(spars);                  % time (ms) of each state
irf = dn_y(spars);                  % IRF for each state after the impulse

irfi = irf(3:end-5);

d = load('darkening.mat');
c = d.avg_neuron_LocOnV;
t = c.x;        % time
y = c.mean;     % average response

%%--------------------------------------------------------------------- %%

% smoothing by averaging over a number of states before and after (in
% addition to the state itself)
smoothB = 40; % number of states before
smoothF = 40; % number of states after

% deconvolution
xh = [];
for e = 1:size(y,1)
    yc = y(e,:); yc(isnan(yc)) = 0;
    xh(:,e) = ifft(fft(yc') ./ fft(irfi, length(yc)), 'symmetric');
    xh(:,e) = xh(1:length(yc),e);
    xhSmoothed(:,e) = movmean(xh(:,e),[smoothB smoothF]);
end

% re-convolve recovered response as a sanity check
for e = 1:size(y,1)
    yh = conv(xh(:,e),irfi);
    hold on
    %  plot(t,yh(1:length(t)),'LineWidth',2)
end

gamma = .999;
[Vh, V] = deal(zeros(length(xhSmoothed),size(xh,2)));
for e = 1:size(xh,2)
    xt = xhSmoothed(:,e);
    xt=xt-min(xt);      % set baseline to be RPE = 0
    [~,b] = max(xt);    % get end of trial
    
    % define uncertainty-free value
    V(1:length(t),e) = gamma.^(b-(1:length(t))');
    V(V>1) = 0;
    
    for q = 1:b
        Vh(q,e) = sum(xt(1:q)'./gamma.^(q:-1:1));
    end
end

% derive L2-S2
% the derived value is exp(L2-S2)*V, where V = gamma*(T-t)

% [normalBRIGHT normalDARK fastBRIGHT fastDARK]
% dh = log(Vh./V./max(Vh)+1).*2/(log(gamma)).^2;
DARK = xhSmoothed(:,[2 4]); BRIGHT = xhSmoothed(:,[1 3]);
del = log(DARK./BRIGHT).*2/(log(gamma))^2;

%%--------------------------------------------------------------------- %%

% color scheme
col = [34 32 32             % black
    184 184 180             % light gray
    224 36 44               % red
    248 172 76              % yellow
    ]/255;

figure(1)
subplot(3,1,1)
h{1} = plot(t,y);

subplot(3,1,2)
hold on
h{2} = plot(t,xhSmoothed);

subplot(3,1,3)
hold on
h{3} = plot(t,Vh);
xlabel('Time (s)')

ylbls = {'DA','RPE','Value'};
for e = 1:3
    subplot(3,1,e)
    set(h{e}, {'color'}, num2cell(col,2))
    box off
    yticks([])
    xlim([-2 12])
    ylabel(ylbls{e})
end

figure(101)

subplot(3,1,1)
tirf = (1:length(irfi))*(T0/n);
plot(tirf,irfi)
xlabel('Time (s)')
ylabel('IRF')
title('IRF')

subplot(3,1,2)
hold on
plot(t,xh)
ylabel('Deconvolved DA')
ylim([-1 5])

subplot(3,1,3)
hold on
plot(t,del)
plot(t,del*0,'k--','LineWidth',2)
ylabel('Delta L2-S2')

figure(1)
