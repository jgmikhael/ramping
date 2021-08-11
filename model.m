% Figs 1 & 2 in Mikhael, Kim, Uchida, & Gershman.
% Written 30Sept19 by JGM.

clear; close all; clc

%% --------------------------------------------------------------------- %%

% parameters
gamma = .9;                             % discount factor
alpha = .1;                             % learning rate
n = 50;                                 % number of states
numIter = 1000;                         % number of iterations for learning
CS = 1;                                 % trial onset
T = n-2;                                % reward time
weber = .15;                            % Weber fraction

% true value
t = 1:n;                                % time
r = zeros(n,1); r(T) = 1;               % reward schedule
oT = [1:CS-1 T+1:n];                    % times outside the trial
V = gamma.^(T-t)'; V(oT) = 0;           % true value

% kernel widths
S = .1+zeros(1,n);                      % SD of small uncertainty kernel
L = 3+zeros(1,n); L(n-4:end) = .1;      % SD of large uncertainty kernel
web = weber*t;

% kernels = (time, kernel mean)
[xs, xl, xw] = deal(zeros(n,n-1));
for y = 1:n
    xs(:,y) = normpdf(t,y,S(y))';       % small kernel
    xl(:,y) = normpdf(t,y,L(y))';       % large kernel
    xw(:,y) = normpdf(t,y,web(y))';     % Weber's law
end
xs(:,oT)=0; xl(:,oT)=0; xw(:,oT)=0;     % leave out times outside trial
xs=xs./sum(xs); xl=xl./sum(xl); xw=xw./sum(xw);    % make prob dist's
xs(isnan(xs))=0; xl(isnan(xl))=0; xw(isnan(xw))=0; % nan's to zeros

% correction term:  [without correction | with correction]
beta = [zeros(n,1) alpha*(exp((log(gamma))^2*(L.^2-S.^2)'/2)-1)];

% learning with feedback
[w, Vh, delta] = deal(zeros(n,3));     	% weights, estimated value, RPE
for e = 1:2
    for iter = 1:numIter
        for y = 1:T+1
            Vh(y,e) = w(:,e)'*xs(:,y);
            Vh(y+1,e) = w(:,e)'*xl(:,y+1);
            delta(y,e) = r(y) + gamma*Vh(y+1,e) - Vh(y,e);
            w(:,e) = w(:,e) + (alpha*delta(y,e)-beta(y,e)*w(:,e)).*xs(:,y);
            w(T+1:end,e) = 1;          % value stays high until reward
        end
    end
end

% learning without feedback
for iter = 1:numIter
    for y = 1:T+1
        Vh(y,3) = w(:,3)'*xw(:,y);
        Vh(y+1,3) = w(:,3)'*xw(:,y+1);
        delta(y,3) = r(y) + gamma*Vh(y+1,3) - Vh(y,3);
        w(:,3) = w(:,3) + alpha*delta(y,3).*xw(:,y);
        w(T+1:end,3) = r(T);         	% value stays high until reward
    end
end

%% --------------------------------------------------------------------- %%

% plot uncertainty kernels
figure(101)
subplot(2,1,1)
plot(t,xl)
title('Before Feedback')
subplot(2,1,2)
plot(t,xs)
title('After Feedback')
xlabel('Time')

figure(1)
fL = [3 1];                             % index of relevant value estimates
hL = [10 20 30];                        % initial location of red curves
y = 10;                                 % length of red curves along x-axis
cols=[205 92 92; 255 0 0; 150 0 0]/255; % red color schemes
feed = {'Without Feedback','With Feedback'};
for e = [2 1]
    f = fL(e);
    subplot(1,2,e) 
    hold on
    plot(t,V)
    plot(t,Vh(:,f),'k')
    for g = 1:3
        redcurve = nan(1,n);
        h = hL(g);
        redcurve(h+(0:y)) = Vh(h,f).*gamma.^(0:-1:-y);
        plot(redcurve,'Color',cols(g,:))
    end
    % title(feed{e})
    xlim([0 n])
    xlabel('Time')
    ylim([0 1])
    ylabel('Value')
    box off
end
legend('True Value', 'Estimated Value', 'Location','Northwest','box','off')

figure(2)
corr = {'Without Correction','With Correction'};
for e = 1:2
    subplot(2,2,e)
    hold on
    plot(t,V)
    plot(t,Vh(:,e),'k')
    ylabel('Value')
    ylim([min(min([Vh V])) max(max([Vh V]))])
    title(corr{e})
    
    subplot(2,2,e+2)
    plot(t,delta(:,e),'k')
    ylabel('RPE')
    ylim([min(min(delta(:,1:2))) max(max(delta(:,1:2)))])
end

subplot(2,2,1)
legend('True Value','Estimated Value','Location','Northwest','box','off')

for e = 1:4
    subplot(2,2,e)
    xlabel('Time')
    xlim([0 n])
    box off
end
