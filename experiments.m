% This code produces the experiment simulations in "Ramping and state
% uncertainty in the dopamine signal" by Mikhael, Kim, Uchida, & Gershman.
% Written 30Sept19 by JGM.

clear; close all; clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% parameters
gamma = .8;                             % discount factor
alpha = .1;                             % learning rate
n = 50;                                 % arbitrary (see: CS)
numIter = 1000;                         % number of iterations for learning
T = n-2;                                % reward time
CS = [T-10 T-25];                       % trial onset, sets number of states
weber = .05;                          	% Weber fraction

% kernel widths
S = .1+zeros(1,n);                      % SD of small uncertainty kernel
L = 3+zeros(1,n); L(n-4:end) = .1;      % SD of large uncertainty kernel

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

% initialize
[w, Vh, delta] = deal(zeros(n,2));     	% weights, estimated value, RPE

% learning without feedback (Schultz et al., 1997)
for iter = 1:numIter
    for y = CS(1):T+1
        Vh(y,1) = w(:,1)'*xw(:,y);
        Vh(y+1,1) = w(:,1)'*xw(:,y+1);
        delta(y,1) = r(y) + gamma*Vh(y+1,1) - Vh(y,1);
        w(:,1) = w(:,1) + alpha*delta(y,1).*xw(:,y);
        w(T+1:end,1) = 1;
    end
end

% learning with feedback (Howe et al., 2013)
beta = alpha*(exp((log(gamma))^2*(L.^2-S.^2)'/2)-1);
for iter = 1:numIter
    for y = CS(2):T+1
        Vh(y,2) = w(:,2)'*xs(:,y);
        Vh(y+1,2) = w(:,2)'*xl(:,y+1);
        delta(y,2) = r(y) + gamma*Vh(y+1,2) - Vh(y,2);
        w(:,2) = w(:,2) + (alpha*delta(y,2)-beta(y)*w(:,2)).*xs(:,y);
        w(T+1:end,2) = 1;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lbls = {'CS','R';'Click','Goal'};
for e = 1:2
    figure(e)
    plot(t,delta(:,e),'k')
    xlabel('Time','FontSize',26)
    box off
    xticks([CS(e),T-1])
    xticklabels({lbls{e,:}})
    xlim([CS(e)-3 n])
    ylabel('RPE','FontSize',26)
    yticks(0:.05:.1)
    ylim([min(delta(:)) max(delta(:))])
end