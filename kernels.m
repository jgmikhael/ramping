% This code produces the kernels illustrated in "Ramping and state
% uncertainty in the dopamine signal" by Mikhael, Kim, Uchida, & Gershman.
% Written 10Oct19 by JGM.

clear; close all; clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% parameters
n = 6;              % number of kernels
mu = 1:n;           % means of kernel
w = .1;             % Weber fraction
eps = .1;           % uncertainty prior to first state 
prec = .01;
t = (0:prec:n)';    % time

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

col = [0 .3 .5 .7 .8 .95]'*[1 1 1];             % color schemes
col2 = .85*[1 1 1];
v = 1.3*normpdf(mu(1),mu(1),eps+w);           	% vertical spacing

figure(1)
for f = 1:2
    subplot(2,1,f)
    for e = 0:3
        x = normpdf(t, mu+e*(f-1), w*mu+eps);   % x(i,j) = (time i, cell j)
        if f == 1                               % no feedback
            xa = x(:,e+1:e+2);
        else                                    % feedback
            xa = x(:,1:2);
        end
        xb = x(:,e+1:end); xb(1:e/prec,:) = nan;
        plot(t,xb-e*v,'-.','Color',col2,'LineWidth',2)      % dotted curves
        hold on
        xa(1:e/prec,:) = nan;
        p = plot(t,xa-e*v);
        set(p, {'color'}, num2cell(col(e+1:e+2,:),2));      % gray curves
        
        xlim([-1 n])
        xticks(0:1:n)
        yticks([])
        xticklabels({'0','10','20','30','40','50'})
        box off
        xlim([0 5])
        ylim([-9 3])
    end
end