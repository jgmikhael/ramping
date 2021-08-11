% Figs 4 & 5 in Mikhael, Kim, Uchida, & Gershman.
% Written 10Apr21 by JGM and HRK

clear; close all; clc

%% --------------------------------------------------------------------- %%

% define parameters

gamma = .93;            % discount factor
alpha = .1;             % learning rate
numIter = 2000;         % number of iterations for learning

CSUS = 7.6;             % time (s) between CS and US in standard task
T0 = 20;                % total trial duration (s)
n = 200;                % number of states (keep as multiple of T0)
%                         time per state: T0/n = (15000 ms)/(150 states)
%                                              = 100 ms/state
CSs = 1;                        % CS time (s), arbitrary
CS = round(CSs*(n/T0));         % CS state
T = round((CSUS+CSs)*(n/T0));   % reward (US) state: (s)(states/s)

weber = .15;                    % Weber fraction
eps = 1;                        % generalized Weber's law
t = linspace(T0/n,T0,n);        % true time
ti = 1:n;                       % list of states

% true value
r = zeros(n,1); r(T) = 1;      	% reward schedule
oT = [1:CS-1 T+1:n];           	% times outside the trial
V = gamma.^(T-ti)'; V(oT) = 0;	% true value

%% --------------------------------------------------------------------- %%

% visualize GCaMP impulse response function

load('kernel_GCaMP6m_UnexpR.mat')
% dn_x represents 10s, and length(dn_x) = 1001, so each time bin is 10ms

% complete the tail part (see Kim et al. (2020), above 'Note on the shape
% of the fitted value function')
dn_x = [dn_x; 10*(501:1000)'];      % add 5 extra seconds
dn_y = dn_y(1:end-1);
xx = linspace(0,dn_y(end-1),length(dn_x)-length(dn_y));
dn_y = [dn_y; flipud(xx')];

% cut out pre-impulse
dn_x = dn_x(501:end);
dn_y = dn_y(501:end);
lirf = length(dn_x);                    % length of IRF before sparsifying

% sparsify the IRF so that each bin corresponds to a state
spars = 1:(round(1000*T0/n)/10):lirf;% (ms/state)/(ms/bin) = bins/state
dnx = dn_x(spars);                   % time (ms) of each state
irf = dn_y(spars);                   % IRF for each state after the impulse

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

% create uncertainty kernels

L = zeros(1,n)+eps+weber*T0/n;      % SD of large uncertainty kernel
S = zeros(1,n)+eps/2;               % SD of small uncertainty kernel

% kernels = (time, kernel mean)
web = eps+weber*(t-CSs); web(1:CS-1) = 0; web(T+1:end) = 0;
[xs, xl, xw] = deal(zeros(n,n-1));
for y = 1:n
    xs(:,y) = normpdf(ti,y,S(y))';          % small kernel
    xl(:,y) = normpdf(ti,y,L(y))';          % large kernel
    xw(:,y) = normpdf(t,y*T0/n,web(y))';    % Weber kernels
end
xs(:,oT)=0; xl(:,oT)=0; xw(:,oT)=0;         % leave out times outside trial
xs=xs./sum(xs); xl=xl./sum(xl); xw=xw./sum(xw);         % make prob dist's
xs(isnan(xs))=0; xl(isnan(xl))=0; xw(isnan(xw))=0;      % NaNs to zeros

%-------------------------------------------------------------------------%

figure(102)
subplot(3,3,1:6)
hold on
plot(t,S)
plot(t,L)
plot(t,web)
xlabel('Time (s)')
ylabel('SD of Uncertainty Kernels')
title('Uncertainty Kernel Width')
legend('Small (S)','Large (L)','Weber','Location','Northwest','Box','Off')

subplot(3,3,7)
plot(t,xl)
title('Large Kernels')
ylim([0 1])
xlabel('Time (s)')

subplot(3,3,8)
plot(t,xs)
title('Small Kernels')
ylim([0 1])
xlabel('Time (s)')

subplot(3,3,9)
plot(t,xw)
title('Weber')
xlabel('Time (s)')

%% --------------------------------------------------------------------- %%

% learning rules

% initialize weights, estimated value, RPE
[w, Vh, delta] = deal(zeros(n,2));

beta = alpha*(exp((log(gamma))^2*(L.^2-S.^2)'/2)-1);
for iter = 1:numIter
    for y = CS:n-1
        % estimated value without feedback
        Vh(y,1) = w(:,1)'*xw(:,y);
        Vh(y+1,1) = w(:,1)'*xw(:,y+1);
        
        % estimated value with feedback
        Vh(y,2) = w(:,2)'*xs(:,y);
        Vh(y+1,2) = w(:,2)'*xl(:,y+1);
        
        % update weights
        delta(y,:) = r(y) + gamma*Vh(y+1,:) - Vh(y,:);
        w(:,1) = w(:,1) + alpha*delta(y,1).*xw(:,y);
        w(:,2) = w(:,2) + (alpha*delta(y,2)-beta(y)*w(:,2)).*xs(:,y);
        w(T+1:end,:) = r(T);
    end
end

% fixed estimated value under small and large kernels
VhS = w(:,2)'*xs;
VhL = w(:,2)'*xl;

%-------------------------------------------------------------------------%

figure(103)
subplot(3,2,1)
hold on
plot(t,Vh(:,1))
plot(t,V,'k--')
ylabel('Value')
title('Without Feedback')

subplot(3,2,2)
hold on
plot(t,Vh(:,2))
plot(t,V,'k--')
title('With Feedback')

subplot(3,2,3)
plot(t,delta(:,1))
ylabel('RPE')

subplot(3,2,4)
hold on
plot(t,delta(:,2))
Vas = (beta/alpha).*Vh(:,2);        % asymptote of Vh with feedback
plot(t,Vas,'k--')

subplot(3,2,5)
dcx = conv(delta(:,1),irf);
dc(:,1) = dcx(1:n);
plot(t,dc(:,1))
ylabel('RPE + IRF')

subplot(3,2,6)
dcx = conv(delta(:,2),irf);
dc(:,2) = dcx(1:n);
plot(t,dc(:,2))

for e = 1:6
    subplot(3,2,e)
    hold on
    if e < 3
        MnMx = [min([Vh(:,e); V]) max([Vh(:,e); V])];
    elseif e > 4
        MnMx = [min(dc(:,e-4)) max(dc(:,e-4))];
        xlabel('Time (s)')
    else
        MnMx = [min(delta(:,e-2)) max(delta(:,e-2))];
    end
    plot(T*T0/n*[1 1],MnMx,'k--','LineWidth',2)
    ylim(MnMx)
    if e > 2
        minY = min([delta(:); dc(:)]);
        maxY = max([delta(:); dc(:)]);
    end
end

%% --------------------------------------------------------------------- %%

% VR manipulations: teleport and speed (Kim et al., 2020)

% color scheme
col = [235 32 35;                   % red
    0 0 0;                          % black
    249 181 0;                      % yellow
    234 89 0;                       % orange
    ]/255;

axLabelSize = 50;
wdth = 15;                          % width of curves

%%-----------------------------------------------------------------------%%

% teleport manipulations

figure(104)
[mn,mx] = deal(zeros(1,4));         % initialize min and max for y-axis

% (a) different magnitudes, same end
colx = col([1 4 2],:);              % colors: [red orange black]

% define [start end] of teleport
endTime = CS+floor((T-CS)*.7);
telep = endTime-round([(T-CS)*.3 0; (T-CS)*.05 0; 1 0]);
pauseTime = endTime-CS;

for sInd = 1:length(telep)          % for each task
    
    % experienced domain (i.e. exclude region that was teleported over)
    dom = [CS:telep(sInd,1) telep(sInd,2):n-1];
    
    % estimated value over the experienced domain
    VhLx = VhL(dom);                % under large kernel
    VhSx = VhS(dom);                % under small kernel
    
    % RPE without and with convolution
    dx = r(dom(1:end-1))'+gamma*VhLx(2:end)-VhSx(1:end-1);
    
    % normalize dx jump by state size
    [~,v1] = max(dx);
    dx(v1) = dx(v1)*T0/n;
    
    dcx = conv(dx,irf);
    dcx = dcx(1:length(dx));        % with convolution
    
    aa = dom(end)+(-length(dx)+1:0);
    ab = telep(3,1)*[1 1];
    for k = [1 3]
        figure(104); subplot(2,2,k)
        if k == 3
            figure(1)
            ylabel('DA','FontSize',axLabelSize)
        end
        
        hold on
        if k == 3; dx = dcx; end
        dx(end-(n-1-T):end) = nan; % remove post-US response
        plot(t(aa),dx,'Color',colx(sInd,:),'LineWidth',wdth)
        plot(t(ab),[-1 1],'Color',colx(1,:),'LineWidth',2)
        plot(t(T)*[1 1],[-1 1],'k--','LineWidth',2)
        mx(k) = max([mx(k) dx]); mn(k) = min([mn(k) dx]);
        
        % plot pause condition
        if sInd == length(telep)
            dom = [CS:telep(sInd,1) telep(sInd,2):n-1];
            dx = r(dom(1:end-1))'+ gamma*VhLx(2:end)-VhSx(1:end-1);
            pauseDur = 5*n/T0;              % 5s-pause, and T0/n states/s
            dx = [dx(1:pauseTime) zeros(1,pauseDur) dx(pauseTime+1:end)];
            dx((T+pauseDur-CS):end) = nan;  % remove post-US response
            dom = CS+(1:length(dx));
            
            if k == 3                       % convolution with IRF
                dcx = conv(dx,irf); dx = dcx(1:length(dx));
            end
            mx(k) = max([mx(k) dx]); mn(k) = min([mn(k) dx]);
            
            tx = t(1):diff(t):3*t(end); % elongate t to accommodate pause
            plot(tx(dom),dx,'Color',col(3,:),'LineWidth',wdth);
            plot(tx(T+pauseDur*[1 1]),[-1 1],'--',...
                'Color',col(3,:),'LineWidth',2)
        end
        ylim([mn(k) mx(k)])
    end
end

% (b) same magnitude, different start/ends

% define [start end] of teleport
mag = .3*(T-CS);                % size of jump
early = CS+.05*(T-CS);
mid = CS+.25*(T-CS);
late = CS+.45*(T-CS);
telep=round([early early+mag; mid mid+mag; late late+mag; mid mid+1]);
colx = col([1 4 3 2],:);        % colors: [red orange yellow black]

for sInd = 1:length(telep)      % for each task
    
    % experienced domain (i.e. exclude region that was teleported over)
    dom = [CS:telep(sInd,1) telep(sInd,2):n-1];
    
    % estimated value over the experienced domain
    VhLx = VhL(dom);            % under large kernel
    VhSx = VhS(dom);            % under small kernel
    
    % RPE
    dx = r(dom(1:end-1))'+gamma*VhLx(2:end)-VhSx(1:end-1);
    
    % normalize dx jump by state size
    [~,v1] = max(dx);
    dx(v1) = dx(v1)*T0/n;
    
    dcx = conv(dx,irf);
    dcx = dcx(1:length(dx));    % convolution
    for k = [2 4]
        if k == 4; dx = dcx; end
        
        figure(104); subplot(2,2,k)
        if k == 4
            figure(2)
        end
        hold on
        gap = telep(:,2)-telep(:,1);
        xx = n-gap-CS;
        dom = CS+(1:xx(sInd));
        dx(T-gap(sInd)-CS+1:end) = nan;   % remove post-US response
        plot(t(dom),dx(1:xx(sInd)),'Color',colx(sInd,:),'LineWidth',wdth);
        if sInd < 4                % no teleport for standard condition
            plot(t(telep(sInd,1))*[1 1],[-1 1],...
                'Color',colx(sInd,:),'LineWidth',2)
            plot(t(T-gap(sInd)+[1 1]),[-1 1],'--',...
                'Color',colx(3,:),'LineWidth',2)
        end
        plot(t(T)*[1 1],[-1 1],'k--','LineWidth',2)
        
        % update ylim
        mn(k) = min(mn(k),min(dx(:)));
        mx(k) = max(mx(k),max(dx(:)));
        ylim([mn(k) mx(k)])
    end
end

ttl = {'Different Magnitudes, Same Endpoint',...
    'Same Magnitude, Different Endpoints'};
for k = 1:2
    figure(104); subplot(2,2,k)
    ylabel('RPE','FontSize',axLabelSize)
    title(ttl{k},'FontSize',20)
    figure(k)
end

%% --------------------------------------------------------------------- %%

% speed manipulation

% create uncertainty kernels
S = 1; L = 3;

gain = [2 1 .5];         	% [slow medium fast]

figure(105)
sgtitle('Speed','FontSize',25)
colx = col([3 2 1 1],:);  	% colors: [red black arbitrary yellow]

[mn,mx] = deal([0 0]);
for ee = 1:length(gain)    	% relative speed, compared to slowest condition
    
    % reset reward timing
    r = zeros(n,1); r(round(T/gain(ee))) = 1;       % reward schedule
    
    % kernels = (time, kernel mean)
    [xs, xl] = deal(zeros(n,n-1));
    for y = 1:n
        xs(:,y) = normpdf(ti,y*gain(ee),S)';        % small kernel
        xl(:,y) = normpdf(ti,y*gain(ee),L)';        % large kernel
    end
    oT = 1:CS-1;
    xs(:,oT)=0; xl(:,oT)=0;                 % leave out times outside trial
    xs=xs./sum(xs); xl=xl./sum(xl);         % make prob dist's
    xs(isnan(xs))=0; xl(isnan(xl))=0;       % nan's to zeros
    
    % fixed estimated value under small and large kernels
    VhS = w(:,2)'*xs;
    VhL = w(:,2)'*xl;
    
    VhS(round(T/gain(ee))+1:end) = nan; VhL(round(T/gain(ee))+1:end) = nan;
    dx = r(1:end-1)' + gamma*VhL(2:end) - VhS(1:end-1);
    dcx = conv(dx,irf);
    dc = dcx(1:length(dx));
    
    % map RPEs to spikes
    dc(dc<0) = dc(dc<0)/5;
    
    mn(1) = min(mn(1),min(dx(:))); mx(1) = max(mx(1),max(dx(:)));
    mn(2) = min(mn(2),min(dc(:))); mx(2) = max(mx(2),max(dc(:)));
    
    figure(105); subplot(2,1,1)
    hold on
    plot(dx,'Color',colx(ee,:))
    plot(T*[1 1]/gain(ee),[-1 1],'--','Color',colx(ee,:))
    figure(3)
    hold on
    plot(dc,'Color',colx(ee,:),'LineWidth',wdth)
    plot(T*[1 1]/gain(ee),[-1 1],'--','Color',colx(ee,:))
    
end

for e = 1:2
    figure(3)
    ylim([mn(e) mx(e)])
end

%% --------------------------------------------------------------------- %%

% darkening

% color scheme
colx = [34 32 32          	% black
    184 184 180             % light gray
    224 36 44               % red
    248 172 76              % yellow
    ]/255;

ttl1 = {'Bright','Dark'};
maxY = 0;                 	% initialize ylim

ww = weber*(1:n)+eps;       % generalized Weber
gain = [1 1.7];           	% speed gain

% baseline noise (arbitrary)
q = 8;

x = linspace(-10,10,length(ww)); 
b = 1;
z = .8./(1+exp(b*x));

for ee = 1:2                % speed: [normal fast]
    % reset reward timing
    r = zeros(n,1); r(round(T/gain(ee))) = 1;      	% reward schedule
    
    MnMx = zeros(3,2);          % initialize ylim for the 3 panels
    for e = 1:2                 % [bright dark]
        
        ec = 2*(ee-1)+e;        % color scheme
        
        % darkening: (y-e1)(y-e2) = c
        b2 = -max(ww)/1000;     % controls the point of intersection
        c = 3;                  % controls the smoothness of the transition
        e1 = q;
        e2 = ww+b2;
        y = ((e1+e2)+sqrt((e1-e2).^2+4*c))./2;
        
        S = q+y*(e-1); % second term is zero for bright; ww for dark
        L=S+z;
        
        % kernels = (time, kernel mean)
        [xs, xl] = deal(zeros(n,n-1));
        for y = 1:n
            xs(:,y) = normpdf(ti,y*gain(ee),S(y))';      % small kernel
            xl(:,y) = normpdf(ti,y*gain(ee),L(y))';      % large kernel
        end
        oT = 1:CS-1;
        xs(:,oT)=0; xl(:,oT)=0;           	% leave out times outside trial
        xs=xs./sum(xs); xl=xl./sum(xl);    	% make prob dist's
        xs(isnan(xs))=0; xl(isnan(xl))=0;   % nan's to zeros
        
        % fixed estimated value under small and large kernels
        VhS = w(:,2)'*xs;
        VhL = w(:,2)'*xl;
        
        VhS(round(T/gain(ee))+1:end) = 0; VhL(round(T/gain(ee))+1:end) = 0;
        d = r(1:end-1)' + gamma*VhL(2:end) - VhS(1:end-1);
        
        % map RPEs to spikes
        d(d<0) = d(d<0)/5;
        
        figure(106)
        sgtitle('Uncertainty','FontSize',25)
        ff = [1 4; 7 10];
        subplot(4,3,ff(e,:))
        hold on
        plot(S)
        plot(L)
        if e == 2
            plot(q+e1,'k--','LineWidth',2)
            plot(q+e2,'k--','LineWidth',2)
        end
        xlabel('Time')
        ylabel('SDs of Uncertainty Kernels')
        title(ttl1{e})
        legend('Small (S)','Large (L)','Location','Northwest','Box','Off')
        ylim([0 max(L)])
        
        subplot(4,3,ff(e,1)+2)
        plot(xl)
        ylabel('Large Kernels','FontSize',15)
        xlabel('Time','FontSize',15)
        xticks([]); yticks([])
        ylim([0 max(xs(:))])
        
        subplot(4,3,ff(e,1)+5)
        plot(xs)
        ylabel('Small Kernels','FontSize',15)
        xlabel('Time','FontSize',15)
        xticks([]); yticks([])
        ylim([0 max(xs(:))])
        
        ff2 = [2 5; 8 11];
        subplot(4,3,ff2(e,:))
        hold on
        plot(VhS)
        plot(VhL,'--')
        ylabel('Estimated Value')
        legend('Small (S)','Large (L)','Location','Northwest','Box','Off')
        
        figure(107)
        sgtitle('Darkening','FontSize',25)
        xx = (1:length(VhS))*T0/n;
        
        subplot(3,1,1)
        hold on
        plot(xx,VhS,'Color',colx(ec,:),'LineWidth',wdth/4)
        plot(xx,VhL,'--','Color',colx(ec,:),'LineWidth',wdth/4)
        ylabel('Value')
        
        % sets ylim
        MnMx(1,1) = min([MnMx(1,1) min([VhS VhL])]);
        MnMx(1,2) = max([MnMx(1,2) max([VhS VhL])]);
        
        subplot(3,1,2)
        plot(xx(1:end-1),d,'Color',colx(ec,:),'LineWidth',wdth/4)
        hold on
        ylabel('RPE')
        
        % sets ylim
        MnMx(2,1) = min([MnMx(2,1) min(d)]);
        MnMx(2,2) = max([MnMx(2,2) max(d)]);
        
        subplot(3,1,3)
        dcx = conv(d,irf);
        dc = dcx(1:length(d));
        plot(xx(1:end-1),dc,'Color',colx(ec,:),'LineWidth',wdth/4)
        hold on
        xlabel('Time (s)')
        ylabel('RPE+IRF')
        
        % sets ylim
        MnMx(3,1) = min([MnMx(3,1) min(dc)]);
        MnMx(3,2) = max([MnMx(3,2) max(dc)]);
        
    end
    
    for e = 1:3
        subplot(3,1,e)
        plot(T*T0/n/gain(ee)*[1 1],[-1 1],'k--','LineWidth',2)
        ylim([MnMx(e,1)-.03 MnMx(e,2)])
    end
    
end

sgtitle('Darkening','FontSize',25)
lgd{1} = {'V(t), Bright','V(t+1), Bright','V(t), Dark','V(t+1), Dark'};
lgd{2} = {'Bright','Dark'};
for e = 1:2
    subplot(3,1,e)
    legend(lgd{e},'Location','Northwest','Box','Off')
end

for e = 1:3
    figure(e)
    xlabel('Time','FontSize',axLabelSize)
    xticks([]); yticks([])
    box off
end
