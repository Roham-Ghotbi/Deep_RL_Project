%% HW 5

%% HW 5
clear all; close all;
home =  '/Users/chnajafi94/Documents/GitHub/cs285Project/cs285/GrpB-Pendulum';
cd(home);

q_a0 = '*A0*';
q_nodiff = '*NoDiff*';
% fol_doubleq = dir([q_string,'doubled*']);
st_eval_avg = '*Eval_AverageReturn.csv';
st_eval_std = '*Eval_StdReturn.csv';
st_eval_max = '*Eval_MaxReturn.csv';
st_ac_loss =  '*Actor_Loss.csv';
st_crit_loss = '*Critic_Loss.csv';
st_train_ret = '*Train_AverageReturn.csv';
st_train_std = '*Train_StdReturn.csv';

fol_eval_avg = dir(st_eval_avg); 
fol_eval_std = dir(st_eval_std);
fol_eval_max = dir(st_eval_max);
fol_ac_loss =  dir(st_ac_loss);
fol_crit_loss = dir(st_crit_loss);
fol_train_avg = dir(st_train_ret);
fol_train_std = dir(st_train_std);
c = {};
%
col = ['b','r'];
mark = ['s','v'];
idx = [1,3,4,5,2,6,8,9,10,7];
legends = { 'Differential Q Learning - 0% Noise', ...
            'Differential Q Learning - 3% Noise', ...
            'Differential Q Learning - 5% Noise', ...
            'Differential Q Learning - 7% Noise', ...
            'Differential Q Learning - 10% Noise', ...
            'Conventional Q Learning - 0% Noise',...
            'Conventional Q Learning - 3% Noise',...
            'Conventional Q Learning - 5% Noise',...
            'Conventional Q Learning - 7% Noise',...
            'Conventional Q Learning - 10% Noise'};

%% Actor Loss
figure();
colormap parula; map = colormap;
style = {'-',':'};
for i = idx
    a = importdata(fol_ac_loss(i).name); 
    j = (i>5) * 1;
    if(j==0) colormap parula; map = colormap; else colormap hot; map = colormap; end
    a = a.data; 
    data = a(:,3); iter = a(:,2); 
    hold on;
    i_mod = mod(i,5)+0.1;
    m = 1-(1-j)*(i_mod/5);
    n = 1-j*(i_mod/5);
    plot(iter, movmean(data,5),'Color',map(round(i_mod/5*64),:),'linewidth',10);
end
c = {'Differential Q Learning', 'Conventional Q Learning'};
grid on;
ax = gca;
ax.LineWidth =5;
set(gcf, 'Position', get(0,'Screensize')); 
set(gca, 'FontSize', 50,'FontWeight', 'Bold'); 
xlabel('Iteration #');
ylabel('Actor Loss');
l = legend(legends,'Location','NorthWest ','Interpreter','None');
l.NumColumns=2;
ylim([-10 85]);
saveas(gcf,'AC_Loss.png');saveas(gcf,'AC_Loss.fig');

%% Critic Loss
figure();
colormap parula; map = colormap;
style = {'-',':'};
for i = idx
    a = importdata(fol_crit_loss(i).name); 
    j = (i>5) * 1;
    if(j==0) colormap parula; map = colormap; else colormap hot; map = colormap; end
    a = a.data; 
    data = a(:,3); iter = a(:,2); 
    hold on;
    i_mod = mod(i,5)+0.1;
    m = 1-(1-j)*(i_mod/5);
    n = 1-j*(i_mod/5);
    plot(iter, movmean(data,5),'Color',map(round(i_mod/5*64),:),'linewidth',6);
end
grid on;
ax = gca;
ax.LineWidth =5;
set(gcf, 'Position', get(0,'Screensize')); 
set(gca, 'FontSize', 50,'FontWeight', 'Bold'); 
xlabel('Iteration #');
ylabel('Actor Loss');
l = legend(legends,'Location','NorthWest ','Interpreter','None');
l.NumColumns=2;
ylim([0.1 2500]);
set(gca, 'YScale', 'log');
saveas(gcf,'CRITIC_Loss.png');saveas(gcf,'CRITIC_Loss.fig');

%% Eval Return (AVG+STD) with error bars
figure();
style = {'-',':'};
for i = idx
    a = importdata(fol_eval_avg(i).name); 
    j = (i>5) * 1;
    if(j==0) colormap parula; map = colormap; else colormap hot; map = colormap; end
    a = a.data; 
    avg = a(:,3); iter = a(:,2); 
    hold on;
    a = importdata(fol_eval_std(i).name); 
    a = a.data; 
    std = a(:,3); iter = a(:,2); 
    n = 4;
    i_mod = mod(i,5)+0.1;
    avg = movmean(avg,n); 
    std = movmean(std,n);
    errorbar(iter(1:2:end), avg(1:2:end),std(1:2:end),'Color',map(round(i_mod/5*64),:),'linewidth',4,'Markersize',1,'CapSize',2);
end
grid on;
ax = gca;
ax.LineWidth =5;
set(gcf, 'Position', get(0,'Screensize')); 
set(gca, 'FontSize', 50,'FontWeight', 'Bold'); 
xlabel('Iteration #');
ylabel('Eval Return');
l = legend(legends,'Location','NorthWest ','Interpreter','None');
l.NumColumns=2;
ylim([-2000 800]);
saveas(gcf,'EvalReturn_withErrors.png');saveas(gcf,'EvalReturn_withErrors.fig');

%% Eval Return (AVG) 
figure();
style = {'-',':'};
for i = idx
    a = importdata(fol_eval_avg(i).name); 
    j = (i>5) * 1;
    if(j==0) colormap parula; map = colormap; else colormap hot; map = colormap; end
    a = a.data; 
    avg = a(:,3); iter = a(:,2); 
    hold on;
    n = 4;
    i_mod = mod(i,5)+0.1;
    avg = movmean(avg,n); 
    plot(iter(1:2:end), avg(1:2:end),'Color',map(round(i_mod/5*64),:),'linewidth',4,'Markersize',1);
end
grid on;
ax = gca;
ax.LineWidth =5;
set(gcf, 'Position', get(0,'Screensize')); 
set(gca, 'FontSize', 50,'FontWeight', 'Bold'); 
xlabel('Iteration #');
ylabel('Eval Return');
l = legend(legends,'Location','NorthWest ','Interpreter','None');
l.NumColumns=2;
ylim([-1800 500]);
saveas(gcf,'EvalReturn_withoutErrors.png');saveas(gcf,'EvalReturn_withoutErrors.fig');

%% Eval Return Max 
figure();
style = {'-',':'};
for i = idx
    a = importdata(fol_eval_max(i).name); 
    j = (i>5) * 1;
    if(j==0) colormap parula; map = colormap; else colormap hot; map = colormap; end
    a = a.data; 
    avg = a(:,3); iter = a(:,2); 
    hold on;
    n = 4;
    i_mod = mod(i,5)+0.1;
    avg = movmean(avg,n); 
    plot(iter(1:2:end), avg(1:2:end),'Color',map(round(i_mod/5*64),:),'linewidth',4,'Markersize',1);
end
c = {'Differential Q Learning', 'Conventional Q Learning'};
grid on;
ax = gca;
ax.LineWidth =5;
set(gcf, 'Position', get(0,'Screensize')); 
set(gca, 'FontSize', 50,'FontWeight', 'Bold'); 
xlabel('Iteration #');
ylabel('Eval Return Maximum');
l = legend(legends,'Location','NorthWest ','Interpreter','None');
l.NumColumns=2;
ylim([-1800 700]);
saveas(gcf,'EvalReturn_Max.png');saveas(gcf,'EvalReturn_Max.fig');


%% Train Return (AVG+STD) with error bars
figure();
for i = 1:length(fol_train_avg)
    a = importdata(fol_train_avg(i).name); 
    a = a.data; 
    avg = a(:,3); iter = a(:,2); 
    hold on;
    a = importdata(fol_train_std(i).name); 
    a = a.data; 
    std = a(:,3); iter = a(:,2); 
    n = 1;
    avg = movmean(avg,n); 
    std = movmean(std,n);
    errorbar(iter(1:2:end), avg(1:2:end),std(1:2:end),'linewidth',4,'Markersize',10,'CapSize',10);
end
c = {'Differential Q Learning', 'Conventional Q Learning'};
grid on;
ax = gca;
ax.LineWidth =5;
set(gcf, 'Position', get(0,'Screensize')); 
set(gca, 'FontSize', 50,'FontWeight', 'Bold'); 
xlabel('Iteration #');
ylabel('Train Return');
legend(c,'Location','Southeast','Interpreter','None');
saveas(gcf,'TrainReturn_withErrors.png');saveas(gcf,'TrainReturn_withErrors.fig');

%% Train Return (AVG) 
figure();
for i = 1:length(fol_train_avg)
    a = importdata(fol_train_avg(i).name); 
    a = a.data; 
    avg = a(:,3); iter = a(:,2); 
    hold on;
    n = 1;
    plot(iter, movmean(avg,n),'linewidth',6);
end
c = {'Differential Q Learning', 'Conventional Q Learning'};
grid on;
ax = gca;
ax.LineWidth =5;
set(gcf, 'Position', get(0,'Screensize')); 
set(gca, 'FontSize', 50,'FontWeight', 'Bold'); 
xlabel('Iteration #');
ylabel('Train Return');
legend(c,'Location','Southeast','Interpreter','None');
saveas(gcf,'TrainReturn_withoutErrors.png');saveas(gcf,'TrainReturn_withoutErrors.fig');