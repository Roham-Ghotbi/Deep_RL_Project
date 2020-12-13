%% HW 5

%% HW 5
clear all; close all;
home =  '/Users/chnajafi94/Documents/GitHub/cs285Project/cs285/GrpA-LunarLanderContinuous';
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
%% Actor Loss
figure();
for i = 1:length(fol_ac_loss)
    a = importdata(fol_ac_loss(i).name); 
    a = a.data; 
    data = a(:,3); iter = a(:,2); 
    hold on;
    plot(iter, data,'linewidth',6);
end
c = {'Differential Q Learning', 'Conventional Q Learning'};
grid on;
ax = gca;
ax.LineWidth =5;
set(gcf, 'Position', get(0,'Screensize')); 
set(gca, 'FontSize', 50,'FontWeight', 'Bold'); 
xlabel('Iteration #');
ylabel('Actor Loss');
legend(c,'Location','NorthEast','Interpreter','None');
saveas(gcf,'AC_Loss.png');saveas(gcf,'AC_Loss.fig');

%% Critic Loss
figure();
for i = 1:length(fol_crit_loss)
    a = importdata(fol_crit_loss(i).name); 
    a = a.data; 
    data = a(:,3); iter = a(:,2); 
    hold on;
    plot(iter, movmean(data,2),'linewidth',6);
end
c = {'Differential Q Learning', 'Conventional Q Learning'};
grid on;
ax = gca;
ax.LineWidth =5;
set(gcf, 'Position', get(0,'Screensize')); 
set(gca, 'FontSize', 50,'FontWeight', 'Bold'); 
xlabel('Iteration #');
ylabel('Critic Loss');
legend(c,'Location','NorthEast','Interpreter','None');
set(gca, 'YScale', 'log');
saveas(gcf,'CRITIC_Loss.png');saveas(gcf,'CRITIC_Loss.fig');

%% Eval Return (AVG+STD) with error bars
figure();
for i = 1:length(fol_eval_avg)
    a = importdata(fol_eval_avg(i).name); 
    a = a.data; 
    avg = a(:,3); iter = a(:,2); 
    hold on;
    a = importdata(fol_eval_std(i).name); 
    a = a.data; 
    std = a(:,3); iter = a(:,2); 
    n = 2;
    errorbar(iter, movmean(avg,n),movmean(std,n),'linewidth',4,'Markersize',10,'CapSize',10);
end
c = {'Differential Q Learning', 'Conventional Q Learning'};
grid on;
ax = gca;
ax.LineWidth =5;
set(gcf, 'Position', get(0,'Screensize')); 
set(gca, 'FontSize', 50,'FontWeight', 'Bold'); 
xlabel('Iteration #');
ylabel('Eval Return');
legend(c,'Location','Southeast','Interpreter','None');
saveas(gcf,'EvalReturn_withErrors.png');saveas(gcf,'EvalReturn_withErrors.fig');

%% Eval Return (AVG) 
figure();
for i = 1:length(fol_eval_avg)
    a = importdata(fol_eval_avg(i).name); 
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
ylabel('Eval Return');
legend(c,'Location','Southeast','Interpreter','None');
saveas(gcf,'EvalReturn_withoutErrors.png');saveas(gcf,'EvalReturn_withoutErrors.fig');

%% Eval Return Max 
figure();
for i = 1:length(fol_eval_max)
    a = importdata(fol_eval_max(i).name); 
    a = a.data; 
    avg = a(:,3); iter = a(:,2); 
    hold on;
    n = 2;
    plot(iter, movmean(avg,n),'linewidth',6);
end
c = {'Differential Q Learning', 'Conventional Q Learning'};
grid on;
ax = gca;
ax.LineWidth =5;
set(gcf, 'Position', get(0,'Screensize')); 
set(gca, 'FontSize', 50,'FontWeight', 'Bold'); 
xlabel('Iteration #');
ylabel('Eval Return Maximum');
legend(c,'Location','Southeast','Interpreter','None');
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
    errorbar(iter, movmean(avg,n),movmean(std,n),'linewidth',4,'Markersize',10,'CapSize',10);
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