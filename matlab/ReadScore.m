clear all
clc
close all

%%%% SELECT
folder = 'Result/';
MODEL = 'E500IZIf';%'E500IZIf', E500AE
test = 'test';
dataset = 'dataset2';

path = sprintf('./../%s', folder );
name = sprintf('%s/%s_normal_training_%s.txt',path,MODEL,dataset);
normal_train = load(name);
name = sprintf('%s/%s_normal_test_%s.txt',path,MODEL,dataset);
normal_test = load(name);
name = sprintf('%s/%s_novel_test_%s.txt',path,MODEL,dataset);
novel_test = load(name);

histogram(normal_train(:,1),10,'Normalization','probability')
hold on
histogram(normal_test(:,1),10,'Normalization','probability')
histogram(novel_test(:,1),30,'Normalization','probability')

axis([-0.05 1 0 1.05])
%title(mode)
xlabel('Anomaly Score')
ylabel('h')
grid on
legend('Normal (training)','Normal (test)', 'Anomaly (test)')
set(gca,'FontSize',18)
