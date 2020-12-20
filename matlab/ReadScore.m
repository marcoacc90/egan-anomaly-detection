clear all
clc
close all

%%%% SELECT
folder = 'Result/';
MODEL = 'E500IZIf'%'E500IZIf', E500AE
test = 'test';
dataset = 'dataset1'
bin = 100;
P = 28;
L = 1000;

path = sprintf('./../%s', folder );
name = sprintf('%s/%s_normal_training_%s_P%d_L%d.txt',path,MODEL,dataset,P,L);
normal_train = load(name);
name = sprintf('%s/%s_normal_test_%s_P%d_L%d.txt',path,MODEL,dataset,P,L);
normal_test = load(name);
name = sprintf('%s/%s_novel_test_%s_P%d_L%d.txt',path,MODEL,dataset,P,L);
novel_test = load(name);


histogram(normal_train(:,1),bin,'Normalization','probability')
hold on
histogram(normal_test(:,1),bin,'Normalization','probability')
histogram(novel_test(:,1),bin,'Normalization','probability')

%title(mode)
xlabel('Anomaly Score')
ylabel('h')
grid on
legend('Normal Training','Normal Test', 'Anomaly Test')
%set(gca,'FontSize',18)
%name = sprintf('%s/%s_hist_%s.png', path,MODEL,mode );
%saveas(gcf,name)
