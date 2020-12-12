clear all
clc
close all

%%%% SELECT
folder = 'E500Result/';
MODEL = 'E500IZIf'%'E500IZIf', E500AE
test = 'test';
dataset = 'dataset2'
bin = 100;

path = sprintf('./../%s', folder );
name = sprintf('%s/%s_novel_%s_%s.txt',path,MODEL,test,dataset);
novel = load(name);
name = sprintf('%s/%s_normal_%s_%s.txt',path,MODEL,test,dataset);
normal = load(name);

histogram(normal(:,1),bin,'Normalization','probability')
hold on
histogram(novel(:,1),bin,'Normalization','probability')
total_data = length(normal(:,1)) + length(novel(:,1))

%title(mode)
xlabel('Anomaly Score')
ylabel('h')
grid on
legend('Normal','Anomaly')
set(gca,'FontSize',18)
name = sprintf('%s/%s_hist_%s.png', path,MODEL,mode );
saveas(gcf,name)
