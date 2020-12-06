clear all
clc
close all

%%%% SELECT
folder = 'E500Result/';
MODEL = 'E500IZIf';
test = 'test';
dataset = 'dataset2'
bin = 50;

path = sprintf('./../%s', folder );
name = sprintf('%s/%s_novel_%s_%s.txt',path,MODEL,test,dataset);
novel = load(name);
name = sprintf('%s/%s_normal_%s_%s.txt',path,MODEL,test,dataset);
normal = load(name);

histogram(normal(1:2000,1),bin)
hold on
histogram(novel(:,1),bin)
total_data = length(normal(:,1)) + length(novel(:,1))

%title(mode)
xlabel('Anomaly Score')
ylabel('Patches')
grid on
legend('Normal','Anomaly')
set(gca,'FontSize',18)
name = sprintf('%s/%s_hist_%s.png', path,MODEL,mode );
saveas(gcf,name)
