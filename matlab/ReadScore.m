clear all
clc
close all

%%%% SELECT
folder = 'E500Result/';
MODEL = 'E500IZIf';
test = 'dmm';
bin = 50;

path = sprintf('./../%s', folder );
name = sprintf('%s/%s_score_novel_%s.txt',path,MODEL,test);
novel = load(name);
name = sprintf('%s/%s_score_normal_%s.txt',path,MODEL,test);
normal = load(name);

%histogram(normal(:,1),bin)
%hold on
%histogram(novel(:,1),bin)
%total_data = length(normal(:,1)) + length(novel(:,1))

histogram(normal(:),bin)
hold on
histogram(novel(:),bin)
total_data = length(normal(:)) + length(novel(:))



%title(mode)
xlabel('Anomaly Score')
ylabel('Patches')
grid on
legend('Normal','Anomaly')
set(gca,'FontSize',18)
name = sprintf('%s/%s_hist_%s.png', path,MODEL,mode );
saveas(gcf,name)
