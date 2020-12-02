clear all
clc
close all

%%%% SELECT
folder = 'E500Result/';
MODEL = 'E500IZI';
dataset = 'dataset';
mode = 'Test';
bin = 50;

path = sprintf('./../%s', folder );
name = sprintf('%s/%s_score_novel_%s.txt',path,MODEL,dataset);
novel = load(name);
name = sprintf('%s/%s_score_normal_%s.txt',path,MODEL,dataset);
normal = load(name);

histogram(normal(:,1),bin)
hold on
histogram(novel(:,1),bin)

%title(mode)
xlabel('Anomaly Score')
ylabel('Patches')
grid on
legend('Normal','Anomaly')
set(gca,'FontSize',18)
name = sprintf('%s/%s_hist_%s_patch.png', path,MODEL,mode );
saveas(gcf,name)
