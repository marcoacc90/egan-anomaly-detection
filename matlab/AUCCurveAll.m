%close all
clc
clear

%%%% SELECT

model = {'AE','IZI','IZIf'};
model_id = 'E500'; 
dataset = 'dataset2';

% DO NOT CHANGE
mode = 'test';
n_thresholds = 1000;
path = './../Result';

color = {'r','b','g','y'};
auc = zeros(1,length(model));
for i = 1 : length( model )
    name = sprintf('%s/%s%s_novel_%s_%s.txt',path,model_id,model{i},mode,dataset);
    novel = load(name);
    name = sprintf('%s/%s%s_normal_%s_%s.txt',path,model_id,model{i},mode,dataset);
    normal = load(name);
    
    normal = normal(:,1);
    novel = novel(:,1);
    
    [p,n,tp,tn,fp,fn,acc, precision, sensitivity, specificity,fscore,mcc,threshold] = ComputeMetricsPatch( normal, novel, n_thresholds );
    plot(fp/n,tp/p,'LineWidth',3,'color',color{i})
    hold on
    auc(i) = abs(trapz(fp/n,tp/p));
end
grid on
auc
xlabel('True positive rate')
ylabel('False positive rate')
set(gca,'FontSize',18)

%cmd = sprintf('%s(AUC=%0.4f)',model,auc);
% legend(cmd)
% %%Horizontal: fp, vertical tp 2018Wang_NoveltyDetection, 2019Abati
% plot([0 1],[0 1],'color',[0.5 0.5 0.5])
% 
