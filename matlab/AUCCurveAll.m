%close all
clc
clear
close all

%%%% SELECT

model = {'BIGAN','ZIZ','AE','IZI','IZIf'};
model_id = 'E500'; 
dataset = 'dataset2';



% DO NOT CHANGE
mode = 'test';
n_thresholds = 1000;
path = './../Result';
auc = zeros(1,length(model));
leg = cell(1,length(model)+1);
for i = 1 : length( model )
    name = sprintf('%s/%s%s_novel_%s_%s.txt',path,model_id,model{i},mode,dataset);
    novel = load(name);
    name = sprintf('%s/%s%s_normal_%s_%s.txt',path,model_id,model{i},mode,dataset);
    normal = load(name);
    
    if strcmp(model{i},'BIGAN')
      normal = normal(:,2);     
      novel = novel(:,2);
    else
      normal = normal(:,1);
      novel = novel(:,1);        
    end
    [p,n,tp,tn,fp,fn,acc, precision, sensitivity, specificity,fscore,mcc,threshold] = ComputeMetricsPatch( normal, novel, n_thresholds );
    
    
    disp(model{i})
    [value,id] = max(acc)
    
    mythreshold = threshold(id)
    
    auc(i) = abs(trapz(fp/n,tp/p));
    
    
    
    if strcmp(model{i},'IZIf')
        cmd = sprintf('f-AnoGAN (%0.4f)',auc(i));
    else
        cmd = sprintf('%s (%0.4f)',model{i},auc(i));
    end
    leg{ i } = cmd;
    plot(fp/n,tp/p,'-','LineWidth',3)
    hold on 
end
grid on
xlabel('False positive rate')
ylabel('True positive rate')
set(gca,'FontSize',18)
axis([0 1 0 1.01])
plot([0 1],[0 1],'LineWidth',1,'color',[0.5 0.5 0.5])
leg{length(model)+1} = 'Random';
legend( leg, 'FontSize',18)
