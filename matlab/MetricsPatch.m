%close all
clc
clear

%%%% SELECT
MODEL = 'E500AE';%'E500IZIf';
model = 'AE';
dataset = 'dataset2';
patch = 32;
latensize = 100;

% DO NOT CHANGE
mode = 'test';

n_thresholds = 1000;
path = './../Result';

%oname = sprintf('%s/%s_score_%s_patch.txt',path,MODEL,mode);
%fileID = fopen( oname, 'w' );

% name = sprintf('%s/%s_novel_%s_%s_P%d_L%d.txt',path,MODEL,mode,dataset,patch,latensize);
% novel = load(name);
% name = sprintf('%s/%s_normal_%s_%s_P%d_L%d.txt',path,MODEL,mode,dataset,patch,latensize);
% normal = load(name);


name = sprintf('%s/%s_novel_%s_%s.txt',path,MODEL,mode,dataset);
novel = load(name);
name = sprintf('%s/%s_normal_%s_%s.txt',path,MODEL,mode,dataset);
normal = load(name);




normal = normal(:,1);
novel = novel(:,1);

[p,n,tp,tn,fp,fn,acc, precision, sensitivity, specificity,fscore,mcc,threshold] = ComputeMetricsPatch( normal, novel, n_thresholds );
 

%%Horizontal: fp, vertical tp 2018Wang_NoveltyDetection, 2019Abati
hold on
plot(fp/n,tp/p,'LineWidth',3,'color','r')
grid on
xlabel('True positive rate ')
ylabel('False positive rate ')
set(gca,'FontSize',18)
auc = abs(trapz(fp/n,tp/p))
plot([0 1],[0 1],'color',[0.5 0.5 0.5])

cmd  = sprintf('%s(AUC=%0.4f)',model,auc);
legend(cmd)

% index = find( acc == max(acc) );
% acc(index)
% precision(index)
% sensitivity(index)
% specificity(index)
% fscore(index)
% mcc(index)



% if length(index) == 1
%     id = index;
% else
%     id = floor((index(end)-index(1))/2);
% end
% fprintf(fileID,'max_acc     = %f\n', acc(id) );
% fprintf(fileID,'Precison    = %f\n', precision(id) );
% fprintf(fileID,'Sensitivity = %f\n', sensitivity(id) );
% fprintf(fileID,'Specificity = %f\n', specificity(id) );
% fprintf(fileID,'Fscore      = %f\n', fscore(id) );
% fprintf(fileID,'MCC         = %f\n', mcc(id) );
% fprintf(fileID,'Threshold   = %f\n', threshold(id));
%   
% fclose(fileID);
% cmd = sprintf('%s is ready!!!',oname);
% disp(cmd)
% 
