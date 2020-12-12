%close all
clc

%%%% SELECT
EPOCHS = 500;
MODEL = 'E500AUTOENC';%'E500IZIf';
model = 'AENC';
dataset = 'dataset3';

% DO NOT CHANGE
mode = 'Test';

n_thresholds = 10000;
path = sprintf('./../E%dResult', EPOCHS );

oname = sprintf('%s/%s_metrics_%s_patch.txt',path,MODEL,mode);
%fileID = fopen( oname, 'w' );

%fprintf(fileID,'\n%s\n',mode);
name = sprintf('%s/%s_novel_%s_%s.txt',path,MODEL,mode,dataset);
novel = load(name);
name = sprintf('%s/%s_normal_%s_%s.txt',path,MODEL,mode,dataset);
normal = load(name);


normal = normal(:,1);
novel = novel(:,1);

[p,n,tp,tn,fp,fn,acc, precision, sensitivity, specificity,fscore,mcc,threshold] = ComputeMetricsPatch( normal, novel, n_thresholds );
 
% 
% hold on
% plot(fn/p,tn/n,'LineWidth',3,'color','r')
% grid on
% xlabel('False Negative Rate ')
% ylabel('True Negative Rate ')
% set(gca,'FontSize',18)
% auc = abs(trapz(fn/p,tn/n))
% plot([0 1],[0 1],'color',[0.5 0.5 0.5])
% cmd  = sprintf('izi_f(AUC=%0.4f)',auc)
% legend(cmd)


%%Horizontal: fp, vertical tp 2018Wang_NoveltyDetection, 2019Abati
hold on
plot(fp/n,tp/p,'LineWidth',3,'color','r')
grid on
xlabel('True positive rate ')
ylabel('False positive rate ')
set(gca,'FontSize',18)
auc = abs(trapz(fp/n,tp/p))
plot([0 1],[0 1],'color',[0.5 0.5 0.5])


cmd  = sprintf('%s(AUC=%0.4f)',model,auc)
legend(cmd)




%name = sprintf('%s/%s_auc_%s_patch.png', path,MODEL,mode );
%saveas(gcf,name)


% fprintf(fileID,'\nMaximum acc\n');
% index = find( acc == max(acc) );
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
