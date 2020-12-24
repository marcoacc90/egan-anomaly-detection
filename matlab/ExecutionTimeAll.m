%close all
clc
clear
close all

%%%% SELECT

model = {'BIGAN','ZIZ','AE','IZI','IZIf'};
model_id = 'E500';


% DO NOT CHANGE
mode = 'test';
n_thresholds = 1000;
path = './../Result';
auc = zeros(1,length(model));

for i = 1 : length( model )

    name = sprintf('%s/%s%s_novel_%s_dataset1.txt',path,model_id,model{i},mode);
    novel_d1 = load(name);
    name = sprintf('%s/%s%s_normal_%s_dataset1.txt',path,model_id,model{i},mode);
    normal_d1 = load(name);

    name = sprintf('%s/%s%s_novel_%s_dataset2.txt',path,model_id,model{i},mode);
    novel_d2 = load(name);
    name = sprintf('%s/%s%s_normal_%s_dataset2.txt',path,model_id,model{i},mode);
    normal_d2 = load(name);

    if strcmp(model{i},'BIGAN')
      mytime = [ normal_d1(:,4); novel_d1(:,4); normal_d2(:,4); novel_d2(:,4) ];
    else
      mytime = [ normal_d1(:,2); novel_d1(:,2); normal_d2(:,2); novel_d2(:,2) ];
    end
    mean( mytime )  * 1000
    
end
