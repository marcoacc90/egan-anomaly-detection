close all
% 10, 100, 1000
auc = [0.9980, 0.9980, 0.9968;   %24
       0.9981, 0.9983, 0.9984;   %28
       0.9988, 0.9991, 0.9984];  %32

   % 24, 28, 32
auc1 = auc'   % 10, 100, 1000
   
colormap autumn(10)
bar3(auc1,0.5)
%xlabel('Latent vector size (log)')
%ylabel('Patch size')

xlabel('Patch size')
ylabel('Size of the latent representation')

zlabel('AUC')
grid on
axis([0 4 0 4 0.995 0.9992])
set(gca,'FontSize',18)