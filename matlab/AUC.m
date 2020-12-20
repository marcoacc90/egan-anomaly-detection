close all
% 10, 100, 1000
auc = [0.9980, 0.9980, 0.9968;   %24
       0.9981, 0.9983, 0.9984];  %28

colormap autumn(10)
bar3(auc)
xlabel('Latent vector size (log)')
ylabel('Patch size')
zlabel('AUC')
grid on
axis([0 4 0 3 0.995 0.999])
set(gca,'FontSize',18)