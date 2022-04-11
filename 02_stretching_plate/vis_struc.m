clc,clear
sd=3;

load(['out.mat'])
u=double(u);
xy=double(xy);

%% Deformation
figure(1)
subplot(3,2,1)
xy_new=xy+(20*u);
scatter(xy(:,1),xy(:,2),1),hold on
scatter(xy_new(:,1),xy_new(:,2),1)
xlabel('{\it x} (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
ylabel('{\it y} (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
box on
axis equal
axis([-0.05 1.4 -.05 1.05])

%% Stress
subplot(3,2,2)
scatter(xy(:,1),xy(:,2),sd,s11,'filled')
xlabel('{\it x} (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
ylabel('{\it y} (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
box on
axis equal
axis([-0.05 1.05 -.05 1.05])
title('\sigma_{xx}')
colorbar
colormap(jet)
caxis([0. 1.])

subplot(3,2,4)
scatter(xy(:,1),xy(:,2),sd,s22,'filled')
xlabel('{\it x} (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
ylabel('{\it y} (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
box on
axis equal
axis([-0.05 1.05 -.05 1.05])
title('\sigma_{yy}')
colorbar
colormap(jet)
caxis([-0.123629000000000 0.394693000000000])

subplot(3,2,6)
scatter(xy(:,1),xy(:,2),sd,s12,'filled')
xlabel('{\it x} (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
ylabel('{\it y} (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
box on
axis equal
axis([-0.05 1.05 -.05 1.05])
title('\tau_{xy}')
colorbar
colormap(jet)
caxis([-0.106118000000000 0.001815130000000])
%%
%     figure(4)
subplot(3,2,3)
scatter(xy(:,1),xy(:,2),sd,u(:,1),'filled')
xlabel('{\it x} (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
ylabel('{\it y} (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
box on
axis equal
axis([-0.05 1.05 -.05 1.05])
title('U')
colorbar
colormap(jet)
caxis([0 0.015135000000000])

subplot(3,2,5)
scatter(xy(:,1),xy(:,2),sd,u(:,2),'filled')
xlabel('{\it x} (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
ylabel('{\it y} (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
box on
axis equal
axis([-0.05 1.05 -.05 1.05])
title('V')
colorbar
colormap(jet)
caxis([-0.003923100000000 2.535000000000010e-04])

