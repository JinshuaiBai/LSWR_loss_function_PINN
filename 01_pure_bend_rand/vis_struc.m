clc,clear
sd=10;

load(['out.mat'])
u=double(u);
xy=double(xy);

%% Deformation
figure(1)
subplot(3,2,1)
xy_new=xy+(1*u);
scatter(xy(:,1),xy(:,2),sd),hold on
scatter(xy_new(:,1),xy_new(:,2),sd)
xlabel('{\it x} (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
ylabel('{\it y} (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
box on
axis equal
axis([-0.05 0.55 -.16 0.06])

%% Stress
subplot(3,2,2)
scatter(xy(:,1),xy(:,2),sd,s11,'filled')
xlabel('{\it x} (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
ylabel('{\it y} (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
box on
axis equal
axis([-0.05 0.55 -.1 0.1])
title('\sigma_{xx}')
colorbar
colormap(jet)

subplot(3,2,4)
scatter(xy(:,1),xy(:,2),sd,s22,'filled')
xlabel('{\it x} (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
ylabel('{\it y} (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
box on
axis equal
axis([-0.05 0.55 -.1 0.1])
title('\sigma_{yy}')
colorbar
colormap(jet)
caxis([-2 2])

subplot(3,2,6)
scatter(xy(:,1),xy(:,2),sd,s12,'filled')
xlabel('{\it x} (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
ylabel('{\it y} (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
box on
axis equal
axis([-0.05 0.55 -.1 0.1])
title('\tau_{xy}')
colorbar
colormap(jet)
caxis([-2 2])

%% Displacment
subplot(3,2,3)
scatter(xy(:,1),xy(:,2),sd,u(:,1),'filled')
xlabel('{\it x} (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
ylabel('{\it y} (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
box on
axis equal
axis([-0.05 0.55 -.1 0.1])
title('U')
colorbar
colormap(jet)

subplot(3,2,5)
scatter(xy(:,1),xy(:,2),sd,u(:,2),'filled')
xlabel('{\it x} (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
ylabel('{\it y} (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
box on
axis equal
axis([-0.05 0.55 -.1 0.1])
title('V')
colorbar
colormap(jet)