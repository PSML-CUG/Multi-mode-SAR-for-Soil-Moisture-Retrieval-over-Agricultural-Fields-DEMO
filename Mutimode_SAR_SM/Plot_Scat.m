function Plot_Scat(Name,Ytest, Ypred,RESULTS)
figure();
x_line=0:0.1:50;
y_line=x_line;
plot(x_line,y_line,'--','LineWidth',0.5,'Color',[120,120,120]/255);
hold on;

x_ubline=0:0.1:40;
y_ubline=x_ubline+10;
plot(x_ubline,y_ubline,'--','LineWidth',0.5,'Color',[120,120,120]/255);

x_lbline=10:0.1:50;
y_lbline=x_lbline-10;
plot(x_lbline,y_lbline,'--','LineWidth',0.5,'Color',[120,120,120]/255);

plot (Ytest,Ypred,'o','color',[0 114 189]/255,'LineWidth', 0.5,'MarkerSize', 6,'MarkerFaceColor',[0 114 189]/255);
set(gca,'LineWidth',1,'fontsize',14,'fontname' ,'Times New Roman');
xlabel('Measured SM [vol.%]');
ylabel('Estimated SM [vol.%]');
str = {['ME= ',num2str(RESULTS.ME,'%.2f')],['RMSE= ',num2str(RESULTS.RMSE,'%.2f')],['MAE= ',num2str(RESULTS.MAE,'%.2f')],['R= ',num2str(RESULTS.R,'%.2f')]};

text(3,40,str,'k','FontSize',14,'fontname','Times New Roman');
title (Name);
% grid;
end