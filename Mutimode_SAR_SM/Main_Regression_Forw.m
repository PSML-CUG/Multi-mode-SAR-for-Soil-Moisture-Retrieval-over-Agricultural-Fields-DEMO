%%% Function: Forward selection procedure to find the optimal feature combination;

%% Setup
clear;clc;close all;

fontname = 'Bookman';
fontsize = 20;
fontunits = 'points';
set(0,'DefaultAxesFontName',fontname,'DefaultAxesFontSize',fontsize,'DefaultAxesFontUnits',fontunits,...
    'DefaultTextFontName',fontname,'DefaultTextFontSize',fontsize,'DefaultTextFontUnits',fontunits,...
    'DefaultLineLineWidth',3,'DefaultLineMarkerSize',10,'DefaultLineColor',[0 0 0]);

% Paths
addpath('./AUXF')       % Auxiliary functions for visualization, results analysis, plots, etc.
addpath('./DATA')       % Put your data here
% addpath('./FIGURES')    % All figures are saved here
% addpath('./RESULTS')    % All files with results are saved here

% Paths for the methods
addpath('./standard')   % Train-Test functions for all methods
addpath('./SVM')        % libsvm code and kernel matrix
addpath('./MRVM')       % Relevance vector machine (RVM)
addpath('./VHGPR')      % Variational Heteroscedastic Gaussian Process regression [Lázaro-Gredilla, 2011]
addpath('./ARES')       % ARESLab -- Adaptive Regression Splines toolbox for Matlab/Octave, ver. 1.5.1, by Gints Jekabsons
addpath('./LWP')        % Locally-Weighted Polynomials, Version 1.3, by Gints Jekabsons
addpath('./WGP')        % Warped GPs
addpath('./SSGP')       % Sparse Spectrum Gaussian Process (SSGP)  [Lázaro-Gredilla, 2008]
addpath('./TGP')        % Twin Gaussian Process (TGP) [Liefeng Bo and Cristian Sminchisescu]  http://www.maths.lth.se/matematiklth/personal/sminchis/code/TGP.html
addpath('./XGB')        % Extreme Gradient Boosting Trees
addpath(genpath('./CCFS/src')); % Canonical Correlation Forests

%% Load DATA.mat
%   X: Input data of size n x d
%   Y: Output/target/observation of size n x do
%   n: number of samples/examples/patterns (in rows)
%   d: input data dimensionality/features (in columns)
%   do: output data dimensionality (variables, observations).
% load DATA_corn_SM.mat;
% X = Features;
% Y = Soil_Moisture;

%% Split training-testing data (Stratified Sampling)
% rate = 0.75;
% restrate = 0.25;
% [num_point,num_Feature] = size(Features);       % samples x bands
% 
% Z = [Y,X];
% % Z = sortrows(Z,1);
% Z_0_15 = Z((Z(:,1)<=15 & Z(:,1)>0),:);
% Z_15_25 = Z((Z(:,1)<=25 & Z(:,1)>15),:);
% Z_25_35 = Z((Z(:,1)<=35 & Z(:,1)>25),:);
% Z_35_50 = Z((Z(:,1)<=50 & Z(:,1)>35),:);
% labels = {Z_0_15,Z_15_25,Z_25_35,Z_35_50};
% 
% Xtrain = [];
% Ytrain = [];
% Xtest = [];
% Ytest = [];
% for i = 1:length(labels)
%     Z_cate = cell2mat(labels(i));
%     [Z_cate_num,~] = size(Z_cate);   
%     r = randperm(Z_cate_num);                 % random index
%     ntrain = round(rate*Z_cate_num);          % training samples
%     Xtrain_cate = Z_cate(r(1:ntrain),2:end);       % training set
%     Ytrain_cate = Z_cate(r(1:ntrain),1);       % observed training variable
%     Xtest_cate  = Z_cate(r(ntrain+1:end),2:end);   % test set
%     Ytest_cate  = Z_cate(r(ntrain+1:end),1);   % observed test variable
%     Xtrain = [Xtrain;Xtrain_cate];
%     Ytrain = [Ytrain;Ytrain_cate];
%     Xtest = [Xtest;Xtest_cate];
%     Ytest = [Ytest;Ytest_cate];
% end

%Import all model samples and divide them into 3 models for training and testing
load DATA_soybean_SM_DP_73_save_RF5.mat;     %PUT YOUR DATA
%CP(test and train)
%Xtest_CP=Xtest(:,1:34);
%Ytest_CP=Ytest;
%Xtrain_CP=Xtrain(:,1:34);
%Ytrain_CP=Ytrain

%DP(test and train)
% Xtest_DP=Xtest(:,35:46);
% Ytest=Ytest;
% Xtrain=Xtrain(:,35:46);
% Ytrain=Ytrain

%FP(test and train)
%Xtest_FP=Xtest(:,47:78);
%Ytest_FP=Ytest;
%Xtrain_FP=Xtrain(:,47:78);
%Ytrain_FP=Ytrain

[~,num_Feature] = size(Xtrain);    
[ntrain,~] = size(Ytrain);
[ntest,~] = size(Ytest);

%% Split training-testing data (Random Sampling)
% r = randperm(num_point);                 % random index
% ntrain = round(rate*num_point);          % training samples
% Xtrain = X(r(1:ntrain),:);       % training set
% Ytrain = Y(r(1:ntrain),:);       % observed training variable
% Xtest  = X(r(ntrain+1:end),:);   % test set
% Ytest  = Y(r(ntrain+1:end),:);   % observed test variable
% [ntest,do] = size(Ytest);

%% Remove the mean of Y for training only
my      = mean(Ytrain);
Ytrain  = Ytrain - repmat(my,ntrain,1);

%% SELECT METHODS FOR COMPARISON
METHODS = {'GPR'};
%METHODS = {'LASSO';'XGB';'WKNNR';'WGPR';'VHGPR';'TREE';'TGP';'SVR';'SSGPR';'SKRRlin';'RVM';'RKS';'RF';'NN';'MSVR';'LSBoost';'KRR';'KNNR';'GPR';'ENET';'ELM';'CCF';'BOOST';'BAGTREE';'ARES'};
%% FORWARD SELECTION PROCEDURE
numModels = numel(METHODS);
%FP
% Feature_name=["C11";"C22";"C33";"T11";"T22";"Span";"Ra_HHVV";"Ra_HVHH";"Ra_HVVV";"Mo_HHVV";"Mo_HVVV";"Mo_HHHV";"Mo_T1T2";"Ph_HHVV";...
%    "Ph_HVVV";"Ph_HHHV";"Ph_T1T2";"Ps";"Pd";"Pv";"Ps_f";"Pd_f";"Pv_f";"Pc_f";"Entropy";"Anisotropy";"Alpha";"Ne_del_m";"Ne_de1_p";"Ne_tau";"RVI";"GRVI"];
% CP
% Feature_name={'SV0','SV1','SV2','SV3','Sigma_RH','Sigma_RV','Sigma_RL','Sigma_RR','Entropy','Anisotropy','Alpha',...
%               'M_Chi_S','M_Chi_D','M_Chi_V','M_Delta_S','M_Delta_D','M_Delta_V','MF3_eve','MF3_odd','MF3_diff','MF3_theta',...
%                'SE_Pol','SE_Int','DoP','DoD','DoC','DoE','ConCoef','Alpha_s','PD_RHRV','CO_RHRV','LPR','CPR','CpRVI'};
%DP
Feature_name={'VV','VH','Ra_VHVV','Span','H','A','Alpha','RVI','DpRVI','M_c','H_c','Theta_c','ms','mv'};
Feature_order=[]; 

for m=1:numModels
    
    RMSE_box = zeros(num_Feature,num_Feature);
    R_box = zeros(num_Feature,num_Feature);
    Xtrain_minRMSE = [];
    Xtest_minRMSE = [];
    RMSE_min = 1000;
    F_index = zeros(1,num_Feature);
    
    disp('Program running, please wait...');
    progressbar('Main Iteration Progress','Secondary Iteration Progress');
    for i=1:num_Feature
        Rmse_min = 1000;
        for j=1:(num_Feature-i+1)
            
            XTRAIN = [Xtrain_minRMSE,Xtrain(:,j)];
            XTEST = [Xtest_minRMSE,Xtest(:,j)];
            eval(['model = train' METHODS{m} '(XTRAIN,Ytrain);']); % Train the model
            eval(['Yp = test' METHODS{m} '(model,XTEST);']);       % Test the model
            Yp = Yp + repmat(my,ntest,1);
            RESULTS = assessment(Ytest, Yp, 'regress');
            RMSE_box(j,i) = RESULTS.RMSE;
            R_box(j,i) = RESULTS.R;
            
            if RESULTS.RMSE < Rmse_min
                XTRAIN_minRMSE = XTRAIN;
                XTEST_minRMSE = XTEST;
                F_index(i) = j;
                Results_minRMSE = RESULTS;
                Rmse_min = RESULTS.RMSE;
                Yp_min = Yp;
            end
            
            if RESULTS.RMSE < RMSE_min
                RESULTS_minRMSE = RESULTS;
                RMSE_min = RESULTS.RMSE;
                YP_minRMSE = Yp;
            end
                        
            frac2 = j/(num_Feature-i+1);
            frac1 = ((i-1) + frac2)/num_Feature;
            progressbar(frac1, frac2);
        end
        Feature_order=[Feature_order;Feature_name(F_index(i))];
        Xtrain(:,F_index(i)) = [];
        Xtest(:,F_index(i)) = [];
        Feature_name(F_index(i)) = [];
        Xtrain_minRMSE = XTRAIN_minRMSE;
        Xtest_minRMSE = XTEST_minRMSE;
       % ScatPlot(METHODS,Ytest,Yp_min,Results_minRMSE);
    end
    
    % Plot the optimal feature combination
    Plot_Scat(METHODS,Ytest,YP_minRMSE,RESULTS_minRMSE);
    
    RMSE_box(RMSE_box==0)=Inf;
    RMSE_box = sort(RMSE_box);
    R_box(R_box==0)=-Inf;
    R_box = sort(R_box,'descend');
    [~,num_OptComb_RMSE]=find(RMSE_box==min(min(RMSE_box)));
    [~,num_OptComb_R]=find(R_box==max(max(R_box)));
    Feature_OptComb=Feature_order(1:num_OptComb_RMSE(1));
    
    disp('The optimal feature combination is : ');
    disp(Feature_OptComb);
    
    %% Plot and save: feature combination analysis with RMSE and R
    %write_path='E:\Results\DUAL-POL\soybean';
    %cd(write_path);
    xlswrite(['E:\Chapter1 Results\Results\FFS_ML\DP\soybean\GPR55\',METHODS{m},'.xlsx'],["ME";"RMSE";"RELRMSE";"MAE";"R";"RP";"R2"],'RESULTS','A1');
    xlswrite(['E:\Chapter1 Results\Results\FFS_ML\DP\soybean\GPR55',METHODS{m},'.xlsx'],struct2cell(RESULTS_minRMSE),'RESULTS','B1');
    xlswrite(['E:\Chapter1 Results\Results\FFS_ML\DP\soybean\GPR55\',METHODS{m},'.xlsx'],["Measured SM";Ytest],'Scat_Plot','A1');
    xlswrite(['E:\Chapter1 Results\Results\FFS_ML\DP\soybean\GPR55\',METHODS{m},'.xlsx'],["Estimated SM";YP_minRMSE],'Scat_Plot','B1');
    xlswrite(['E:\Chapter1 Results\Results\FFS_ML\DP\soybean\GPR55\',METHODS{m},'.xlsx'],["Feature_order";Feature_order],'Feature_comb','A1');
    xlswrite(['E:\Chapter1 Results\Results\FFS_ML\DP\soybean\GPR55\',METHODS{m},'.xlsx'],["Feature_OptComb";Feature_OptComb],'Feature_comb','B1');
    xlswrite(['E:\Chapter1 Results\Results\FFS_ML\DP\soybean\GPR55\',METHODS{m},'.xlsx'],RMSE_box,'RMSE_box','A1');
    xlswrite(['E:\Chapter1 Results\Results\FFS_ML\DP\soybean\GPR55\',METHODS{m},'.xlsx'],R_box,'R_box','A1');
    %cd(path);
    
    figure();
    x=1:1:num_Feature;
    y_minRMSE=RMSE_box(1,:);
    % y_others=RMSE_box(2:end,:);
    % plot(x,y_others,'o','LineWidth',0.5,'MarkerSize',5,'MarkerEdgeColor',[180 180 180]/255);
    plot(x,y_minRMSE,'o-','color',[202,62,71]/255,'LineWidth',0.5,'MarkerSize',5,'MarkerEdgeColor',[202,62,71]/255,'MarkerFaceColor',[202,62,71]/255);
    hold on;
    axis([0 33 2 8]);
    set(gca,'xTick',(0:5:33),'yTick',(2:1:8))
    set(gca,'fontsize',14,'fontname','Times New Roman');
    xlabel('Number of Features');
    ylabel('RMSE');
    MinRMSE=min(min(RMSE_box));
    plot(num_OptComb_RMSE(1),MinRMSE,'o','LineWidth',0.5,'MarkerSize',5.2,'MarkerEdgeColor',[65,65,65]/255,'MarkerFaceColor',[65,65,65]/255);
    text(num_OptComb_RMSE(1)-1.2,MinRMSE-0.3,num2str(MinRMSE,'%.2f'),'FontSize',14,'fontname','Times New Roman');
    grid;
    
    figure();
    x=1:1:num_Feature;
    y_minR=R_box(1,:);
    y_others=R_box(2:end,:);
    plot(x,y_others,'o','LineWidth',0.5,'MarkerSize',5,'MarkerEdgeColor',[180 180 180]/255);
    plot(x,y_minR,'o-','color',[202,62,71]/255,'LineWidth',0.5,'MarkerSize',5,'MarkerEdgeColor',[202,62,71]/255,'MarkerFaceColor',[202,62,71]/255);
    hold on;
    axis([0 33 0.3 0.9]);
    set(gca,'xTick',(0:5:33),'yTick',(0.3:0.1:0.9));
    set(gca,'fontsize',14,'fontname','Times New Roman');
    xlabel('Number of Features');
    ylabel('R');
    MaxR=max(max(R_box));
    plot(num_OptComb_R(1),MaxR,'o','LineWidth',0.5,'MarkerSize',5.2,'MarkerEdgeColor',[65,65,65]/255,'MarkerFaceColor',[65,65,65]/255);
    text(num_OptComb_R(1)-1.2,MaxR+0.035,num2str(MaxR,'%.2f'),'FontSize',14,'fontname','Times New Roman');
    grid;
    
    print(figure(1),['E:\Chapter1 Results\Results\FFS_ML\DP\soybean\GPR55\Scatplot_OptComb_',METHODS{m},'.jpg'],'-djpeg','-r600');
    print(figure(2),['E:\Chapter1 Results\Results\FFS_ML\DP\soybean\GPR55\Analysi_RMSE_',METHODS{m},'.jpg'],'-djpeg','-r600');
    print(figure(3),['E:\Chapter1 Results\Results\FFS_ML\DP\soybean\GPR55\Analysi_R_',METHODS{m},'.jpg'],'-djpeg','-r600');
end

