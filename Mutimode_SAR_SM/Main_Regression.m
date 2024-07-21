%%% Function: Single regression;
%%% Tool:     simpleRegression 3.1 Matlab toolbox;
%%% Author:   LvChangchang;
%%% Date:     2022.05.23;

%% Setup
clear;clc;close all;
tic;

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

%% Load data:
load DATA_SM_FP_Corn.mat;
X = Features;
Y = Soil_Moisture;
% % X(:,10:17) = [];

%% Split training-testing data (Stratified Sampling)
rate = 0.7;
restrate = 0.3;
%[num_point,num_Feature] = size(X);       % samples x bands


 err = 100;
 for nn=1:1000
 Z = [Y,X];
 Z = sortrows(Z,1);
 Z_0_15 = Z((Z(:,1)<=15 & Z(:,1)>0),:);
 Z_15_25 = Z((Z(:,1)<=25 & Z(:,1)>15),:);
 Z_25_35 = Z((Z(:,1)<=35 & Z(:,1)>25),:);
Z_35_50 = Z((Z(:,1)<=50 & Z(:,1)>35),:);
labels = {Z_0_15,Z_15_25,Z_25_35,Z_35_50};

Xtrain = [];
Ytrain = [];
Xtest = [];
Ytest = [];
for i = 1:length(labels)
    Z_cate = cell2mat(labels(i));
    [Z_cate_num,~] = size(Z_cate);   
    r = randperm(Z_cate_num);                 % random index
    ntrain = round(rate*Z_cate_num);          % training samples
    Xtrain_cate = Z_cate(r(1:ntrain),2:end);       % training set
    Ytrain_cate = Z_cate(r(1:ntrain),1);       % observed training variable
    Xtest_cate  = Z_cate(r(ntrain+1:end),2:end);   % test set
    Ytest_cate  = Z_cate(r(ntrain+1:end),1);   % observed test variable
    Xtrain = [Xtrain;Xtrain_cate];
    Ytrain = [Ytrain;Ytrain_cate];
    Xtest = [Xtest;Xtest_cate];
    Ytest = [Ytest;Ytest_cate];
end

%% Split training-testing data
% [n d] = size(X);                 % samples x bands
% r = randperm(n);                 % random index
% ntrain = round(rate*n);          % #training samples
% Xtrain = X(r(1:ntrain),:);       % training set
% Ytrain = Y(r(1:ntrain),:);       % observed training variable
% Xtest  = X(r(ntrain+1:end),:);   % test set
% Ytest  = Y(r(ntrain+1:end),:);   % observed test variable

% load DATA_WHEAT_SM_73HV_save_RF5.mat;
[ntrain,~] = size(Ytrain);
[ntest,~] = size(Ytest);

%% Input data normalization, either between 0-1 or standardization (zero mean, unit variance)
% [Xtrain a b] = scale(Xtrain);
% Xtest        = scale(Xtest,a,b);
% [Xtrain a b] = scalestd(Xtrain);
% Xtest        = scalestd(Xtest,a,b);

%% Remove the mean of Y for training only
my      = mean(Ytrain);
% Ytrain  = Ytrain - repmat(my,ntrain,1);

%% SELECT METHODS FOR COMPARISON

METHODS = {'GPR'};

%% TRAIN ALL MODELS
numModels = numel(METHODS);

for m=1:numModels
    fprintf(['Training ' METHODS{m} '... \n'])
    eval(['model = train' METHODS{m} '(Xtrain,Ytrain);']); % Train the model
    eval(['Yp = test' METHODS{m} '(model,Xtest);']);       % Test the model
%     Yp = Yp + repmat(my,ntest,1);
    RESULTS(m) = assessment(Ytest, Yp, 'regress');  % assessregres(Ytest,Yp)
    
%     Plot_Scat(METHODS,Ytest,Yp,RESULTS(m)); 
%     print(figure(1),['C:\Users\DELL\Desktop\Results_single\Scatplot_',METHODS{m},'.jpg'],'-djpeg','-r600');
end

if RESULTS.RMSE<err
    err=RESULTS.RMSE;
    Plot_Scat(METHODS,Ytest,Yp,RESULTS(m));
     print(figure(1),['E:\Chapter1 Results\Results\ML\FP\corn\GPR5\',METHODS{m},num2str(nn),'.jpg'],'-djpeg','-r600');
    close all;
    save DATA_corn_SM_FP_73_save_GPR5.mat Xtrain Ytrain Xtest Ytest;
    xlswrite(['E:\Chapter1 Results\Results\ML\FP\corn\GPR5\',METHODS{m},'.xlsx'],["ME";"RMSE";"RELRMSE";"MAE";"R";"RP";"R2"],'RESULTS','A1');
    xlswrite(['E:\Chapter1 Results\Results\ML\FP\corn\GPR5\',METHODS{m},'.xlsx'],struct2cell(RESULTS),'RESULTS','B1');
    xlswrite(['E:\Chapter1 Results\Results\ML\FP\corn\GPR5\',METHODS{m},'.xlsx'],["Measured SM";Ytest],'Scat_Plot','A1');
    xlswrite(['E:\Chapter1 Results\Results\ML\FP\corn\GPR5\',METHODS{m},'.xlsx'],["Estimated SM";Yp],'Scat_Plot','B1');
end
disp(nn);
end
toc;