clc;clear all;close all; split_ratio=0.7; nFolds=5; addpath(genpath('C:\Users\2023_TNNLS_GE_IFWRVFL_CIL\GE_IFWRVFL_CIL'))
temp_data=load('abalone9-18.txt');
trainX=temp_data(:,1:end-1); mean_X = mean(trainX,1); std_X = std(trainX);
trainX = bsxfun(@rdivide,trainX-repmat(mean_X,size(trainX,1),1),std_X);
All_Data=[trainX,temp_data(:,end)];

[samples,~]=size(All_Data);
rng('default')
test_start=floor(split_ratio*samples);
training_Data = All_Data(1:test_start-1,:); testing_Data = All_Data(test_start:end,:);
Opt_para.method='Algo1';
Opt_para.N=23; Opt_para.graph_type=1;
Opt_para.C=10^4; Opt_para.kerfPara.type='rbf';
Opt_para.kerfPara.pars=0.125; Opt_para.lambda=10^-3; Opt_para.scale=1;

[GE_IFRVFL_CIL_Model,TrainAcc,TestAcc]  = GE_IFRVFL_CIL(training_Data(:,1:end-1),training_Data(:,end),testing_Data(:,1:end-1),testing_Data(:,end),Opt_para);

%---------------Testing---------------
xtest0=testing_Data(:,1:end-1);   ytest0=testing_Data(:,end);
no_test=size(xtest0,1);
classifier=GE_IFRVFL_CIL_Model.Predict_Y;
obs1=ytest0;
match = 0.;
match1=0;
posval=0;
negval=0;
for i = 1:no_test
    if(obs1(i)==1)
        if(classifier(i) == obs1(i))
            match = match+1;
        end
        posval=posval+1;
    elseif(obs1(i)==-1)
        if(classifier(i) ~= obs1(i))
            match1 = match1+1;
        end
        negval=negval+1;
    end
end
if(posval~=0)
    a_pos=(match/posval);
else
    a_pos=0;
end

if(negval~=0)
    am_neg=(match1/negval);
else
    am_neg=0;
end

AUC=(1+a_pos-am_neg)/2;

AUC=AUC*10
