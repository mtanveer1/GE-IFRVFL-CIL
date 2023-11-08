function [RVFLModel,TrainAcc,TestAcc]  = IFW_RVFL_CIL(trainX,trainY,testX,testY,option)



[RVFLModel,TrainAcc] = IFW_RVFL_train_CIL(trainX,trainY,option);


[TestAcc,Predict_Y] = RVFL_predict(testX,testY,RVFLModel);
Predict_Y(Predict_Y~=1)=-1;
RVFLModel.Predict_Y=Predict_Y;

end
