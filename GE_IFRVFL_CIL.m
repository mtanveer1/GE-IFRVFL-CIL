function [GE_IFRVFL_CIL_Model,TrainAcc,TestAcc]  = GE_IFRVFL_CIL(trainX,trainY,testX,testY,option)



[GE_IFRVFL_CIL_Model,TrainAcc] = GE_IFRVFL_CIL_Train(trainX,trainY,option);


[TestAcc,Predict_Y] = GE_IFRVFL_CIL_Predict(testX,testY,GE_IFRVFL_CIL_Model);
Predict_Y(Predict_Y~=1)=-1;
GE_IFRVFL_CIL_Model.Predict_Y=Predict_Y;

end
