function [test_accuracy,indx] = GE_IFRVFL_CIL_Predict(X,Y,model)

beta = model.beta;
W = model.W;
b = model.b;

Nsample = size(X,1);

X1 = X*W+repmat(b,Nsample,1);

X1 = relu(X1);

X1=[X1,ones(Nsample,1)];
X = [X,X1];
rawScore = X*beta;

rawScore_temp1 = bsxfun(@minus,rawScore,max(rawScore,[],2));
num = exp(rawScore_temp1);
dem = sum(num,2);
prob_scores = bsxfun(@rdivide,num,dem);
[max_prob,indx] = max(prob_scores,[],2);
[~, ind_corrClass] = max(Y,[],2);
test_accuracy = mean(indx == ind_corrClass);

end

