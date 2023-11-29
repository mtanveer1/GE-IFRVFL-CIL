function [model,train_accuracy] = GE_IFRVFL_CIL_Train(trainX,trainY,option)

N = option.N;
C = option.C;
graph_type=option.graph_type;
s = option.scale;

option.mu=option.kerfPara.pars;
tic
[S1,S2]=IF_non_linear_score_values([trainX,trainY],option.mu);
time1=toc;
A1=trainX(trainY==1,:);
B1=trainX(trainY~=1,:);

m1=size(A1,1);
m2=size(B1,1);

if strcmp(option.method,'Class_Spec')

    D1=m2/(m1+m2);
    D2=m1/(m1+m2);
else

    D1=1;
    D2=m1/m2;
end

trainX=[A1;B1];
trainY=[ones(size(A1,1),1);-1*ones(size(B1,1),1)];

[Nsample,Nfea] = size(trainX);
trainY(trainY==-1)=2;
U_trainY = unique(trainY);
option.trainY=U_trainY ;
nclass = numel(U_trainY);
rng('default')
tic
W = (rand(Nfea,N)*2*s-1);
b = s*rand(1,N);
X1 = trainX*W+repmat(b,Nsample,1);

X1 = relu(X1);


X = [trainX,X1,ones(Nsample,1)];

lp=C*D1; ln=C*D2;

S=[(1+lp/ln)*S1;(1+ln/lp)*S2];

U_trainY=unique(trainY);
nclass=numel(U_trainY);
trainY_temp=zeros(numel(trainY),nclass);

for i=1:nclass
    idx= trainY==U_trainY(i);

    trainY_temp(idx,i)=1;
end
lambda=option.lambda;
SW = calc_ge_data2(X',trainY,graph_type,5);
X=diag(S)*X;
beta = ( (1/lp+1/ln)*(eye(size(SW))+lambda*SW )+X'*X) \ X'*trainY_temp;
train_time=toc;
train_time=train_time+time1;
model.beta = beta;
model.W = W;
model.b = b;

trainY_temp = X*beta;

trainY_temp1 = bsxfun(@minus,trainY_temp,max(trainY_temp,[],2)); %for numerical stability
num = exp(trainY_temp1);
dem = sum(num,2);
prob_scores = bsxfun(@rdivide,num,dem);
[max_prob,indx] = max(prob_scores,[],2);
train_accuracy = mean(indx == trainY);
model.train_time=train_time;

st = dbstack;
model.function_name= st.name;
end
