function [Ktrain, Ktest] = kernel_rbf(M_train, M_test, A, M_ref)

N = size(M_train,2);  NN = size(M_test,2);
if nargin < 3,  A = -1.0;  end  % finetune A for getting a better kernel, if needed
 
if nargin < 4
    Dtrain = ((sum(M_train'.^2,2)*ones(1,N))+(sum(M_train'.^2,2)*ones(1,N))'-(2*(M_train'*M_train)));
    Dtest = ((sum(M_train'.^2,2)*ones(1,NN))+(sum(M_test'.^2,2)*ones(1,N))'-(2*(M_train'*M_test)));
    if A<0.0,  A = mean(mean(Dtrain)) * 2;  end
else
    MM = size(M_ref,2);
    Dtrain = ((sum(M_ref'.^2,2)*ones(1,N))+(sum(M_train'.^2,2)*ones(1,MM))'-(2*(M_ref'*M_train)));
    Dtest = ((sum(M_ref'.^2,2)*ones(1,NN))+(sum(M_test'.^2,2)*ones(1,MM))'-(2*(M_ref'*M_test)));
    if A<0.0,  A = mean(mean(Dtrain)) * 2;  end
end

Ktrain = exp(-Dtrain/A);
Ktest = exp(-Dtest/A);
