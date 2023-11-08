% Example of kernel matrices calculation
% train_data --> DxN data matrix (N training samples)
% train_lbls --> Nx1 vector of training labels
% test_data --> DxNt test data matrix
% Cval, Rval, Lval --> regularization parameters

% parameters 
graph_type = 1;  % type of graph
kNN = 5;  % number of NNs for kNN graph

% example of usage
[D,N] = size(train_data);  [D,M] = size(test_data);
noOfClasses = length(unique(train_lbls));

% calculate kernel matrices
[Ktrain, Ktest] = kernel_rbf(train_data, test_data, -1.0, train_data);

% calculate GE kernel
[Ktrain_ge,Ktest_ge] = calc_ge_kernel(Ktrain,Ktest,train_lbls,graph_type,Lval,Rval,kNN);
