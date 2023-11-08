% Example of calculating the transformed data
% train_data --> DxN data matrix (N training samples)
% train_lbls --> Nx1 vector of training labels
% test_data --> DxNt test data matrix
% Cval, Rval, Lval --> regularization parameters

% parameters
kNN = 5;  % number of NNs for kNN graph 
graph_type = 1;  % graph type

% example of usage
[D,N] = size(train_data);  [D,M] = size(test_data);
noOfClasses = length(unique(train_lbls));

% calculate GE data
[train_data_ge,test_data_ge] = calc_ge_data(train_data,test_data,train_lbls,graph_type,Lval,Rval,kNN);
