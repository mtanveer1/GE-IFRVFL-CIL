%function [train_data_ge,test_data_ge] = calc_ge_data(train_data,test_data,train_lbls,graph_type,Lval,Rval,kNN)
function [S] = calc_ge_data2(train_data,train_lbls,graph_type,kNN)
% train_data --> DxN matrix (N training samples)
% test_data --> DxM matrix (M test samples)
% train_lbls --> Nx1 training label vector
% graph_type --> graph type (1->LDA, 2->LFDA, 3->MDA)
% Lval, Rval --> regularization parameetrs 
% kNN --> NN parameter for MDA graph

% center data
[D,N] = size(train_data);  %[D,M] = size(test_data);
%% Our Data is already normalized
% noOfClasses = length(unique(train_lbls));
% mean_vec = mean(train_data,2);
% train_data = train_data - mean_vec*ones(1,N);
%test_data = test_data - mean_vec*ones(1,M);

% calculate graph matrices
if graph_type==1  % LDA
    [S_w, S_b] = matrices_LDA(train_data, train_lbls);
elseif graph_type==2  % LFDA
    [S_w, S_b] = matrices_LFDA(train_data, train_lbls);
elseif graph_type==3  % MDA
    [S_w, S_b] = matrices_MDA(train_data, train_lbls, kNN);
end

% calculate modified samples
Rval=10^-3;
S = pinv(S_b + Rval*eye(size(S_b)))*S_w;
%[train_data_ge, test_data_ge] = project_data(train_data,test_data, (Lval*S + eye(size(S))) );
