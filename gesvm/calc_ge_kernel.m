
function [Ktrain_ge,Ktest_ge] = calc_ge_kernel(Ktrain,Ktest,train_lbls,graph_type,Lval,Rval,kNN)
% Ktrain --> NxN kernel matrix (N training samples)
% Ktest --> NxM kernel matrix (M test samples)
% train_lbls --> Nx1 training label vector
% graph_type --> graph type (1->LDA, 2->LFDA, 3->MDA)
% Lval, Rval --> regularization parameetrs 
% kNN --> NN parameter for MDA graph

% center kernel matrix
Ktrain = (eye(N,N)-ones(N,N)/N) * Ktrain * (eye(N,N)-ones(N,N)/N);
Ktest = (eye(N,N)-ones(N,N)/N) * (Ktest - (Ktrain*ones(N,1)/N)*ones(1,M));

% calculate graph matrices
if graph_type==1  % LDA
    [S_w, S_b, L_w, L_b] = matrices_LDA(train_data, train_lbls);
    clear S_w;  clear S_b;
elseif graph_type==2  % LFDA
    [S_w, S_b, W_w, W_b, L_w, L_b] = matrices_LFDA_kernel(Ktrain, train_lbls);
    clear S_w;  clear S_b;  clear W_w;  clear W_b;
elseif graph_type==3  % MDA
    [S_w, S_b, W_w, W_b, L_w, L_b] = matrices_MDA_kernel(Ktrain, train_lbls, kNN);
    clear S_w;  clear S_b;  clear W_w;  clear W_b;
end
pL_b = pinv(L_b);


% calculate modified kernel matrices
tmp_val1 = (Lval+Rval) / Rval;    tmp_val2 = Lval / (Rval*Rval);
Q = ( tmp_val1*eye(size(L_w)) - tmp_val2 * Ktrain * pinv(pL_b + Ktrain/Rval) * Ktrain * L_w );
pQlda = pinv(Q);    Ktrain_ge = pQ' * Q * Ktrain * pQ;    Ktest_ge = pQ' * Ktest;
if isreal(Ktrain_ge) ~=1,  Ktrain_ge = abs(Ktrain_ge);   end
if isreal(Ktest_ge) ~=1,   Ktest_ge = abs(Ktest_ge);     end
