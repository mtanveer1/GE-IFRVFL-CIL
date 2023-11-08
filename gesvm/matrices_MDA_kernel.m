

function [Smda_w, Smda_b, Wmda_w, Wmda_b, Lmda_w, Lmda_b] = matrices_MDA_kernel(K, lbls, kNN)
% and Locality Sensitive Discriminant Analysis, D. Cai et al. 2007

N = size(K,2);
Dmat = zeros(size(K));
for ii=1:N
    for jj=1:N
        Dmat(ii,jj) = sqrt(K(ii,ii) + K(jj,jj) - 2*K(ii,jj));
    end
end
Wmda_b=zeros(size(Dmat));   Wmda_w=zeros(size(Dmat));
for ii=1:length(lbls)
    
    curr_ind = find(lbls==lbls(ii));
    [vals ind] = sort(Dmat(ii,curr_ind),'ascend');
    knn_ind = ind(1:kNN);  
    Wmda_w(ii,curr_ind(knn_ind)) = 1;
    
    curr_ind = find(lbls~=lbls(ii));
    [vals ind] = sort(Dmat(ii,curr_ind),'ascend');
    knn_ind = ind(1:kNN);  
    Wmda_b(ii,curr_ind(knn_ind)) = 1;
end

Wmda_w = Wmda_w + Wmda_w';	Wmda_w(Wmda_w~=0) = 1.0; 
Wmda_b = Wmda_b + Wmda_b';	Wmda_b(Wmda_b~=0) = 1.0;

Dmda_w = sum(Wmda_w')';  Lmda_w = diag(Dmda_w) - Wmda_w;
Dmda_b = sum(Wmda_b')';  Lmda_b = diag(Dmda_b) - Wmda_b;

Smda_w = [];  Smda_b = [];