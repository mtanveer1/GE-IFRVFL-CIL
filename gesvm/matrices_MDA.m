

function [Smda_w, Smda_b, Wmda_w, Wmda_b, Lmda_w, Lmda_b] = matrices_MDA(data, lbls, kNN)
% and Locality Sensitive Discriminant Analysis, D. Cai et al. 2007

N = size(data,2);
Dmat = ((sum(data'.^2,2)*ones(1,N))+(sum(data'.^2,2)*ones(1,N))'-(2*(data'*data)));
Wmda_b=zeros(size(Dmat));   Wmda_w=zeros(size(Dmat));
for ii=1:length(lbls)
    
    curr_ind = find(lbls==lbls(ii)); 
    [vals ind] = sort(Dmat(ii,curr_ind),'ascend');
    try
    knn_ind = ind(1:kNN);  
    catch
        kNN=length(ind);
        knn_ind = ind(1:kNN); 
    end
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

Smda_w = data * Lmda_w * data';
Smda_b = data * Lmda_b * data';