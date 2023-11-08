
function [Tr Te] = project_data(train_data,test_data,Smat, dim)

if nargin==3, dim = size(Smat,1); end
[U,S] = eig(Smat);    S = diag(S);    [U, S] = sortEigVecs(U,S);
z_ind = find(S<=0);   S(z_ind) = [];  U(:,z_ind) = [];  S = diag(S);
Q = (S^(0.5)) * U';   Tr = Q * train_data;   Te = Q * test_data; 
if dim < size(Tr,1),  Tr = Tr(1:dim,:);  Te = Te(1:dim,:);  end