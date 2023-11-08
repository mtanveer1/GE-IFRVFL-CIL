

function [Slfda_w, Slfda_b, Wlfda_w, Wlfda_b, Llfda_w, Llfda_b] = matrices_LFDA(data, lbls)

N = size(data,2); 
Dmat = ((sum(data'.^2,2)*ones(1,N))+(sum(data'.^2,2)*ones(1,N))'-(2*(data'*data)));
sigma2 = mean(mean(Dmat));    Amat = exp(-Dmat/(2*sigma2));


Wlfda_b=zeros(size(Amat));   Wlfda_w=zeros(size(Amat));
for ii=1:length(lbls)
    curr_ind = find(lbls==lbls(ii));  other_ind = find(lbls~=lbls(ii));
    Wlfda_w(ii,curr_ind) = Amat(ii,curr_ind) / length(curr_ind);
    Wlfda_b(ii,curr_ind) = Amat(ii,curr_ind) * (1/N - 1/length(curr_ind));
    Wlfda_b(ii,other_ind) = 1/N;
end

Wlfda_w = (Wlfda_w + Wlfda_w')/2;
Wlfda_b = (Wlfda_b + Wlfda_b')/2;

Dlfda_w = sum(Wlfda_w')';  Llfda_w = diag(Dlfda_w) - Wlfda_w;
Dlfda_b = sum(Wlfda_b')';  Llfda_b = diag(Dlfda_b) - Wlfda_b;

Slfda_w = data * Llfda_w * data';
Slfda_b = data * Llfda_b * data';