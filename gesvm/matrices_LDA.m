

function [Slda_w, Slda_b, Llda_w, Llda_b] = matrices_LDA(data, lbls)
 
noOfClasses = length(unique(lbls));  N = size(data,2);
W = zeros(N);
for k=1:noOfClasses
    curr_ind = find(lbls==k);     n_k = length(curr_ind);
    e_k = zeros(size(data,2),1);  e_k(curr_ind) = 1.0;
    try
    W = W + (1/n_k)*(e_k*e_k');
    catch
        disp('ERRORs')
    end
end

Llda_w = eye(size(W)) - W;
Llda_b = W - (1/N)*ones(N);

Slda_w = data * Llda_w * data';
Slda_b = data * Llda_b * data';

