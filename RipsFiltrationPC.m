function [ PDs ] = RipsFiltrationPC( X, maxHomDim, thresh, coeff )
    if nargin < 4
        coeff = 2;
    end
    D = pdist2(X, X);
    if nargin < 3
        thresh = max(D(:))*2;
    end
    PDs = RipsFiltrationDM(D, maxHomDim, thresh, coeff);
end

