%Programmer: Chris Tralie
%Purpose: A wrapper around Uli Bauer's ripser
%Parameters: D (N x N distance matrix)
%           maxHomDim: The maximum homological dimension (default 1)
%           thresh: The maximum length at which to add an edge
%           coeff: The field coefficients to use (default 2)
function [PDs] = RipsFiltrationDM(D, maxHomDim, thresh, coeff)
    if nargin < 3
        thresh = max(D(:))*2;
    end
    if nargin < 4
        coeff = 2;
    end
    %Step 1: Extract and output lower triangular distance matrix
    DLT = D(triu(~~D));
    dlmwrite('DLower.txt', DLT, ' ');
    [~, cmdout] = system(sprintf('ripser/ripser-coeff --dim %i --threshold %g --modulus %i DLower.txt', maxHomDim, thresh, coeff));
    lines = strsplit(cmdout, '\n');
    PDs = {};
    for ii = 1:length(lines)
        if length(lines{ii}) < 4
            continue
        end
        if strcmp(lines{ii}(1:4), 'dist')
            continue
        elseif strcmp(lines{ii}(1:4), 'valu')
            continue
        elseif strcmp(lines{ii}(1:4), 'pers')
            PDs{end+1} = {};
        else
            s = strrep(lines{ii}, ' ', '');
            s = strrep(s, '[', '');
            s = strrep(s, ']', '');
            s = strrep(s, '(', '');
            s = strrep(s, ')', '');
            fields = strsplit(s, ',');
            PDs{end}{end+1} = str2double(fields);
        end
    end
    for ii = 1:length(PDs)
        PDs{ii} = cell2mat(PDs{ii}');
    end
end