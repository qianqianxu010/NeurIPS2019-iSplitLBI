function [ res ] = mergeCell( X )
%MERGECELL Summary of this function goes here
%   Detailed explanation goes here
res = [];
for i = 1:numel(X)
    res = [res; X{i}] ;
end

end

