function [ imOut ] = imNormalize( im, saturated_prctile)
% IMNORMALIZE Normalize an image by puting 1% at each end of spectrum
% 
% Copyright 2015-2018, Mohammad Haft-Javaherian. (mh973@cornell.edu)

if nargin < 2
    saturated_prctile = [1, 99]; 
end
im = double(im);
im2 = nonzeros(im .* (~imdilate(im==0, strel('square', 5))));
p = prctile(im2, [saturated_prctile(1), saturated_prctile(2)]);
if p(1) == p(2)
    p(1) = min(im(:));
    p(2) = max(im(:));
    if p(1) == p(2)
        imOut = nan;
        return
    end
end
imOut = (im - p(1) + 1)/(p(2) - p(1) + 2);
imOut(:) = min(1, max(0, imOut(:)));
end

