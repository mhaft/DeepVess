function [ imOut ] = imNormalize( im )
% IMNORMALIZE Normalize an image by puting 1% at each end of spectrum
% 
% Copyright 2015-2018, Mohammad Haft-Javaherian. (mh973@cornell.edu)

im2 = double(im) .* double(~imdilate(im==0, strel('sphere', 10)));
im2=double(im(and(im2>0,im2<max(im(:)))));
p1=prctile(im2,1);
p99=prctile(im2,99);
if p1==p99
    p1=min(double(im(:)));
    p99=max(double(im(:)));
    if p1==p99
        return
    end
end
imOut=(double(im)-p1+1)/(p99-p1+2);
imOut(:)=min(1,max(0,imOut(:)));
end

