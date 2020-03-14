function [ im ] = readtif( fileName )
%READTIF read a gray multi-page tif file into a 3D matrix 
%   READTIF(FILENAME) read the file at FILENAME.
%
% Example
% ---------
% This example read a 3D uint8 image into a file located at one level up 
%   folder.
%
% im = readtif('../rand.tif');
%
% size(im) 
% ans =
%    256   256    10
% 
% class(im)
% ans =
% uint8

% Copyright 2015-2018, Mohammad Haft-Javaherian. (mh973@cornell.edu)

% If filename is not in argin
if nargin<1
    [fileName, path] = uigetfile('*.tif*', 'select the tif file');
    fileName = [path, '/', fileName];
end
info=imfinfo(fileName);
% initiate the im
im = zeros(info(1).Height, info(1).Width, info(1).SamplesPerPixel, ...
    size(info,1));

for k = 1:size(info, 1)
   im(:, :, :, k) = imread(fileName, k); 
end

% convert the hyperstack to stack
im = reshape(im, info(1).Height, info(1).Width, []);

% change the class of im base on the Bit Depth in tif header
if info(1).BitDepth==16
    im=uint16(im);
else
    im=uint8(im);
end

end

