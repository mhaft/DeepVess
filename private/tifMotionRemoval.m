function [  ] = tifMotionRemoval(inFile)
%Remove motion artifact using displacement method for each slice based on
%   the previous slice and then save the result with same file name and
%   'noMotion-' prefix.
%   TIFMOTIONREMOVAL() ask for the tif file address. remove the motion and
%   save the resutl. the progress will be displayed.
%   TIFMOTIONREMOVAL(INFILE) remove the motion from INFILE and save the
%   result
%
% Example
% ---------
% This example fix the motion artifact of an image.
%
% im = readtif('cell.tif');
%
% size(im) 
% ans =
%    256   256    10
%
% tifMotionRemoval(im);
% 1-2-3-4-5-6-7-8-9-10-
% 
% dir
%    cell.tif
%    noMotion-cell.tif

% Copyright 2015-2018, Mohammad Haft-Javaherian. (mh973@cornell.edu)

% If inFile is not in argin
if nargin<1    
    [FileName,PathName] = uigetfile('*.*','Image with motion (*.tif)');
    inFile=[PathName,FileName];
    outFile=[PathName,'noMotion-',FileName];
end
% get the inFile info from the tif file header
info=imfinfo(inFile);
nr=info(1,1).Height;
nc=info(1,1).Width;
np=length(info);
% initiate the im
im=zeros(nr,nc,np);
if info(1).BitDepth==8
    im=uint8(im);
elseif info(1).BitDepth==16
    im=uint16(im);
end

% read first slice
im(:,:,1)=imread(inFile,1);
imwrite(im(:,:,1),outFile,'Compression','none')
fprintf('%d-',1)

% read other slices and remove the motion based on the previous slice
for k=2:np
    im(:,:,k)=imread(inFile,k);
    [~,im(:,:,k)]= imregdemons(im(:,:,k),im(:,:,k-1), ...
        [500 400 200],'AccumulatedFieldSmoothing', 1.3, 'DisplayWaitbar', false);
    imwrite(im(:,:,k),outFile,'WriteMode','append','Compression','none')
    fprintf('%d-',k)
end
fprintf('\n')
