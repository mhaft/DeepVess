function [  ] = writetif(im, fileName)
%WRITETIF write a 3D matrix into a multi-page tif file without compression
%   WRITETIF(IM) ask for the tif file name and write the 3D matrix IM at
%       that address. IM can be 2-D or 3-D intensity images with any data type.
%   WRITETIF(IM, FILENAME) write the 3D matrix IM into a file at FILENAME. 
%       FILENAME can be relative or absolute file address.
%
% Example
% ---------
% This example wirte a 3D uint8 image into a file located at one level up 
%   folder.
%
% im = uint8(rand(256, 256, 10)*255)
% writetif(im, '../rand.tif')

% Copyright 2015-2018, Mohammad Haft-Javaherian. (mh973@cornell.edu)

% If filename is not in argin
if nargin<2
    path=uigetdir('select the folder for savinh tif file');
    fileName=inputdlg('Enter the file name');
    fileName=fileName{1};
    % if file name does not contain '.tif'
    if length(fileName)<4 || ~strcmpi(fileName(end-3:end),'.tif')
        fileName=strcat(fileName,'.tif');
    end
    fileName=[path, '/', fileName];
end

% write the first page of tif file. If the fileName exsit, will overwrite 
imwrite(im(:,:,1),fileName,'Compression','none')

% write the rest of tif pages
for i=2:size(im,3)
    imwrite(im(:,:,i),fileName,'WriteMode', 'append','Compression','none')
end

end

