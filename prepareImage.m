function prepareImage(zStart, isFolder, VesselCh, totalCh, inPath, ...
                      inFile, saturated_prctile, isMotion)
% extract the vessel channel of a stack, normalize it and save it as 8 bit 
% image, then remove the motion artifact and save the result as h5 file. 
%
%   PREPAREIMAGE() run the function with the default parameters and user 
%       interface to locate the input file. 
%   PREPAREIMAGE(zStart, isFolder, VesselCh, totalCh) run the function with the input parameters as
%       described in bellow and user interface to locate the input file.
%   PREPAREIMAGE(zStart, isFolder, VesselCh, totalCh, inPath, inFile) run the function with the input parameters as
%       described in bellow and the file located at [inPath, '/', inFile]
%
% Parameters
%     zStart - the z start of stack right after dura. Put 1 if no cut off is needed.
%     isFolder - Put true if all the tif files in a folder need to be prepared
%     VesselCh , totalCh - for cases we have more than one channels.
%       Otherwise both should be 1. e.g. VesselCh=2 , totalCh=4
%     inPath - input path to folder of file
%     saturated_prctile - image normalization saturation prctile [1 98]    
%     isMotion - if there is motion and should apply motion removel
%
% Example
% ---------
% user interface will ask for a single tif filethat has four channel and 
%   the first channel is the vessel channel and start the image from 
%   slice 10. h5 file with similar name will be writen to dame folder. 
%
% prepareImage(10, 0, 1, 4); 

% Copyright 2017-2018, Mohammad Haft-Javaherian. (mh973@cornell.edu)

%   References:
%   -----------
%   [1] Haft-Javaherian, M; Fang, L.; Muse, V.; Schaffer, C.B.; Nishimura, 
%       N.; & Sabuncu, M. R. (2018) Deep convolutional neural networks for 
%       segmenting 3D in vivo multiphoton images of vasculature in 
%       Alzheimer disease mouse models. *arXiv preprint, arXiv*:1801.00880.

% Default input arguments
if nargin < 1
    zStart = 1;
    isFolder = false; 
    VesselCh = 1;   
    totalCh = 1;  
end
if nargin < 7
    saturated_prctile = [1 98];
end
if nargin < 8
    isMotion = false;
end

% extract the file addresses
if isFolder
    if nargin == 5
        PathName = inPath;
    else
        PathName = uigetdir('*.*', 'Select the folder of raw image (*.tif)');
    end
    f = dir(fullfile(PathName, '/*.tif'));
else
    if nargin > 4
        f(1).name = inFile;
        f(1).folder = inPath;
    else
        [f(1).name, f(1).folder] = uigetfile('*.*', 'Select raw image (*.tif)');
    end
end

for i=1:numel(f)
    inFile = fullfile(f(i).folder, f(i).name);
    h5FileName = fullfile(f(i).folder, [f(i).name(1:end-3), 'h5']);
    % read multipage tif file
    im = readtif(inFile);
    % extract just vessel slices
    im = im(:, :, VesselCh:totalCh:end); 
    % remove the top layer of the image
    im = im(:, :, zStart:end);
    im = imNormalize(im, saturated_prctile);
    % remove the motion artifact 
    if isMotion
        outFile = [f(i).folder, 'noMotion-', f(i).name];
    	im = tifMotionRemoval(outFile);
    end
    % shift im to [-0.5,0.5]
    im = single(im);
    im=im / max(im(:)) - 0.5;
    % write h5 file
    if exist(h5FileName,'file')
        delete(h5FileName)
    end
    h5create(h5FileName, '/im', size(im), 'Datatype', 'single')
    h5write(h5FileName, '/im', im)
end

end
