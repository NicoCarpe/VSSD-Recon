function img4ranking = run4Ranking_2025(img,filetype)
% to reduce the computing burden and space, we only evaluate the central 2 slices
% For cine: use the first 3 time frames for ranking!
% For mapping: we need all weighting for ranking!
% crop the middle 1/6 of the original image for ranking
%
% this function helps you to convert your data for ranking
% img: complex images reconstructed with dimensions (sx,sy,sz,t/w)
% filetype: mat file name
% img4ranking: "single" format images with dims (sx/3,sy/2,2,3) for ranking

% check if it is BlackBlood/T1w/T2w (single-frame modalities)

% -----------------------------------------------------------------------------
% NOTE ON PYTHON↔MATLAB HDF5 DIMENSION ORDERING
%
% • Python/NumPy uses row-major (C-order): shape = (T, Z, H, W)
% • MATLAB uses column-major (Fortran-order): when you load an HDF5 dataset
%   written by Python, MATLAB will reverse the axes in memory, so
%     Python’s (T, Z, H, W) → MATLAB sees size(img) = [W, H, Z, T]
% -----------------------------------------------------------------------------

fprintf('>> run4Ranking_2025: file=”%s”, pre-permute size = %s\n', ...
            filetype, mat2str(size(img)));

isBlackBlood = 0;
if contains(filetype,'blackblood') || contains(filetype,'T1w') || contains(filetype,'T2w')
    [sx, sy, sz] = size(img);         
    t = 1;    
    isBlackBlood = 1;
else
    [sx, sy, sz, t] = size(img);
end

% detect mapping modalities
detectMap = {'T1map','T2map','T2smap','T1mappost'};
isMapping = any(cellfun(@(x) contains(filetype,x), detectMap));

% detect T1rho modality (independent but same handling as mapping)
isT1rho = contains(filetype,'T1rho');

% clipping slices
if sz < 3
    sliceToUse = 1:sz;
else
    center = round(sz/2);
    sliceToUse = (center-1):(center);
end

% clipping time frames
if isBlackBlood || t == 1
    timeFrameToUse = 1;
elseif isMapping || isT1rho
    timeFrameToUse = 1:t;
else
    timeFrameToUse = 1:min(3,t);
end

% coil-combine via sum-of-squares
% sosImg = squeeze(sos(img,3));
%
% % if single slice, ensure 4D shape
% if sz == 1
%     sosImg = reshape(sosImg,[size(sosImg,1),size(sosImg,2),1,size(sosImg,3)]);
% end

% note: inference does rss so images are real and coil combined already
sosImg = img;

% select and crop
if isBlackBlood
    selectedImg = sosImg(:,:,sliceToUse);
    if length(sliceToUse)>1
        img4ranking = single(crop(abs(selectedImg),[round(sx/3),round(sy/2),length(sliceToUse)]));
    else
        img4ranking = single(crop(abs(selectedImg),[round(sx/3),round(sy/2)]));
    end
else
    selectedImg = sosImg(:,:,sliceToUse,timeFrameToUse);
    if length(timeFrameToUse)>1 && length(sliceToUse)>1
        img4ranking = single(crop(abs(selectedImg),[round(sx/3),round(sy/2),length(sliceToUse),length(timeFrameToUse)]));
    elseif length(timeFrameToUse)>1 && length(sliceToUse)==1
        img4ranking = single(crop(abs(selectedImg),[round(sx/3),round(sy/2),1,length(timeFrameToUse)]));
    elseif length(timeFrameToUse)==1 && length(sliceToUse)>1
        img4ranking = single(crop(abs(selectedImg),[round(sx/3),round(sy/2),length(sliceToUse)]));
    end
end

fprintf('>> run4Ranking_2025: file=”%s”, ranking image size = %s\n', ...
            filetype, mat2str(size(img4ranking)));


% -------------------------------------
% QA: write out the first ranking image as a PNG
qaDir = '/home/nicocarp/scratch/VSSD-Recon/predict/cmr25-cardiac/test_VSSD-Recon_R1/Submission_pngs';
if ~exist(qaDir,'dir')
    mkdir(qaDir);
end

% img4ranking is either:
%   3-D: [sx/3, sy/2, #slices]           (static case)
%   4-D: [sx/3, sy/2, #slices, #frames]  (cine/mapping)
if ndims(img4ranking)==3
    pngImg = img4ranking(:,:,1);
else
    pngImg = img4ranking(:,:,1,1);
end

% normalize to [0,1] and save
pngImg = mat2gray(abs(pngImg));
[~, base] = fileparts(filetype);
outName = fullfile(qaDir, base + "_rank1.png");
imwrite(pngImg, outName);
fprintf("  → QA PNG saved: %s\n", outName);
% -------------------------------------


return
