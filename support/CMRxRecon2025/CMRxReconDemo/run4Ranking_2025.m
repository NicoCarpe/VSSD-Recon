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
% note: inference does rss so images are real and coil combined
% note: input shape differs slightly so permuatations are made
fprintf('>> run4Ranking_2025: file=”%s”, pre‑permute size = %s\n', ...
            filetype, mat2str(size(img)));

isBlackBlood = 0;
if contains(filetype,'blackblood') || contains(filetype,'T1w') || contains(filetype,'T2w')
    % non‑temporal: [sz,sy,sx] → [sx,sy,sz]
    img = permute(img, [3, 2, 1]);    
    [sx, sy, sz] = size(img);         
    t = 1;
else
    % temporal:     [t,sz,sy,sx] → [sx,sy,sz,t]
    img = permute(img, [4, 3, 2, 1]);  
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
% note: in our case image is already coil combined
sosImg = img;

% if single slice, ensure 4D shape
% if sz == 1
%     sosImg = reshape(sosImg,[size(sosImg,1),size(sosImg,2),1,size(sosImg,3)]);
% end

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

return
