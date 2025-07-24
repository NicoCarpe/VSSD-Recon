function runRecon_2025(basePath, mainSavePath, coilInfo, setName, dataType, taskType, sampleStatusType, reconType, imgShow)
%RECON2025 Batch reconstruction for CMRxRecon 2025 with updated directory structure
%   Directory: basePath/coilInfo/dataType/setName/[FullSample|UnderSample_taskType]/Center/Vendor/Patient/*.mat
% inputs:
%   basePath         - root of 'MultiCoil'
%   mainSavePath     - output root for recon results
%   coilInfo         - e.g. 'MultiCoil'
%   setName          - 'TrainingSet','ValidationSet','TestSet'
%   dataType         - one of {'Cine','BlackBlood','T1w','T2w','Mapping','Perfusion','LGE','T1rho'}
%   taskType         - 'taskR1' or 'taskR2'
%   sampleStatusType - 0: full k-space, 1: undersampled
%   reconType        - 0: zero-fill, 1: GRAPPA, 2: SENSE, 3: both
%   imgShow          - 0:no display, 1:show images

% determine mask suffix
if strcmpi(taskType,'TaskR1')
    maskDirSuffix = 'TaskR1';
elseif strcmpi(taskType,'TaskR2')
    maskDirSuffix = 'TaskR2';
elseif strcmpi(taskType,'TaskS1')
    maskDirSuffix = 'TaskS1';
elseif strcmpi(taskType,'TaskS2')
    maskDirSuffix = 'TaskS2';
else
    error('Wrong taskType! Use ''taskR1'' or ''taskR2''.');
end

% set k-space directory
if contains(setName,'Train')
    kspaceDir = fullfile(basePath,coilInfo,dataType,setName,'FullSample');
    taskName=['FullSample_',taskType];
else
    kspaceDir = fullfile(basePath,taskType,coilInfo,dataType,setName,['UnderSample_' taskType]);
    taskName=['UnderSample_',taskType];
end
maskBase = ['Mask_' maskDirSuffix];

% traverse Center folders
centerList = dir(kspaceDir);
for iC = 1:numel(centerList)
    cInfo = centerList(iC);
    if ~cInfo.isdir || startsWith(cInfo.name,'.'), continue; end
    centerPath = fullfile(kspaceDir, cInfo.name);
    maskCenterPath = strrep(centerPath,'FullSample',maskBase);

    % traverse Vendor folders
    vendorList = dir(centerPath);
    for iV = 1:numel(vendorList)
        vInfo = vendorList(iV);
        if ~vInfo.isdir || startsWith(vInfo.name,'.'), continue; end
        vendorPath = fullfile(centerPath, vInfo.name);

        % traverse Patient folders
        patientList = dir(vendorPath);
        for iP = 1:numel(patientList)
            pInfo = patientList(iP);
            if ~pInfo.isdir || startsWith(pInfo.name,'.'), continue; end
            patientPath = fullfile(vendorPath, pInfo.name);

            % find .mat files
            matFiles = dir(fullfile(patientPath,'*.mat'));
            for iF = 1:numel(matFiles)
                fInfo = matFiles(iF);
                fullMat = fullfile(patientPath, fInfo.name);

                % load k-space
                ks = load(fullMat);
                fld = fieldnames(ks);
                kspace = ks.(fld{1});

                % determine modalities
                isBlackBlood = contains(fInfo.name,'blackblood') || contains(fInfo.name,'T1w') || contains(fInfo.name,'T2w');
                isMapping    = contains(fInfo.name,'T1map') || contains(fInfo.name,'T2map') || contains(fInfo.name,'T2smap') || contains(fInfo.name,'T1mappost');
                isT1rho      = contains(fInfo.name,'T1rho');

                % get dimensions
                if isBlackBlood
                    [sx,sy,scc,sz] = size(kspace); t=1;
                else
                    [sx,sy,scc,sz,t] = size(kspace);
                end

                % select central slices
                if sz < 3
                    sliceToUse = 1:sz;
                else
                    c = round(sz/2);
                    sliceToUse = c-1:c;
                end

                % select time frames
                if isBlackBlood || t == 1
                    timeToUse = 1;
                elseif isMapping || isT1rho
                    timeToUse = 1:t;
                else
                    timeToUse = 1:min(3,t);
                end

                % reconstruction branches
                if sampleStatusType == 1
                    % TrainingSet with multiple masks
                    if contains(setName,'Train')
                        maskFiles = dir(fullfile(maskCenterPath,'*.mat'));
                        for iM = 1:numel(maskFiles)
                            mInfo = maskFiles(iM);
                            if ~contains(mInfo.name, replace(fInfo.name,'.mat','')), continue; end
                            maskFull = fullfile(maskCenterPath, mInfo.name);
                            isRadial = contains(mInfo.name,'Radial');
                            tmp = load(maskFull,'mask'); mask = tmp.mask;
                            % apply mask
                            masked = zeros(sx,sy,scc,sz,t);
                            if ndims(mask) > 2
                                for fr = 1:size(mask,3)
                                    masked(:,:,:,:,fr) = kspace(:,:,:,:,fr).*mask(:,:,fr);
                                end
                            else
                                masked = kspace .* mask;
                            end
                            selectedK = masked(:,:,:,sliceToUse,:);
                            if numel(timeToUse)>1, selectedK = selectedK(:,:,:,:,timeToUse); end

                            reconImg = ChallengeRecon(selectedK, sampleStatusType, reconType, imgShow, isRadial);
                        end
                        
                    else
                        % Validation/Test undersampled
                        isRadial = contains(fInfo.name,'Radial');
                        selectedK = kspace(:,:,:,sliceToUse,:);
                        if numel(timeToUse)>1, selectedK = selectedK(:,:,:,:,timeToUse); end
                        reconImg = ChallengeRecon(selectedK, sampleStatusType, reconType, imgShow, isRadial);
                    end
                else
                    % full-sample
                    selectedK = kspace(:,:,:,sliceToUse,:);
                    if numel(timeToUse)>1, selectedK = selectedK(:,:,:,:,timeToUse); end
                    reconImg = ChallengeRecon(selectedK, sampleStatusType, reconType, imgShow);
                end

                % crop for ranking
                if numel(timeToUse)>1
                    img4ranking = single(crop(abs(reconImg),[round(sx/3),round(sy/2),numel(sliceToUse),numel(timeToUse)]));
                elseif numel(sliceToUse)>1
                    img4ranking = single(crop(abs(reconImg),[round(sx/3),round(sy/2),numel(sliceToUse)]));
                else
                    img4ranking = single(crop(abs(reconImg),[round(sx/3),round(sy/2)]));
                end

                % save output
                outDir = fullfile(mainSavePath,taskType,coilInfo,dataType,setName,taskName,cInfo.name,vInfo.name,pInfo.name);
                if ~exist(outDir,'dir'), createRecursiveDir(outDir); end
                save(fullfile(outDir,fInfo.name),'img4ranking');
                fprintf('Reconstructed %s/%s/%s/%s\n', dataType, cInfo.name, vInfo.name, pInfo.name);
            end
        end
    end
end

disp('All recon tasks completed!');
end
