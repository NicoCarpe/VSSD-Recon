%% This is a demo to generate validation results into the submission folder
% MICCAI "CMRxRecon" challenge 2025
% 2023.03.06 @ fudan university
% Email: wangcy@fudan.edu.cn
% Revise: Huang Mingkai
% Updated directory structure: Task -> Center -> Vendor -> Patient -> .mat
clc; clear;

%% add path to utilities
addpath('./utils');

% set your data directories
basePath     = '/home/nicocarp/scratch/PromptUMamba/predict/cmr25-cardiac/test_promptmamba_R2/reconstructions/';    % superior directory of 'MultiCoil/'
mainSavePath = '/home/nicocarp/scratch/PromptUMamba/predict/cmr25-cardiac/test_promptmamba_R2/Submission/';     % output path
taskType     = 'TaskR2';                      % options: 'TaskR1', 'TaskR2','TaskS1','TaskS2'

%% fixed settings
dataTypeList = {'Cine','BlackBlood','T1w','T2w','Mapping','Flow2d','Perfusion','LGE','T1rho'};
setName      = 'ValidationSet/';          
coilInfo     = 'MultiCoil/';

%% traverse data types
for iType = 1:numel(dataTypeList)
    dataType = dataTypeList{iType};
    taskDir  = fullfile(basePath, coilInfo, dataType, setName, ['UnderSample_',taskType]);
    centerList = dir(taskDir);

    % loop over center folders
    for iCenter = 1:numel(centerList)
        centerInfo = centerList(iCenter);
        if ~centerInfo.isdir || startsWith(centerInfo.name,'.')
            continue;
        end
        centerPath = fullfile(taskDir, centerInfo.name);

        % loop over vendor folders
        vendorList = dir(centerPath);
        for iVendor = 1:numel(vendorList)
            vendorInfo = vendorList(iVendor);
            if ~vendorInfo.isdir || startsWith(vendorInfo.name,'.')
                continue;
            end
            vendorPath = fullfile(centerPath, vendorInfo.name);

            % loop over patient folders
            patientList = dir(vendorPath);
            for iPatient = 1:numel(patientList)
                patientInfo = patientList(iPatient);
                if ~patientInfo.isdir || startsWith(patientInfo.name,'.')
                    continue;
                end
                patientPath = fullfile(vendorPath, patientInfo.name);

                % find all .mat files in patient folder
                matFiles = dir(fullfile(patientPath,'*.mat'));
                for iFile = 1:numel(matFiles)
                    fileInfo = matFiles(iFile);
                    fullMatPath = fullfile(patientPath, fileInfo.name);

                    % load k-space data (binary MAT or generic HDF5)
                    try
                        tmp    = load(fullMatPath);            % try native MAT load
                        fld    = fieldnames(tmp);
                        kspace = tmp.(fld{1});
                    catch
                        % fallback to HDF5 reader
                        try
                            info    = h5info(fullMatPath);
                            dsNames = {info.Datasets.Name};
                            % if no root-level datasets, recurse into groups
                            if isempty(dsNames)
                                dsNames = collectDatasets(info);
                            end
                            if isempty(dsNames)
                                error('No HDF5 datasets found in %s', fullMatPath);
                            end
                            % read the first dataset
                            dsPath = ['/' dsNames{1}];
                            kspace = h5read(fullMatPath, dsPath);
                        catch ME2
                            warning('  [WARN] cannot read %s: %s', fullMatPath, ME2.message);
                            continue;   % skip this file
                        end
                    end

                    % image reconstruction and ranking
		    % note: our outputs are in image space not kspace so no ifft2c
                    img         = kspace; % ifft2c(kspace);
                    img4ranking = run4Ranking_2025(img, fileInfo.name);

                    % prepare save directory mirroring input structure
                    saveDir = fullfile(mainSavePath, taskType, coilInfo, dataType, setName, ['UnderSample_',taskType], centerInfo.name, vendorInfo.name, patientInfo.name);
                    if ~exist(saveDir,'dir')
                        createRecursiveDir(saveDir);
                    end

                    % save result
                    savePath = fullfile(saveDir, fileInfo.name);
                    save(savePath, 'img4ranking');
                end
                fprintf('Processed %s/%s/%s\n', dataType, centerInfo.name, patientInfo.name);
            end
        end
    end
end

disp('All data generation successful!');

%% Local helper for recursive dataset discovery
function names = collectDatasets(grp)
    names = {};
    % add all Datasets in this group
    for i = 1:numel(grp.Datasets)
        names{end+1} = grp.Datasets(i).Name; %#ok<AGROW>
    end
    % recurse into sub-groups
    for j = 1:numel(grp.Groups)
        names = [names collectDatasets(grp.Groups(j))]; %#ok<AGROW>
    end
end
