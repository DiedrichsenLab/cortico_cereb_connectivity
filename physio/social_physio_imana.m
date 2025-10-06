function social_physio_imana(mode, varargin)
% social_physio_imana  Generalized GLM builder + estimator + contrasts
%
% USAGE:
%   social_physio_imana(mode, sub_list, include_task, include_retro, include_hrcrf, include_hr)
%   social_physio_imana('f-contrast', sub_list)
%   social_physio_imana('t-contrast', sub_list)
%
% MODES:
%   'GLM:spec'      - create GLM specification (design matrix)
%   'GLM:estimate'  - estimate GLM from existing SPM.mat
%   'GLM:all'       - run both spec and estimate
%   'f-contrast'    - create & evaluate RETROICOR F-contrast
%   't-contrast'    - create & evaluate HR*CRF (and HR) T-contrasts
%   'SURF:reconall' - placeholder
%   'SURF:fs2wb'    - placeholder
%
% Notes:
% - For GLM modes you must pass include_task, include_retro, include_hrcrf, include_hr (logical)
% - If sub_list is omitted, function calls get_subj() to obtain one

%% -------------- parse inputs --------------
% defaults
include_task  = false;
include_retro = false;
include_hrcrf = false;
include_hr    = false;
sub_list = {};

% extract sub_list if provided
args = varargin;
if ~isempty(args)
    if iscell(args{1})
        sub_list = args{1};
        args(1) = [];
    else
        sub_list = getSubj(workDir);
    end
end


% For GLM modes, parse include flags
if numel(args) >= 4
    include_task  = logical(args{1});
    include_retro = logical(args{2});
    include_hrcrf = logical(args{3});
    include_hr    = logical(args{4});
else
    error('For GLM modes you must pass include_task, include_retro, include_hrcrf, include_hr.');
end

%% -------------- configuration --------------
workDir     = '/cifs/diedrichsen/data';
baseDir     = fullfile(workDir,'Cerebellum/Social/data');
imaging_dir = fullfile(baseDir,'imaging_data');
physio_dir  = fullfile(baseDir,'physio','regressors');
task_desc   = fullfile(baseDir,'task_descriptions','social_task_description.tsv');
behavioral_dir = fullfile(baseDir,'behavioral');
model_name = sprintf('glm_task%d_retro%d_hrcrf%d_hr%d', include_task, include_retro, include_hrcrf, include_hr);

nRuns    = 8;
numTRs   = 590;
numDummys= 3;
TR       = 1.1;
fmri_t   = 16;
fmri_t0  = 1;

%% -------------- dispatch by mode --------------
switch lower(mode)

    %% ---------------- GLM:spec (build design) ----------------
    case 'glm:spec'
        for s=1:length(sub_list)
            sub_id = sub_list{s};
            outDir_base = fullfile(baseDir,'GLM_physio', model_name, sub_id);
            if ~exist(outDir_base,'dir'), mkdir(outDir_base); end

            % --- Build J struct across runs (one GLM per subject, multi-session)
            J = struct();
            J.dir = {outDir_base};
            J.timing.units = 'secs';
            J.timing.RT = TR;
            J.timing.fmri_t = fmri_t;
            J.timing.fmri_t0 = fmri_t0;
            
            % load task description if tasks requested
            if include_task
                C = dload(task_desc);
                Cc = getrow(C, C.condNum==1);
                nCond = max(C.condNumUni); % number of unique conditions expected
            end
            
            % prepare per-run sessions
            for r = 1:nRuns
                % scans
                scans = cell(1, numTRs - numDummys);
                for ii = 1:(numTRs - numDummys)
                    scans{ii} = fullfile(imaging_dir, sub_id, 'ses-01', sprintf('r%s_ses-01_run-%02d.nii,%d', sub_id, r, ii + numDummys));
                end
                sess.scans = scans;
                
                % conditions
                sess.cond = struct('name', {}, 'onset', {}, 'duration', {}, 'tmod', {}, 'pmod', {}, 'orth', {});
                if include_task
                    bf = fullfile(behavioral_dir, sub_id, sprintf('%s_ses-01.tsv', sub_id));
                    A = dload(bf);
                    taskfile_names = A.task_file;
                    A.condition_name = regexprep(taskfile_names, '_\d+\.tsv$', '');
                    P = getrow(A, A.run_num == r);
                    
                    % instruction
                    instruct_onset = P.real_start_time - TR * numDummys;
                    sess.cond(1).name = 'Instruct';
                    sess.cond(1).onset = instruct_onset;
                    sess.cond(1).duration = 5;
                    sess.cond(1).tmod = 0;
                    sess.cond(1).orth = 0;
                    
                    % task conditions
                    cond_index = 1;
                    for ic = 1:nCond
                        ST = find(strcmpi(P.condition_name, C.condNames{ic}));
                        if isempty(ST), continue; end
                        onset = P.real_start_time(ST) - TR * numDummys + 5;
                        end_time = P.real_end_time(ST) - TR * numDummys;
                        duration = end_time - onset;
                        cond_index = cond_index + 1;
                        sess.cond(cond_index).name = C.condNames{ic};
                        sess.cond(cond_index).onset = onset;
                        sess.cond(cond_index).duration = duration;
                        sess.cond(cond_index).tmod = 0;
                        sess.cond(cond_index).orth = 0;
                    end
                end
                
                % --- physio regressors per run (explicit with names) ---
                physio_run_dir = fullfile(physio_dir, sub_id, sprintf('run-%02d', r));
                
                sess.multi     = {''};
                sess.multi_reg = {''};
                sess.regress   = struct('name', {}, 'val', {});
                
                if include_retro
                    f_retro = fullfile(physio_run_dir, 'reg_retro.txt');
                    tmp = dlmread(f_retro);   % [N x 6]
                    retro_names = {'RETROICOR:sin1','RETROICOR:cos1','RETROICOR:sin2',...
                        'RETROICOR:cos2','RETROICOR:sin3','RETROICOR:cos3'};
                    for k = 1:size(tmp,2)
                        sess.regress(end+1).name = retro_names{k};
                        sess.regress(end).val    = tmp(:,k);
                    end
                end

                if include_hrcrf
                    f_hrcrf = fullfile(physio_run_dir, 'reg_hrcrf.txt');
                    hrcrf = dlmread(f_hrcrf);  % [N x 1]
                    sess.regress(end+1).name = 'HR*CRF';
                    sess.regress(end).val    = hrcrf;
                end
                
                if include_hr
                    f_hr = fullfile(physio_run_dir, 'reg_hr.txt');
                    hr = dlmread(f_hr);  % [N x 1]
                    sess.regress(end+1).name = 'HR';
                    sess.regress(end).val    = hr;
                end
                
                sess.hpf = inf;
                J.sess(r) = sess;
            end % runs
            
            % final J fields (preserve your settings)
            J.fact = struct('name', {}, 'levels', {});
            J.bases.hrf.derivs = [0 0];
            J.bases.hrf.params = [4.5 11];
            J.volt = 1;
            J.global = 'None';
            J.mask = {fullfile(imaging_dir, sub_id, 'ses-01', 'rmask_noskull.nii')};
            J.mthresh = 0.01;
            J.cvi_mask = {fullfile(imaging_dir, sub_id, 'ses-01', 'rmask_gray.nii')};
            J.cvi = 'fast';
            
            % save spec (SPM.mat)
            spm_rwls_run_fmri_spec(J);
            fprintf('Specification written for %s (model: %s)\n', sub_id, model_name);
        end % sub
    

    %% ---------------- GLM:estimate ----------------
    case 'glm:estimate'
        for s=1:length(sub_list)
            sub_id = sub_list{s};
            outDir_base = fullfile(baseDir,'GLM_physio', model_name, sub_id);
            if ~exist(outDir_base,'dir'), mkdir(outDir_base); end

            spmMatFile = fullfile(outDir_base, 'SPM.mat');
            if ~exist(spmMatFile,'file')
                error('SPM.mat not found in %s. Run spec first.', outDir_base);
            end
            tmp = load(spmMatFile); SPM = tmp.SPM;
            SPM.swd = outDir_base;
            SPM = spm_rwls_spm(SPM);
            save(spmMatFile, 'SPM', '-v7.3');
            fprintf('Estimation done for %s (model: %s)\n', sub_id, model_name);
        end % sub
    

    %% ---------------- GLM:all ----------------
    case 'glm:all'
        % run spec then estimate
        social_physio_imana('GLM:spec', sub_list, include_task, include_retro, include_hrcrf, include_hr);
        social_physio_imana('GLM:estimate', sub_list, include_task, include_retro, include_hrcrf, include_hr);
    
    
    %% ---------------- f-contrast ----------------
    case 'f-contrast'
        for s=1:length(sub_list)
            sub_id = sub_list{s};
            outDir_base = fullfile(baseDir,'GLM_physio', model_name, sub_id);
    
            % Load SPM.mat
            spm_path = fullfile(outDir_base, 'SPM.mat');
            load(spm_path);
    
            % Find indices of RETROICOR regressors
            retro_idx = find(contains(SPM.xX.name, 'RETROICOR:'));
    
            if isempty(retro_idx)
                warning('No RETROICOR regressors found in design matrix.');
            else
                % Build F-contrast: identity on RETROICOR regressors
                n_reg = length(SPM.xX.name);
                con = zeros(length(retro_idx), n_reg);
                for i = 1:length(retro_idx)
                    con(i, retro_idx(i)) = 1;
                end
    
                % Create contrast structure
                fcon = spm_FcUtil('Set', 'RETROICOR_all', 'F', 'c', con', SPM.xX.xKXs);
    
                % Append into SPM.xCon
                if ~isfield(SPM, 'xCon') || isempty(SPM.xCon)
                    SPM.xCon = fcon;
                else
                    SPM.xCon(end+1) = fcon;
                end
    
                % Save updated SPM.mat
                save(spm_path, 'SPM');
    
                % Estimate contrasts
                spm_contrasts(SPM);
                fprintf('F-contrast for RETROICOR regressors of %s created and estimated.\n\n', sub_id);
            end
        end % sub

    
    %% ---------------- t-contrast ----------------
    case 't-contrast'
        for s=1:length(sub_list)
            sub_id = sub_list{s};
            outDir_base = fullfile(baseDir,'GLM_physio', model_name, sub_id);

            % Load SPM.mat
            spm_path = fullfile(outDir_base, 'SPM.mat');
            load(spm_path);
    
            % List of regressors to create T-contrasts for
            reg_names = {};
            if include_hrcrf
                reg_names{end+1} = 'HR*CRF';
            end
            if include_hr
                reg_names{end+1} = 'HR';
            end
    
            for rn = 1:numel(reg_names)
                idx = find(contains(SPM.xX.name, reg_names{rn}));
    
                if isempty(idx)
                    warning('No regressor named "%s" found in design matrix.', reg_names{rn});
                    continue;
                end
    
                % Build t-contrast vector
                n_reg = length(SPM.xX.name);
                tvec = zeros(1, n_reg);
                tvec(idx) = 1;  % set 1 for all columns matching this regressor
    
                % Create contrast structure
                tcon = spm_FcUtil('Set', reg_names{rn}, 'T', 'c', tvec', SPM.xX.xKXs);
    
                % Append into SPM.xCon
                if ~isfield(SPM, 'xCon') || isempty(SPM.xCon)
                    SPM.xCon = tcon;
                else
                    SPM.xCon(end+1) = tcon;
                end
    
                fprintf('T-contrast for %s regressor created.\n', reg_names{rn});
            end
    
            % Save updated SPM.mat
            save(spm_path, 'SPM');
    
            % Estimate contrasts
            spm_contrasts(SPM);
            fprintf('T-contrasts for %s estimated.\n\n', sub_id);
        end % sub

    
    %% ---------------- SURF placeholders ----------------
    case 'surf:reconall'
        fprintf('SURF:reconall not implemented; placeholder for subject %s\n', sub_id);

    case 'surf:fs2wb'
        fprintf('SURF:fs2wb not implemented; placeholder for subject %s\n', sub_id);

    otherwise
        error('Unknown mode: %s', mode);
end

end

function subj_name = getSubj(workDir)
    pinfo = readtable(sprintf('%s/FunctionalFusion_new/Social/participants.tsv', workDir), ...
                      'FileType','text','Delimiter','\t','VariableNamingRule','preserve');
    subj_name = pinfo.participant_id(pinfo.exclude==0 & pinfo.pilot==0);
    excluded_subj = ["sub-03"; "sub-04"; "sub-10"; "sub-14"; "sub-24"; "sub-26"];
    subj_name = subj_name(~ismember(subj_name, excluded_subj));
end
