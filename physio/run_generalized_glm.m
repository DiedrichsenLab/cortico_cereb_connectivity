function run_generalized_glm(mode, sub_id, include_task, include_retro, include_hr)
% run_generalized_glm  Generalized GLM builder + estimator (concatenate runs)
%
% USAGE:
%   run_generalized_glm(mode, sub_id, include_task, include_retro, include_hr)
%
% INPUTS:
%   mode          - 'spec'        : Create GLM specification (design matrix)
%                   'est'         : Estimate GLM from existing SPM.mat
%                   'all'         : Run both 'spec' and 'est'
%                   'f-contrast'  : Build F-contrast for RETROICOR regressors
%                   't-contrast'  : Build T-contrast for a single regressor (e.g., 'HR*CRF')
%   sub_id        - Subject ID string (e.g., 'sub-05')
%   include_task  - (logical) Include task regressors (18 per run + instruct)
%   include_retro - (logical) Include RETROICOR regressors (6 per run)
%   include_hr    - (logical) Include heart-rate regressors (1-2 per run)
%
% NOTES:
% - Assumes 8 runs per subject, TR=1.1, numTRs=590, numDummys=3
% - Expects physio files per run:
%       baseDir/data/physio/regressors/<sub_id>/run-XX/reg_retro.txt  (586x6 after discarding dummies)
%       baseDir/data/physio/regressors/<sub_id>/run-XX/reg_hr.txt    (586x1 or 586x2)
% - Expects behavioural task description (task_descriptions folder).

%% --- config ---
baseDir     = '/cifs/diedrichsen/data/Cerebellum/Social/data';
imaging_dir = fullfile(baseDir,'imaging_data');
physio_dir  = fullfile(baseDir,'physio','regressors');
task_desc   = fullfile(baseDir,'task_descriptions','social_task_description.tsv');
behavioral_dir = fullfile(baseDir,'behavioral');

nRuns    = 8;
numTRs   = 590;
numDummys= 3;
TR       = 1.1;
fmri_t   = 16;
fmri_t0  = 1;

model_name = sprintf('glm_task%d_retro%d_hr%d', include_task, include_retro, include_hr);
outDir_base = fullfile(baseDir,'GLM_physio', model_name, sub_id);
if ~exist(outDir_base,'dir'), mkdir(outDir_base); end

%% sanitize inputs
if isstring(sub_id), sub_id = char(sub_id); end

%% Build J struct across runs (one GLM per subject, multi-session)
if any(strcmpi(mode,{'spec','all'}))
    % initialize J
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
        
        % conditions: if include_task, build per-run conds using behavioral files
        sess.cond = struct('name', {}, 'onset', {}, 'duration', {}, 'tmod', {}, 'pmod', {}, 'orth', {});
        if include_task
            % load behavioral for subject (expects <sub_id>_ses-01.tsv in behavioral_dir)
            bf = fullfile(behavioral_dir, sub_id, sprintf('%s_ses-01.tsv', sub_id));
            A = dload(bf);
            % Add conditions
            taskfile_names = A.task_file;                       
            A.condition_name = regexprep(taskfile_names, '_\d+\.tsv$', ''); % Remove the last '_<number>.tsv' from 
            P = getrow(A, A.run_num == r);
            
            % instruction regressor (one per run)
            instruct_onset = P.real_start_time - TR * numDummys; % first start time adjusted
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
                onset = P.real_start_time(ST) - TR * numDummys + 5; % announceTime=5
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
            tmp = dlmread(f_retro);   % [N x 6] (sin1 cos1 sin2 cos2 sin3 cos3)
            retro_names = {'RETROICOR:sin1','RETROICOR:cos1','RETROICOR:sin2',...
                'RETROICOR:cos2','RETROICOR:sin3','RETROICOR:cos3'};
            for k = 1:size(tmp,2)
                sess.regress(end+1).name = retro_names{k};
                sess.regress(end).val    = tmp(:,k);
            end
        end
        
        if include_hr
            f_hr = fullfile(physio_run_dir, 'reg_hrcrf.txt');
            tmp2 = dlmread(f_hr);  % [N x 1] or [N x 2]
            if size(tmp2,2) == 2
                hr_names = {'HR*CRF','HR'};
            else
                hr_names = {'HR*CRF'};
            end
            for k = 1:size(tmp2,2)
                sess.regress(end+1).name = hr_names{k};
                sess.regress(end).val    = tmp2(:,k);
            end
        end
        
        sess.hpf = inf;
        J.sess(r) = sess;
    end % runs
    
    % final J fields
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
end

%% Estimate if requested
if any(strcmpi(mode, {'est','all'}))
    % Expect SPM.mat exists in outDir_base
    spmMatFile = fullfile(outDir_base, 'SPM.mat');
    if ~exist(spmMatFile,'file')
        error('SPM.mat not found in %s. Run spec first.', outDir_base);
    end
    tmp = load(spmMatFile); SPM = tmp.SPM;
    SPM.swd = outDir_base;
    SPM = spm_rwls_spm(SPM);
    save(spmMatFile, 'SPM', '-v7.3');
    fprintf('Estimation done for %s (model: %s)\n', sub_id, model_name);
end

%% Calculate f-contrast of RETROICOR regressors
if strcmp(mode, 'f-contrast')
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
end

%% Calculate t-contrast of HR*CRF
if strcmp(mode, 't-contrast')
    % Load SPM.mat
    spm_path = fullfile(outDir_base, 'SPM.mat');
    load(spm_path);

    % List of regressors to create T-contrasts for
    reg_names = {'HR*CRF', 'HR'};

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
end

end