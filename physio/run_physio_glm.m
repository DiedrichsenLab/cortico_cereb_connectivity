function run_physio_glm(action, sub_id, runnum)
% run_physio_glm  Run PhysIO-only GLM stages per run
%
% Usage:
%   run_physio_glm('glm_spec',   'sub-05', 1)
%   run_physio_glm('glm_est',    'sub-05', 1)
%   run_physio_glm('g_contrast', 'sub-05', 1)
%   run_physio_glm('all',        'sub-05', 1)

%% --- configuration ---
baseDir     = '/cifs/diedrichsen/data/Cerebellum/Social';
imaging_dir = fullfile(baseDir,'data','imaging_data');
physio_dir  = fullfile(baseDir,'data','physio','regressors');

numTRs    = 590;
numDummys = 3;
TR        = 1.1;
fmri_t    = 16;
fmri_t0   = 1;

glmSubjDir = fullfile(baseDir,'data','GLM_physio', sub_id, sprintf('run-%02d', runnum));
if ~exist(glmSubjDir,'dir'), mkdir(glmSubjDir); end
spmMatFile = fullfile(glmSubjDir,'SPM.mat');
funcDir    = fullfile(imaging_dir, sub_id, 'ses-01');
physioFile = fullfile(physio_dir, sub_id, sprintf('run-%02d', runnum), ...
                      'reg_retro.txt');

%% --- build specification struct ---
function J = build_spec()
    J.dir            = {glmSubjDir};
    J.timing.units   = 'secs';
    J.timing.RT      = TR;
    J.timing.fmri_t  = fmri_t;
    J.timing.fmri_t0 = fmri_t0;

    scans = cell(1, numTRs - numDummys);
    for ii = 1:(numTRs - numDummys)
        scans{ii} = fullfile(funcDir, sprintf('r%s_ses-01_run-%02d.nii,%d', sub_id, runnum, ii+numDummys));
    end

    sess.scans     = scans;
    sess.cond      = struct('name', {}, 'onset', {}, 'duration', {}, 'tmod', {}, 'pmod', {}, 'orth', {});
    sess.multi     = {''};
    sess.regress   = struct('name', {}, 'val', {});
    sess.multi_reg = {physioFile};
    sess.hpf       = inf;

    J.sess             = sess;
    J.fact             = struct('name', {}, 'levels', {});
    J.bases.hrf.derivs = [0 0];
    J.volt             = 1;
    J.global           = 'None';
    J.mask             = {fullfile(funcDir,'rmask_noskull.nii')};
    J.mthresh          = 0.01;
    J.cvi_mask         = {fullfile(funcDir,'rmask_gray.nii')};
    J.cvi              = 'fast';
end

%% Actions
switch lower(action)
    case 'glm_spec'
        J = build_spec();
        spm_rwls_run_fmri_spec(J);
        fprintf('Specification saved for %s run-%02d\n', sub_id, runnum);

    case 'glm_est'
        tmp = load(spmMatFile);
        SPM = tmp.SPM;
        SPM.swd = glmSubjDir;
        SPM = spm_rwls_spm(SPM);
        save(spmMatFile,'SPM');
        fprintf('Estimation finished for %s run-%02d\n', sub_id, runnum);

    case 'f_contrast'
        tmp = load(spmMatFile);
        SPM = tmp.SPM;
        SPM.swd = glmSubjDir;

        % Manual F-contrast for first 6 regressors
        C = eye(6);
        xCon = struct();
        xCon.name    = 'RETROICOR_all';
        xCon.STAT    = 'F';
        xCon.c       = C;
        xCon.iX0     = [];
        xCon.X0      = [];
        xCon.X1o     = [];
        xCon.sessrep = 'none';

        if isfield(SPM,'xCon') && ~isempty(SPM.xCon)
            SPM.xCon(end+1) = xCon;
            con_idx = numel(SPM.xCon);
        else
            SPM.xCon = xCon;
            con_idx = 1;
        end

        SPM = spm_contrasts(SPM, con_idx);
        save(spmMatFile,'SPM');
        fprintf('F-contrast evaluated for %s run-%02d\n', sub_id, runnum);

    case 'all'
        % Simply call each stage sequentially
        run_physio_glm('GLM_spec', sub_id, runnum);
        run_physio_glm('GLM_est', sub_id, runnum);
        % run_physio_glm('F_contrast', sub_id, runnum);

    otherwise
        error('Unknown action: %s', action);
end
end