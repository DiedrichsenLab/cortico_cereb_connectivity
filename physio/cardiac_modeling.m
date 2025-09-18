%% Starting
clc; clear; close all;

% Define the data basedirectory 
[workDir, baseDir] = setDirs();
outDir = fullfile(baseDir, 'data', 'physio', 'regressors');

numTRs = 590;
numDummys = 3;
TR = 1.1;

% Get subject list/cell
excluded_subj = ["sub-03"; "sub-04"; "sub-10"; "sub-14"; "sub-24"; "sub-26"];
subj_name = getSubj(workDir, excluded_subj);

addpath(genpath('/cifs/diedrichsen/matlab'));

%% Cardiac Modeling
for sn = 1:length(subj_name)
    logDir = fullfile(baseDir, 'data/physio', subj_name{sn}, 'ses-01/');
    runnum = 1:8;

    for nrun = runnum
        cd(logDir);

        PULS = dir(sprintf('run-%02d_puls.log', nrun));  % Your PPU log
        RSP  = dir(sprintf('run-%02d_resp.log', nrun));  % Your respiratory log
        log  = dir(sprintf('run-%02d_info.log', nrun));  % Scan timing log

        % Initialize TAPAS physio model
        physio = tapas_physio_new();
        physio.verbose.level = 0;

        % Output files
        outSubjDir = fullfile(outDir, subj_name{sn}, sprintf('run-%02d', nrun));
        if ~exist(outSubjDir, 'dir')
            mkdir(outSubjDir);
        end
        physio.save_dir = {outSubjDir};

        % Input files
        physio.log_files.cardiac = {PULS.name};
        physio.log_files.respiration = {RSP.name};
        physio.log_files.scan_timing = {log.name};
        physio.log_files.vendor = 'Siemens_Tics';
        physio.log_files.relative_start_acquisition = 0;
        physio.log_files.align_scan = 'last';

        % Scan timing
        physio.scan_timing.sqpar.Nslices = 56;
        physio.scan_timing.sqpar.Ndummies = numDummys;
        physio.scan_timing.sqpar.Nscans = numTRs-numDummys;
        physio.scan_timing.sqpar.Nechoes = 1;
        physio.scan_timing.sqpar.onset_slice = 28;
        physio.scan_timing.sync.method = 'scan_timing_log';
        physio.scan_timing.sqpar.TR = TR;

        % Cardiac pre-processing
        physio.preproc.cardiac.modality = 'PPU';
        physio.preproc.cardiac.initial_cpulse_select.max_heart_rate_bpm = 110;
        physio.preproc.cardiac.initial_cpulse_select.auto_matched.min = 0.6;
        physio.preproc.cardiac.initial_cpulse_select.auto_matched.file = 'initial_cpulse_kRpeakfile.mat';
        physio.preproc.cardiac.posthoc_cpulse_select.off = struct([]);

        % TAPAS options
        physio.model.output_multiple_regressors = sprintf('physio_regressors_run-%02d.txt', nrun);
        physio.model.output_physio = sprintf('physio_run-%02d.mat', nrun);
        physio.model.order.c = 6;  % number of cardiac RETROICOR components
        physio.model.order.r = 0;  % number of respiratory components
        physio.model.order.cr = 0; % interaction terms off

        % --- HR regressor ---
        physio.model.hrv.include = true;             % include HR variability model
        physio.model.hrv.delays = 0;                 % HR at time of scan
        physio.model.hrv.orthogonalise = 'none';     % don’t orthogonalise
        physio.model.hrv.logtrans = false;           % don’t log-transform
        physio.model.hrv.relativeRVT = false;        % standard HRV, not relative

        % Run TAPAS PhysIO main script
        [physio_out, R, ons_sec] = tapas_physio_main_create_regressors(physio);
        
        % --- Save regressors ---
        % 1. RETROICOR (first 6 columns)
        dlmwrite(fullfile(outSubjDir,'reg_retro.txt'), R(:,1:6), 'delimiter','\t', 'precision','%.8f');
        
        % 2. Heart rate (column 7 + un-convolved HR if available)
        if exist('ons_sec','var') && isfield(ons_sec,'hr')
            dlmwrite(fullfile(outSubjDir,'reg_hr.txt'), [R(:,7) ons_sec.hr], 'delimiter','\t', 'precision','%.8f');
        else
            dlmwrite(fullfile(outSubjDir,'reg_hr.txt'), R(:,7), 'delimiter','\t', 'precision','%.8f');
        end
        fprintf('Cardiac regressors created for subject %s, run %d\n', subj_name{sn}, nrun);

    end
end

%% Functions
function [workDir, baseDir] = setDirs()
    if isfolder('/Volumes/diedrichsen_data$/data')
        workDir='/Volumes/diedrichsen_data$/data';
    elseif isfolder('/srv/diedrichsen/data')
        workDir='/srv/diedrichsen/data';
    elseif isfolder('/cifs/diedrichsen/data')
        workDir='/cifs/diedrichsen/data';
    else
        fprintf('Workdir not found. Mount or connect to server and try again.');
    end
    baseDir = sprintf('%s/Cerebellum/Social', workDir);
end

function subj_name = getSubj(workDir, excluded_subj)
    pinfo = readtable(sprintf('%s/FunctionalFusion/Social/participants.tsv', workDir), ...
                      'FileType','text','Delimiter','\t','VariableNamingRule','preserve');
    subj_name = pinfo.participant_id(pinfo.exclude==0 & pinfo.pilot==0);
    subj_name = subj_name(~ismember(subj_name, excluded_subj));
end
