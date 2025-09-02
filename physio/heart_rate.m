%% Starting
clc; clear; close all;

%% Globals
global baseDir subj_name

% Define the data basedirectory 
if isdir('/Volumes/diedrichsen_data$/data')
    workdir='/Volumes/diedrichsen_data$/data';
elseif isdir('/srv/diedrichsen/data')
    workdir='/srv/diedrichsen/data';
elseif isdir('/cifs/diedrichsen/data')
    workdir='/cifs/diedrichsen/data';
else
    fprintf('Workdir not found. Mount or connect to server and try again.');
end
baseDir=(sprintf('%s/Cerebellum/Social',workdir));

pinfo = readtable('/cifs/diedrichsen/data/FunctionalFusion/Social/participants.tsv', ...
                  'FileType','text','Delimiter','\t','VariableNamingRule','preserve');
subj_name = pinfo.participant_id(pinfo.exclude==0 & pinfo.pilot==0);

% addpath(genpath('/cifs/diedrichsen/matlab/imaging/tapas-master/PhysIO/code'));
addpath(genpath('/cifs/diedrichsen/matlab'));

%% Call Function
% bsp_imana('PHYS:createRegressorPPU', 'sn', 1, 'runnum', 1:9)

%% Functions
for sn = 1:length(subj_name)
    logDir = fullfile(baseDir, 'data/physio', subj_name{sn}, 'ses-01/');

    % Get all files that match run-XX_PULS.log
    files = dir(fullfile(logDir, 'run-*_puls.log'));

    % Extract run numbers using regexp
    runnum = [];
    for i = 1:length(files)
        % match the number between "run-" and "_PULS.log"
        tok = regexp(files(i).name, 'run-(\d+)_puls\.log', 'tokens');
        if ~isempty(tok)
            runnum(end+1) = str2double(tok{1}{1});
        end
    end
    
    % Sort runs just in case they’re not ordered
    runnum = sort(runnum);

    for nrun = runnum
        nrun
        cd(logDir);

        PULS = dir(sprintf('run-%02d_PULS.log', nrun));  % Your PPU log
        log  = dir(sprintf('run-%02d_info.log', nrun));  % Scan timing log

        % Initialize TAPAS physio model
        physio = tapas_physio_new();

        % Input files
        physio.save_dir = {logDir};
        physio.log_files.cardiac = {PULS.name};
        physio.log_files.scan_timing = {log.name};
        physio.log_files.vendor = 'Siemens_Tics';
        physio.log_files.relative_start_acquisition = 0;
        physio.log_files.align_scan = 'last';

        % Scan timing
        physio.scan_timing.sqpar.Nslices = 56;
        physio.scan_timing.sqpar.Nscans = 590;
        physio.scan_timing.sqpar.Nechoes = 1;
        physio.scan_timing.sqpar.onset_slice = 28;
        physio.scan_timing.sync.method = 'scan_timing_log';

        % Cardiac pre-processing
        physio.preproc.cardiac.modality = 'PPU';
        physio.preproc.cardiac.initial_cpulse_select.max_heart_rate_bpm = 110;
        physio.preproc.cardiac.initial_cpulse_select.auto_matched.min = 0.4;
        physio.preproc.cardiac.initial_cpulse_select.auto_matched.file = 'initial_cpulse_kRpeakfile.mat';
        physio.preproc.cardiac.posthoc_cpulse_select.off = struct([]);

        % TAPAS options: no respiration
        physio.model.output_multiple_regressors = sprintf('physio_regressors_run-%02d.txt', nrun);
        physio.model.output_physio = sprintf('physio_run-%02d.mat', nrun);
        physio.model.order.c = 6;  % number of cardiac RETROICOR components
        physio.model.order.r = 0;  % 0 because no respiration
        physio.model.order.cr = 0; % interaction terms off

        % Run TAPAS physio
        tapas_physio_main_create_regressors(physio);
        fprintf('Cardiac regressors created for subject %d, run %d\n', sn, nrun);

    end
end