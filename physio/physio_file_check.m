clc; clear; close all;

% Define base directory and subjects
workdir = '/cifs/diedrichsen/data';
baseDir = (sprintf('%s/Cerebellum/Social',workdir));
pinfo = readtable('/cifs/diedrichsen/data/FunctionalFusion/Social/participants.tsv', ...
                  'FileType','text','Delimiter','\t','VariableNamingRule','preserve');
subj_name = pinfo.participant_id(pinfo.exclude==0 & pinfo.pilot==0);

% Open a diary file to save outputs
outFile = fullfile('/home/UWO/ashahb7/Github/cortico_cereb_connectivity/cortico_cereb_connectivity', 'physio_file_check.txt');
if exist(outFile, 'file'); delete(outFile); end % overwrite if exists
diary(outFile);
diary on;

% Runs to check
runs = 1:9;

for sn = 1:length(subj_name)

    logDir = fullfile(baseDir, 'data', 'physio', subj_name{sn}, 'ses-01');
    fprintf('\nSubject: %s\n', subj_name{sn});
    
    % Initialize lists
    runs_puls = [];
    runs_info = [];
    runs_resp = [];
    
    % Loop over runs
    for r = runs
        f_puls = fullfile(logDir, sprintf('run-%02d_PULS.log', r));
        f_info = fullfile(logDir, sprintf('run-%02d_info.log', r));
        f_resp = fullfile(logDir, sprintf('run-%02d_RESP.log', r));
        
        if exist(f_puls, 'file')
            runs_puls(end+1) = r;
        end
        if exist(f_info, 'file')
            runs_info(end+1) = r;
        end
        if exist(f_resp, 'file')
            runs_resp(end+1) = r;
        end
    end
    
    % Print results
    fprintf('  PULS logs found for runs: %s\n', mat2str(runs_puls));
    fprintf('  INFO logs found for runs: %s\n', mat2str(runs_info));
    fprintf('  RESP logs found for runs: %s\n', mat2str(runs_resp));
    
end

% Close diary
diary off;