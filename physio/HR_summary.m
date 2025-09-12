% This script loads the behavioral data of subjects and then load the 
% Heart-Rate to calculate avg HR for instruction and task period and saves 
% it in a .tsv file

clc; clear; close all;

TR = 1.1; % seconds

% Setting directories
workDir = '/cifs/diedrichsen/data';
baseDir = fullfile(workDir, 'Cerebellum/Social');
outFile = fullfile(baseDir, 'data/physio/HR_summary.tsv');

% Get subject list/cell
excluded_subj = ["sub-03"; "sub-04"; "sub-10"; "sub-14"; "sub-24"; "sub-26"];
subj_name = getSubj(workDir, excluded_subj);
num_subjects = length(subj_name);

% Initialize results table
results = table('Size', [0 6], ...
                'VariableTypes', {'string','double','string','string','double','double'}, ...
                'VariableNames', {'subj_id','run_num','task_name','task_code','avg_instruction_HR','avg_task_HR'});

for sn = 1:num_subjects
    sub_s = subj_name{sn};
    fprintf('Processing subject %s (%d of %d)\n', sub_s, sn, num_subjects);

    % Load behavioral onsets
    behDir = fullfile(baseDir, sprintf('data/behavioral/%s/%s_ses-01.tsv', sub_s, sub_s));
    behData = readtable(behDir, 'FileType','text','Delimiter','\t','VariableNamingRule','preserve');
    
    % Load HR data
    physio_all = cell(1, 8);
    for r = 1:8
        run_s = sprintf('run-%02d', r);
        physDir = fullfile(baseDir, sprintf('data/physio/regressors/%s/%s', sub_s, run_s));
        physio_all{r} = load(fullfile(physDir, sprintf('physio_%s.mat', run_s)), 'physio');
    end

    % Loop through each row in behData (each row = one task occurrence)
    for i = 1:height(behData)
    
        subj_id   = sub_s;
        run_num       = behData.run_num(i);
        task_name = string(behData.task_name(i));
        task_code = string(behData.task_code(i));
        inst_dur  = behData.instruction_dur(i);
        start_t   = behData.real_start_time(i);
        end_t     = behData.real_end_time(i);
    
        % Get HR vector for this subject/run
        hr = physio_all{run_num}.physio.ons_secs.hr;  % 590x1 HR vector
    
        % Convert times → indices
        idx_instr = round(start_t/TR) + 1 : round((start_t+inst_dur)/TR) + 1;
        idx_task  = round((start_t+inst_dur)/TR) + 1 : round(end_t/TR) + 1;
    
        % Safety check: clip indices inside HR vector length
        idx_instr(idx_instr < 1 | idx_instr > length(hr)) = [];
        idx_task(idx_task < 1 | idx_task > length(hr)) = [];
    
        % Compute averages
        avg_inst_HR = mean(hr(idx_instr), 'omitnan');
        avg_task_HR = mean(hr(idx_task), 'omitnan');
    
        % Append row
        newRow = {subj_id, run_num, task_name, task_code, avg_inst_HR, avg_task_HR};
        results = [results; newRow];
    end
end

% Save to .tsv
writetable(results, outFile, 'FileType', 'text', 'Delimiter', '\t');

%% Functions

function subj_name = getSubj(workDir, excluded_subj)
    pinfo = readtable(sprintf('%s/FunctionalFusion/Social/participants.tsv', workDir), ...
                      'FileType','text','Delimiter','\t','VariableNamingRule','preserve');
    subj_name = pinfo.participant_id(pinfo.exclude==0 & pinfo.pilot==0);
    subj_name = subj_name(~ismember(subj_name, excluded_subj));
end
