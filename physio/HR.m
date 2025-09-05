clc; clear; close all;

workdir = '/cifs/diedrichsen/data';
baseDir = fullfile(workdir, 'Cerebellum/Social');

% Parameters
sub_s   = "sub-07";
smooth_flag = true;  % set to false to keep raw HR
smooth_window = 1;   % seconds for moving average
dt = 0.25;            % resampling resolution in sec
win = [-5 15];     % peri-event window
t_common = win(1):0.5:win(2);

all_segments = [];   % event-related HR across all runs
all_onsets   = [];

for r = 1:8
    run_s = sprintf('run-%02d', r);

    logDir = fullfile(baseDir, sprintf('data/physio/regressors/%s/%s', sub_s, run_s));
    behDir = fullfile(baseDir, sprintf('data/behavioral/%s/%s_ses-01.tsv', sub_s, sub_s));

    % --- Load physio from TAPAS output ---
    load(fullfile(logDir, sprintf('physio_%s.mat', run_s)), 'physio');

    % --- Compute instantaneous HR ---
    pulse_times = physio.ons_secs.cpulse;
    IBI = diff(pulse_times);
    HR_inst = 60 ./ IBI;
    t_inst  = pulse_times(2:end);

    % --- Interpolate HR onto continuous grid ---
    t_hr = t_inst(1):dt:t_inst(end);
    hr   = interp1(t_inst, HR_inst, t_hr, 'linear', 'extrap');

    if smooth_flag
        % Moving average smoothing over window
        N = round(smooth_window / dt);  % number of points in window
        hr = smoothdata(hr, 'movmean', N);
    end

    % --- Load behavior and get onsets for this run ---
    tsv_table = readtable(behDir, ...
        'FileType','text', ...
        'Delimiter','\t', ...
        'VariableNamingRule','preserve');

    % select only rows for this run
    onsets = tsv_table.real_start_time(tsv_table.run_num == r);

    % --- Extract peri-event segments ---
    for i = 1:length(onsets)
        t0 = onsets(i);
        t_rel = t_hr - t0;
        mask  = (t_rel >= win(1)) & (t_rel <= win(2));
        if any(mask)
            seg = interp1(t_rel(mask), hr(mask), t_common, 'linear', NaN);
            all_segments(end+1,:) = seg;
            all_onsets(end+1,1)   = t0;
        end
    end
end

%% Plotting
mean_hr = nanmean(all_segments,1);

figure; hold on;
plot(t_common, all_segments', 'Color',[0.8 0.8 0.8]); % single trials
plot(t_common, mean_hr, 'k', 'LineWidth',2);          % mean
xline(0,'r--','Task onset');
xlabel('Time relative to onset (s)');
ylabel('HR (bpm)');
title(sprintf('Event-related HR (N = %d)', size(all_segments,1)));
grid on;

%% --- Function to read raw log ---
function tbl = readTable()
    fid = fopen('run-01_PULS.log','r');
    lines = textscan(fid, '%s', 'Delimiter', '\n');
    lines = lines{1};
    fclose(fid);

    % Find start of data
    data_start = find(~cellfun(@isempty, regexp(lines, '^\s*\d+')), 1);
    data_lines = lines(data_start:end);

    n = numel(data_lines);
    ACQ_TIME_TICS = zeros(n,1);
    CHANNEL = strings(n,1);
    VALUE = zeros(n,1);
    SIGNAL = strings(n,1);

    for i = 1:n
        tokens = strsplit(strtrim(data_lines{i}));
        ACQ_TIME_TICS(i) = str2double(tokens{1});
        CHANNEL(i) = tokens{2};
        VALUE(i) = str2double(tokens{3});
        if numel(tokens) >= 4
            SIGNAL(i) = tokens{4};
        else
            SIGNAL(i) = "";
        end
    end

    tbl = table(ACQ_TIME_TICS, CHANNEL, VALUE, SIGNAL);
end

function subj_name = getSubj(workDir, excluded_subj)
    pinfo = readtable(sprintf('%s/FunctionalFusion/Social/participants.tsv', workDir), ...
                      'FileType','text','Delimiter','\t','VariableNamingRule','preserve');
    subj_name = pinfo.participant_id(pinfo.exclude==0 & pinfo.pilot==0);
    subj_name = subj_name(~ismember(subj_name, excluded_subj));
end