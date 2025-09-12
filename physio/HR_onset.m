clc; clear; close all;

plot_run = false;
plot_subj = false;
plot_all = true;

workDir = '/cifs/diedrichsen/data';
baseDir = fullfile(workDir, 'Cerebellum/Social');

% Get subject list/cell
excluded_subj = ["sub-03"; "sub-04"; "sub-10"; "sub-14"; "sub-24"; "sub-26"];
subj_name = getSubj(workDir, excluded_subj);
% subj_name = {"sub-05"};
% subj_name = subj_name(1:4);

% Parameters
dt = 0.5;             % resampling resolution in sec for continuous HR
win = [-5 20];        % peri-event window (seconds)
t_common = win(1):dt:win(2);  % common time grid for averaging

max_jump = 6;
win_size = 3;

all_segments = [];

for sn = 1:length(subj_name)
    sub_s = subj_name{sn};
    behDir = fullfile(baseDir, sprintf('data/behavioral/%s/%s_ses-01.tsv', sub_s, sub_s));
    % --- Load behavioral onsets ---
    tsv_table = readtable(behDir, 'FileType','text','Delimiter','\t','VariableNamingRule','preserve');
    subj_segments = [];

    for r = 1:8
        run_s = sprintf('run-%02d', r);
    
        logDir = fullfile(baseDir, sprintf('data/physio/regressors/%s/%s', sub_s, run_s));
    
        % --- Load physio ---
        load(fullfile(logDir, sprintf('physio_%s.mat', run_s)), 'physio');
    
        % --- Get Task Onsets ---
        run_onsets = tsv_table.real_start_time(tsv_table.run_num == r);
    
        % --- Get HR data ---
        t_hr = linspace(0, 589*1.1, 590);
        hr = physio.ons_secs.hr;

        % --- Cut-off spikes
        hr = cap_hr_changes(hr, max_jump);
    
        % --- Smoothing ---
        hr = smooth_hr(hr, win_size);
        
        if plot_run
            % --- Plot raw HR for this run ---
            figure('Position', [100, 100, 1200, 600]);
            plot(t_hr, hr, 'b', 'LineWidth', 1.5); hold on;
            xline(run_onsets, 'r--', 'Task onset', 'LineWidth', 0.8, 'HandleVisibility', 'off');
            xlabel('Time (s)');
            ylabel('Heart Rate (BPM)');
            title(sprintf('Raw HR for %s, run %d', sub_s, r));
            % grid on;
            drawnow;
        end

        % --- Extract event-related segments ---
        n_events = length(run_onsets);
        segments = nan(n_events, length(t_common));
        for e = 1:n_events
            t0 = run_onsets(e);
            t_rel = t_hr - t0;   % relative time
            idx = t_rel >= win(1) & t_rel <= win(2);
            hr_seg = hr(idx);
            t_seg  = t_rel(idx);
            segments(e,:) = interp1(t_seg, hr_seg, t_common, 'linear', NaN);
        end
    
        % --- Collect runs of a subject ---
        subj_segments = [subj_segments; segments];
    end

    % --- Collect ubjects ---
    all_segments = [all_segments; subj_segments];

    if plot_subj
        % --- Average across all runs ---
        HR_avg = nanmean(subj_segments, 1);
        
        % --- Plot event-related HR ---
        figure('Position', [100, 100, 1200, 600]); hold on;
        plot(t_common, subj_segments', 'Color', [0.8 0.8 0.8]);  % individual trials
        plot(t_common, HR_avg, 'k', 'LineWidth', 2);            % mean
        xline(0, 'r--', 'Task onset');
        xlabel('Time relative to onset (s)');
        ylabel('Heart Rate (BPM)');
        title(sprintf('Average Event-related HR (N=%d events) for %s', size(all_segments,1), sub_s));
        grid on;
    end
end

if plot_all
    % --- Average across all runs ---
    HR_avg = nanmean(all_segments, 1);
    
    % --- Plot event-related HR ---
    figure('Position', [100, 100, 1200, 600]); hold on;
    plot(t_common, all_segments', 'Color', [0.8 0.8 0.8]);  % individual trials
    plot(t_common, HR_avg, 'k', 'LineWidth', 2);            % mean
    xline(0, 'r--', 'Task onset');
    xlabel('Time relative to onset (s)');
    ylabel('Heart Rate (BPM)');
    title(sprintf('Average Event-related HR'));
    grid on;
end

%% --- Functions ---

function subj_name = getSubj(workDir, excluded_subj)
    pinfo = readtable(sprintf('%s/FunctionalFusion/Social/participants.tsv', workDir), ...
                      'FileType','text','Delimiter','\t','VariableNamingRule','preserve');
    subj_name = pinfo.participant_id(pinfo.exclude==0 & pinfo.pilot==0);
    subj_name = subj_name(~ismember(subj_name, excluded_subj));
end

function hr_out = cap_hr_changes(hr_in, max_delta)
% hr_in: vector of HR values
% max_delta: maximum allowed change per second (or per sample)
% Compares each point to 3 points before

hr_out = hr_in;
n = length(hr_in);

for t = 4:n  % start at 4 to have 3 previous points
    % sum of differences to previous three points
    sum_abs_diff = abs(hr_in(t) - hr_in(t-1)) + abs(hr_in(t) - hr_in(t-2)) ...
                   + abs(hr_in(t) - hr_in(t-3));
    
    if sum_abs_diff > 3*max_delta
        hr_out(t) = NaN;  % mark as invalid
    end
end

% Interpolate to fill NaNs
nan_idx = isnan(hr_out);
hr_out(nan_idx) = interp1(find(~nan_idx), hr_out(~nan_idx), find(nan_idx), 'linear');

end

function hr_smooth = smooth_hr(hr_raw, win_size)
% hr_raw   = HR signal
% win_size = moving average window length (in samples)

    if win_size > 1
        hr_smooth = movmean(hr_raw, win_size);
    else
        hr_smooth = hr_raw;
    end
end