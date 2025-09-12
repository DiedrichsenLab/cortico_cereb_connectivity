clc; clear; close all;

sub_s = "sub-05";
run_list = [2, 3, 4];
workdir = '/cifs/diedrichsen/data';
baseDir = fullfile(workdir, 'Cerebellum/Social');
plot_onset = true;

% Parameters
dt = 0.25;            % resampling resolution in sec for continuous HR
smooth_flag = true;   % smooth HR or not
smooth_window = 2;    % seconds
max_jump = 7;
win_size = 4;

for r = run_list
    run_s = sprintf('run-%02d', r);

    logDir = fullfile(baseDir, sprintf('data/physio/regressors/%s/%s', sub_s, run_s));
    behDir = fullfile(baseDir, sprintf('data/behavioral/%s/%s_ses-01.tsv', sub_s, sub_s));
    
    % --- Load behavioral onsets ---
    tsv_table = readtable(behDir, 'FileType','text','Delimiter','\t','VariableNamingRule','preserve');
    run_onsets = tsv_table.start_time(tsv_table.run_num == r);

    % --- Load physio ---
    load(fullfile(logDir, sprintf('physio_%s.mat', run_s)), 'physio');

    % --- Compute instantaneous HR ---
    pulse_times = physio.ons_secs.cpulse;   % cardiac beats in sec
    IBI = diff(pulse_times);
    hr = 60 ./ IBI;
    t_hr  = pulse_times(2:end);

    % --- Cut-off spikes
    hr_clean = cap_hr_changes(hr, max_jump);

    % --- Smoothing ---
    hr_smooth = smooth_hr(hr_clean, win_size);
    
    % --- Plot raw, cleaned, and smoothed HR together ---
    figure('Position', [100, 100, 1200, 600]);
    subplot(2,1,1)
    hold on;
    % Plot signals
    plot(t_hr, hr, 'k', 'LineWidth', 2);
    plot(t_hr, hr_clean, 'b', 'LineWidth', 1.5);
    plot(t_hr, hr_smooth, 'r', 'LineWidth', 1);
    xlabel('Time (s)');
    ylabel('Heart Rate (BPM)');
    title(sprintf('HR signals for %s, run %d', sub_s, r));
    legend('Raw HR', 'Cleaned HR', 'Smoothed HR');
    % grid on;
    hold off;
    subplot(2,1,2)
    real_t_hr = 0:1.1:649-1.1;
    plot(real_t_hr, physio.ons_secs.hr, 'b', 'LineWidth', 1.5)
    if plot_onset
        xline(run_onsets, 'r--', 'Task onset', 'LineWidth', 0.8,'HandleVisibility', 'off');
    end
end

%% --- Function ---
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