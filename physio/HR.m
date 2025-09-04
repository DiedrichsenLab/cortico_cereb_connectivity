clc; clear; close all;

workdir = '/cifs/diedrichsen/data';
baseDir = fullfile(workdir, 'Cerebellum/Social');
logDir = fullfile(baseDir, 'data/physio/sub-03/ses-01/');

% Load physio from TAPAS output
load(fullfile(logDir, 'physio_run-01.mat'), 'physio');

% Read the raw PULS log to get triggers
tbl = readTable();

acq_time = tbl.ACQ_TIME_TICS;

% Get trigger times
trigger_idx = strcmp(tbl.SIGNAL, 'PULS_TRIGGER');
trigger_tics = acq_time(trigger_idx);

% Convert tics to seconds
dt_tics = 2.5e-3;  % 2.5 ms per tic
trigger_times = (trigger_tics - trigger_tics(1)) * dt_tics;

% --- Compute instantaneous heart rate from TAPAS detected beats ---
pulse_times = physio.ons_secs.cpulse;   % cardiac beats in sec
IBI = diff(pulse_times);                % inter-beat intervals
HR_inst = 60 ./ IBI;                    % bpm
t_inst  = pulse_times(2:end);           % time of each HR value

% --- Interpolate onto continuous grid ---
dt = 0.1;  % time resolution for plotting
t_grid = t_inst(1):dt:t_inst(end);
HR_cont = interp1(t_inst, HR_inst, t_grid, 'linear', 'extrap');

% --- Plot 1: raw (unaligned) heart rate ---
figure;
plot(t_grid, HR_cont, 'b', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Heart Rate (BPM)');
title('Raw Heart Rate Time Series');
grid on;

% --- Align HR to triggers for event-related averaging ---
win = [-10 20];  % seconds around trigger
t_win = win(1):dt:win(2);
n_events = length(trigger_times);
HR_event = nan(n_events, length(t_win));

for e = 1:n_events
    t0 = trigger_times(e);
    t_rel = t0 + t_win;
    idx = t_rel >= t_grid(1) & t_rel <= t_grid(end);
    HR_event(e, idx) = interp1(t_grid, HR_cont, t_rel(idx));
end

HR_avg = nanmean(HR_event, 1);

% --- Plot 2: HR aligned to triggers ---
figure;
plot(t_win, HR_avg, 'r', 'LineWidth', 2);
xlabel('Time relative to PULS\_TRIGGER (s)');
ylabel('HR (BPM)');
title('Average Heart Rate aligned to PULS\_TRIGGER');
grid on;

% Add vertical line at t=0
hold on;
yl = ylim;  % current y-limits
plot([0 0], yl, 'k--', 'LineWidth', 1.5);  % dashed vertical line
text(0, yl(2), '  PULS.TRIGGER', 'VerticalAlignment','top','HorizontalAlignment','left','FontWeight','bold');
hold off;

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