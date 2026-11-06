%% Mackey-Glass Simulation using RK4
% N.P. Diamantopoulos, NTT, Inc.
% updated: 2026-Oct-06
% -------------------------------------------------------------------------
clear; clc;

% MG series parameters
beta        = 0.2;
gamma       = 0.1;
n           = 10;
tau         = 17;       % Delay in Mackey-Glass time units

dt_int      = 0.1;      % RK4 numerical integration step
dt_sample   = 1.0;      % Sampling interval used for RC dataset
N_sampled   = 10000;    % MG samples retained after resampling
t_transient = 1000;     % Initial transient discarded

% Parameters for ML windowing
k           = 25;       % Input window: 25 past MG samples
m           = 10;       % Target: next 10 MG samples
stride      = 1;        % Sliding-window step

% -------------------------------------------------------------------------

% Sampling ratio
sample_factor = dt_sample/dt_int;
if abs(sample_factor - round(sample_factor)) > 1e-12
    error('dt_sample must be an integer multiple of dt_int.');
end
sample_factor = round(sample_factor);

fprintf('RK4 integration step      = %.3f\n', dt_int);
fprintf('RC sampling interval      = %.3f\n', dt_sample);
fprintf('Downsampling factor       = %d\n', sample_factor);

% Time vector
t_max = t_transient + (N_sampled - 1)*dt_sample;
t = 0:dt_int:t_max;
steps = length(t);

% Initialize state vector
x = zeros(1, steps);
x(1) = 1.2;

% Function to evaluate Mackey-Glass derivative
mg_deriv = @(x_t, x_tau) beta*x_tau/(1 + x_tau^n) - gamma*x_t;

% Delay in numerical integration samples
delay_steps = round(tau/dt_int);
if abs(delay_steps*dt_int - tau) > 1e-12
    error('tau must be compatible with dt_int for direct delay indexing.');
end

% Main RK4 loop
for i = 2:steps
    delay_idx = i - delay_steps;

    if delay_idx <= 0
        x_tau = 0;
    else
        x_tau = x(delay_idx);
    end

    % RK4 steps
    k1 = dt_int * mg_deriv(x(i-1),             x_tau);
    k2 = dt_int * mg_deriv(x(i-1) + 0.5*k1,    x_tau);
    k3 = dt_int * mg_deriv(x(i-1) + 0.5*k2,    x_tau);
    k4 = dt_int * mg_deriv(x(i-1) + k3,        x_tau);

    x(i) = x(i-1) + (k1 + 2*k2 + 2*k3 + k4)/6;
end

% Plot result
figure;
plot(t,x,'-','LineWidth',1.2);
xlabel('Time');
ylabel('x(t)');
title(sprintf('Mackey-Glass trajectory, RK4 dt = %.1f',dt_int));
grid on;

% Discard & resample
idx_start = find(t >= t_transient,1,'first');
x_ss = x(idx_start:end);
t_ss = t(idx_start:end);
x_sampled = x_ss(1:sample_factor:end);
t_sampled = t_ss(1:sample_factor:end);

% Restrict to requested number of logical samples
N_keep = min(N_sampled,length(x_sampled));
x_sampled = x_sampled(1:N_keep);
t_sampled = t_sampled(1:N_keep);
fprintf('Number of sampled MG points = %d\n',length(x_sampled));

% Plot sampled series
figure;
plot(t_sampled,x_sampled,'o-','MarkerSize',3);
xlabel('Mackey-Glass time');
ylabel('x(t)');
title(sprintf('Mackey-Glass samples used for RC, \\Deltat_{sample} = %.1f', dt_sample));
grid on;

% Number of usable samples
N = length(x_sampled);
num_samples = floor((N-k-m)/stride) + 1;

% Initialize input/output matrices
X = zeros(num_samples,k);
T = zeros(num_samples,m);

% Construct prediction dataset
for i = 1:num_samples
    start_idx = (i - 1) * stride + 1;
    X(i,:) = x_sampled(start_idx : start_idx+k-1);      % Input: past k values
    T(i,:) = x_sampled(start_idx+k : start_idx+k+m-1);  % Output: next m values
end

% Optional: Normalize data (zero mean, unit variance)
all_values = [X(:); T(:)];
mu_all  = mean(all_values);
std_all = std(all_values,1);
X = (X - mu_all)/std_all;
T = (T - mu_all)/std_all;

% Optional: plot one example
figure;
subplot(2,1,1);
plot(1:k,X(1,:),'o-');
title('Input Window (Past k Values)');
xlabel('Time Steps'); ylabel('x');
subplot(2,1,2);
plot(1:m,T(1,:),'s-');
title('Target Output (Next m Values)');
xlabel('Time Steps Ahead'); ylabel('x');

% Save
save('MackeyGlass_dataset.mat','X','T');
