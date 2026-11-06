%% SANTA FE TIME SERIES A: Chaotic Laser dataset
% [1] Weigend, A.S. (1994). Time Series Prediction: Forecasting The Future 
%       And Understanding The Past (1st ed.). Routledge. 
%       https://doi.org/10.4324/9780429492648
% [2] McNames, J. (n.d.). McNames Data Sets.
%       http://web.cecs.pdx.edu/mcnames/DataSets/index.html
%--------------------------------------------------------------------------

% Load dataset from file (you must have SantaFeA.dat in working directory)
data = load('SantaFeA.dat');  % Should be a single column vector
x = data(:);                  % Ensure it's a column vector
data = load('SantaFeA2.dat'); 
x = [x;data(:)];
N = length(x);

% Plot part of the series
figure;
plot(x(1:1000));
title('Santa Fe Time Series A');
xlabel('Time Index');
ylabel('Laser Intensity');
grid on;

% Parameters
k = 25;   % Input window size
m = 10;   % Output prediction steps
stride = 1;

% Compute number of samples
num_samples = floor((N - k - m + 1) / stride);
X = zeros(num_samples, k);
T = zeros(num_samples, m);

% Construct dataset
for i = 1:num_samples
    start_idx = (i - 1) * stride + 1;
    X(i, :) = x(start_idx : start_idx + k - 1);         % Input: past k values
    T(i, :) = x(start_idx + k : start_idx + k + m - 1); % Output: next m values
end

% Optional: Normalize data (zero mean, unit variance)
X = X-mean(X,'all');
X = X/std(X,1,'all');
T = T-mean(T,'all');
T = T/std(T,1,'all');

% Example visualization
figure;
subplot(2,1,1);
plot(1:k, X(1,:), 'o-'); title('Input Window');
subplot(2,1,2);
plot(1:m, T(1,:), 's-'); title('Target Future Values');

save('santafe_dataset.mat','X','T');
