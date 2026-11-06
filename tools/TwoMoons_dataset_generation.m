%% Generate two-moons dataset
% [1] F. Pedregosa, et al., "Scikit-learn: Machine learning in Python", 
%       J. Mach. Learn. Res. 12, 2825–2830 (2011).
% ------------------------------------------------------------------------
clear; clc; rng(101);

noise       = 0.10;     % increase for harder task, e.g. 0.12-0.20
Ndata_tot   = 10000;
TestRatio   = 20;       % [%]

if mod(Ndata_tot,2) ~= 0
    Ndata_tot = Ndata_tot + 1;
    warning('Ndata_tot changed to %d to have equal numbers of Moon A and Moon B samples.',Ndata_tot);
end

Ndata_test  = round(Ndata_tot*TestRatio/100);
Ndata_train = Ndata_tot - Ndata_test;

N0 = floor(Ndata_tot/2);
N1 = Ndata_tot - N0;

%% Moon 0
theta0 = pi*rand(N0,1);
x0 = [cos(theta0), sin(theta0)];

%% Moon 1
theta1 = pi*rand(N1,1);
x1 = [1 - cos(theta1), 0.5 - sin(theta1)];

%% Add noise
x0 = x0 + noise*randn(size(x0));
x1 = x1 + noise*randn(size(x1));

%% Combine
X = [x0; x1];
LabelID = [zeros(N0,1); ones(N1,1)];

%% Shuffle
idx = randperm(Ndata_tot);
X = X(idx,:).';
LabelID = LabelID(idx).';

%% Split train/test
TrainData = X(:,1:Ndata_train);
TrainLabelID = LabelID(:,1:Ndata_train);
TestData = X(:,Ndata_train+1:end);
TestLabelID = LabelID(:,Ndata_train+1:end);

%% One-hot labels
Nclass = 2;
TrainLabels = zeros(Nclass,Ndata_train);
TestLabels  = zeros(Nclass,Ndata_test);
for nn = 1:Ndata_train
    TrainLabels(TrainLabelID(nn)+1,nn) = 1;
end
for nn = 1:Ndata_test
    TestLabels(TestLabelID(nn)+1,nn) = 1;
end
Labels = [{"Moon A"};{"Moon B"}];

%% Save
save('twomoons_dataset.mat', ...
    'TrainData','TrainLabels','TestData','TestLabels','Labels','noise');

%% Quick check
[~,TrainLabelPlot] = max(TrainLabels,[],1);
TrainLabelPlot = TrainLabelPlot - 1; % back to 0/1

figure; hold on; box on; grid on;
scatter(TrainData(1,TrainLabelPlot==0),TrainData(2,TrainLabelPlot==0),8,'filled');
scatter(TrainData(1,TrainLabelPlot==1),TrainData(2,TrainLabelPlot==1),8,'filled');
axis equal;
xlabel('x_1');
ylabel('x_2');
title(sprintf('Two moons dataset, noise = %.2f',noise));
set(gca,'fontsize',12);
