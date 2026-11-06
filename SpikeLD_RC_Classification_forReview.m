%% Reservoir Computing using Spiking-LD experiment code
% N.P. Diamantopoulos, Device Technology Labs, NTT, Inc., Japan
% last update: 2026-Nov-02
% -------------------------------------------------------------------------
clear; tic; clc; close all; rng(101);
fprintf('----------------------------------------------\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plot_on         = 1;                % {0:OFF,1:ON} summary plot
par.Task        = 'MNIST-28';       % This review script is configured for "MNIST-28" task only

par.Ib_range    = [4.4];            % [mA] (positive polarity only)
par.Rs_range    = [100];            % [GBaud]
par.Vpp_range   = [1.10];           % [Vpp]

par.RCroff      = 0.1;              % roll-off of raised-cosine filter
par.nSamp       = 8;                % [Sa/sym]
par.RClen       = 2;                % [symbols]

par.AWG_Rsampl_range = [100,130];   % [GHz] {min,max}
par.DSO_Rsampl  = 160e9;            % [Sa/s] DSO sampling rate

par.Mask_on     = 1;                % {0:OFF,1:ON} projection mask
par.Mask_mode   = 'blockwise';      % {'blockwise','all2all'}
par.Mask_len    = [16];             % random mask length - vector
par.pausetime   = 2;                % [ns] null time between data points

par.train_method= 'SGD-Adam';       % {'SGD-Adam','MMSE'}
par.epochs      = 100;              % epochs for training (Adam)
par.retrain     = 0;                % {0:OFF,1:ON} re-train weights from scratch (Adam)
par.freezeW     = 0;                % {0:OFF,1:ON} freeze weights (Adam)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Set folders
script_folder = fileparts(mfilename('fullpath'));
metadata_folder   = [fullfile(script_folder,'MNIST_metadata'),filesep];
results_folder    = [fullfile(script_folder,'MNIST_results'),filesep];
if ~isfolder(metadata_folder)
    error('Missing MNIST input folder: %s',metadata_folder);
end
if ~isfolder(results_folder)
    mkdir(results_folder);
end

%% Save command-window log
log_file = [results_folder,'RunLog_',datestr(now,'yyyymmdd_HHMMSS'),'.txt'];
diary(log_file);
diary on;
fprintf('Log file: %s\n',log_file);

%% Initialize figures
if plot_on == 1
    f1 = figure('Name','Live update','Visible',0,'color','w');
    f2 = figure('Name','DSO block diagnostics','Visible',0,'color','w');
    f3 = figure('Name','Training performance','Visible',0,'color','w');
    f4 = figure('Name','Confusion matrix','Visible',0,'color','w');
    f5 = figure('Name','Accuracy vs nodes','Visible',0,'color','w');
end

%% Initialize params
par.Ib_range = abs(par.Ib_range);       % ensure positive
par.Vpp_range = sort(par.Vpp_range);    % lowest Vpp first
par.nSamp = round(par.nSamp);           % ensure integrer
par.AWG_MaxDataLength = 2^19;
if strcmp(par.train_method,'SGD-Adam')
    par.mu          = 1e-4;             % learning rate
    par.Adam_b1     = 0.9;              % Adam hyperparameter
    par.Adam_b2     = 0.999;            % Adam hyperparameter
    par.Adam_eps    = 1e-8;             % Adam hyperparameter
elseif strcmp(par.train_method,'MMSE')
    par.epochs      = 1;
    par.val_frac    = 0.20;             % fraction of training set used for validation
    par.lambda_range= [0,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,1e2,1e3]; % for MMSE regularization
end
if par.freezeW == 1
    par.epochs      = 1;
end

%% Load task data
switch char(par.Task)
    case 'MNIST-28'
        par.MNIST_npx = sscanf(par.Task,'MNIST-%d');
        task_name = 'mnist';
        dataset_file = fullfile(metadata_folder, ...
            sprintf('mnist_dataset_%dx%d.mat',par.MNIST_npx,par.MNIST_npx));

    case 'TwoMoons'
        warning('Two-Moons analysis is included for reference only.');
        task_name = 'twomoons';
        dataset_file = fullfile(metadata_folder,'twomoons_dataset.mat');

    otherwise
        error('Task "%s" is not supported.',par.Task);
end

if ~isfile(dataset_file)
    error('Missing dataset: %s',dataset_file);
end
S = load(dataset_file);

%% Prepare data and labels
par.Ndata_train = length(S.TrainLabels);
par.Ndata_test  = length(S.TestLabels);
par.Ndata_tot   = par.Ndata_train + par.Ndata_test;
par.Nclass      = numel(S.Labels);
par.TaskLabels  = S.Labels;

T0 = [S.TrainLabels,S.TestLabels]; % One-hot labels

if strcmp(par.Task,'MNIST-28')
    n = par.MNIST_npx;
    par.kM = 2;

    if mod(n,par.kM) ~= 0
        error('MNIST pixel size must be divisible by kM.');
    end

    Data = [reshape(S.TrainData,n*n,[]),reshape(S.TestData,n*n,[])];
    X0 = reshape(permute(reshape(Data,par.kM,n/par.kM,n,par.Ndata_tot), ...
        [1 3 2 4]),n*n,par.Ndata_tot);
else
    X0 = [S.TrainData,S.TestData];
end

%% Mask configuration
if par.Mask_on
    switch par.Mask_mode
        case 'blockwise'
            if ~strcmp(par.Task,'MNIST-28')
                error('Blockwise masking is supported for MNIST only.');
            end
            par.mask_in_len = par.kM^2;
        case 'all2all'
            par.mask_in_len = size(X0,1);
        otherwise
            error('Unsupported mask mode: %s',par.Mask_mode);
    end
end

%% Check loaded task
fprintf('\nLoaded task: %s\n',par.Task);
fprintf('X0 size: %d x %d\n',size(X0,1),size(X0,2));
fprintf('T0 size: %d x %d\n',size(T0,1),size(T0,2));
fprintf('Nclass = %d\n',par.Nclass);
fprintf('Ntrain = %d, Ntest = %d, Ntot = %d\n', ...
    par.Ndata_train,par.Ndata_test,par.Ndata_tot);

fprintf('Data processing begins...\n');

%% Normalization
X0 = X0 - mean(X0,'all');
X0 = X0/max(abs(X0),[],'all');

%% Load consolidated MNIST metadata
filename_meta = sprintf('%.1fGBd_%.1fmA_%.2fVpp_%dnodes', ...
    par.Rs_range(1),par.Ib_range(1),par.Vpp_range(1),par.Mask_len(1));
Metadata_file = [metadata_folder,'MNISTmeta_',filename_meta,'.mat'];
if ~isfile(Metadata_file)
    error('Missing MNIST metadata: %s',Metadata_file);
end
Metadata = load(Metadata_file);
fprintf('Loaded MNIST metadata: %s\n',Metadata_file);

%% Projection mask
if mod(size(X0,1),par.mask_in_len) ~= 0
    error('X0 feature length %d is not divisible by mask_in_len = %d.', ...
        size(X0,1),par.mask_in_len);
end

for nodes = par.Mask_len

    %% Load projection mask
    if ~isfield(Metadata,'Mask_save')
        error('Mask_save not found in %s',Metadata_file);
    end
    par.Mask = Metadata.Mask_save;
    if ~isequal(size(par.Mask),[nodes,par.mask_in_len])
        error('Incorrect mask dimensions in %s',Metadata_file);
    end
    fprintf('Loaded projection mask: %s\n',Metadata_file);

    %% Apply projection mask
    passes = size(X0,1)/par.mask_in_len;
    par.Nnodes = passes*nodes;
    U0 = zeros(par.Nnodes,par.Ndata_tot);
    for ff = 1:passes
        ssu = (ff-1)*nodes + 1;
        ssx = (ff-1)*par.mask_in_len + 1;
        U0(ssu:ssu+nodes-1,:) = par.Mask*X0(ssx:ssx+par.mask_in_len-1,:);
    end
    U0 = U0 - mean(U0,'all');
    U0 = U0/max(abs(U0),[],'all');

    %% Main loop
    for ib = par.Ib_range

        for rs = par.Rs_range

            %% Load synchronization pattern
            if ~isfield(Metadata,'Sync_save') || ~isfield(Metadata,'SyncMeta')
                error('Incomplete sync metadata: %s',Metadata_file);
            end
            par.Sync         = Metadata.Sync_save;
            par.SyncRep      = Metadata.SyncMeta.SyncRep;
            par.Sync_nSamp   = Metadata.SyncMeta.Sync_nSamp;
            par.SyncLen      = Metadata.SyncMeta.SyncLen;
            par.Sync_Rs      = Metadata.SyncMeta.Sync_Rs;
            par.Sync_Ampl    = Metadata.SyncMeta.Sync_Ampl;
            par.SyncDummySym = Metadata.SyncMeta.SyncDummySym;
            if numel(par.Sync) ~= par.SyncRep*par.SyncLen*par.Sync_nSamp
                error('Incorrect sync pattern length: %s',Metadata_file);
            end
            if isfield(Metadata.SyncMeta,'rs') && abs(Metadata.SyncMeta.rs-rs) > 1e-9
                error('Sync symbol rate mismatch: %s',Metadata_file);
            end
            fprintf('Loaded sync pattern: %s\n',Metadata_file);

            %% Set AWG sampling rate
            AWG_Rsampl_tmp = floor(max(par.AWG_Rsampl_range)/rs)*rs;
            if AWG_Rsampl_tmp > min(par.AWG_Rsampl_range)
                par.AWG_Rsampl = AWG_Rsampl_tmp;
            else
                par.AWG_Rsampl = floor(max(par.AWG_Rsampl_range)/rs*10)*rs/10;
                if  par.AWG_Rsampl < min(par.AWG_Rsampl_range)
                    error('Can''t find suitable value for "AWG_Rsampl"!');
                end
            end
            par.nSamp_AWG = par.AWG_Rsampl/rs;
            fprintf('AWG sampling rate was set to %f GHz\n',par.AWG_Rsampl);

            %% Calculate no. sample points per DSO capture
            par.Nsym_pause = round(par.pausetime*rs); % no. pause symbols
            par.Nsym_per_point = par.Nnodes + par.Nsym_pause;
            T_per_point = par.Nsym_per_point/rs;                 % [ns]
            T_AWG_output = par.AWG_MaxDataLength/par.AWG_Rsampl; % [ns]
            par.Ndata_per_capture = floor(60/100*T_AWG_output/T_per_point);

            flag_init = 1;
            for vpp = par.Vpp_range

                %% Set filename0
                filename0 = sprintf('%.1fGBd_%.1fmA_%.2fVpp_%dnodes',rs,ib,vpp,nodes);
                fprintf('\n----------------------------------------------\n');
                fprintf(['Next Run: ',filename0,'\n']);

                %% Original pretrained weights
                W_file    = [metadata_folder,'W_',filename0,'.mat'];
                bias_file = [metadata_folder,'bias_',filename0,'.mat'];
                require_weights = par.freezeW || (strcmp(par.train_method,'SGD-Adam') && par.retrain == 0);
                if require_weights && (~isfile(W_file) || ~isfile(bias_file))
                    error(['Missing MMSE-initialized weights for %s.\n' ...
                        'Expected files:\n%s\n%s\n' ...
                        'Run MMSE first or set par.retrain=1 to train from scratch.'], ...
                        filename0,W_file,bias_file);
                end
                if isfile(W_file) && isfile(bias_file) && par.retrain==0
                    load(W_file);
                    load(bias_file);
                    par.W = W_save;
                    par.bias = bias_save;
                    fprintf('Loaded trained W/bias: %s\n',filename0);
                else
                    sigma = 0.01;
                    par.W = sigma*randn(par.Nclass,par.Nnodes);  % N(0,sigma^2)
                    par.bias = zeros(par.Nclass,1);
                end

                par.MSE_epochs = 1000*ones(1,par.epochs);
                par.CEL_epochs = par.MSE_epochs;
                par.Accuracy = 0*par.MSE_epochs;
                if strcmp(par.train_method,'MMSE')
                    X_train = zeros(size(U0));
                    par.epochs = 1;
                end

                %% Load time synchronization metadata
                par.Nblocks = ceil(par.Ndata_tot/par.Ndata_per_capture);
                if ~isfield(Metadata,'start_frame_save')
                    error('start_frame_save not found in %s',Metadata_file);
                end
                par.start_frame = Metadata.start_frame_save(:).';
                if numel(par.start_frame) ~= par.Nblocks || ...
                        any(~isfinite(par.start_frame)) || ...
                        any(par.start_frame < 1) || ...
                        any(par.start_frame ~= round(par.start_frame))
                    error('Invalid start-frame metadata: %s',Metadata_file);
                end
                if isfield(Metadata,'SyncPeakNorm_save')
                    par.SyncPeakNorm_block = Metadata.SyncPeakNorm_save;
                else
                    par.SyncPeakNorm_block = NaN(1,par.Nblocks);
                end
                fprintf('Loaded start frames: %s\n',Metadata_file);

                Y_test = NaN(par.Nclass,par.Ndata_test);
                T_test = T0(:,par.Ndata_train+1:end);

                %% Measurements
                X_cache = zeros(par.Nnodes,par.Ndata_tot);
                epochs_eff = par.epochs; % process/training epochs
                for kkk = 1:epochs_eff

                    if strcmp(par.train_method,'SGD-Adam')
                        mu = par.mu*0.5^(kkk/20);
                    end

                    %% Init Loss
                    par.confMat = zeros(par.Nclass,par.Nclass);

                    % Diagnostics per DSO block/run
                    par.MSE                 = NaN(1,1e4);
                    par.CEL                 = NaN(1,1e4);
                    par.Acc_all_block       = NaN(1,1e4);

                    ii = 1; run = 0;
                    while ii <= par.Ndata_tot

                        %% Define time series symbols
                        run = run + 1;
                        if ii+par.Ndata_per_capture-1 < par.Ndata_tot
                            ii_end = ii+par.Ndata_per_capture-1;
                            indx_true = ii:ii_end;
                            indx_awg  = indx_true; % samples sent to AWG
                        else
                            ii_end = par.Ndata_tot;
                            indx_true = ii:ii_end;
                            Npad = par.Ndata_per_capture - numel(indx_true);
                            indx_pad = randi(par.Ndata_tot,1,Npad); % Pad AWG payload
                            indx_awg = [indx_true, indx_pad];
                        end
                        par.Nbatch = numel(indx_true);
                        T = T0(:,indx_true);

                        if kkk == 1
                            X = U0(:,indx_awg);
                            TxModSym = reshape([X;zeros(par.Nsym_pause,size(X,2))],[],1);
                            TxModSym = TxModSym - mean(TxModSym);
                            TxModSym = TxModSym/max(abs(TxModSym)); % Normalize
                            par.Nsym = length(TxModSym);
                            dummy = zeros(par.SyncDummySym*par.Sync_nSamp,1);
                            TxSym0 = [dummy;par.Sync_Ampl*par.Sync;zeros(par.Nsym_pause,1);TxModSym];

                            %% Adjust pattern length
                            if flag_init == 1; par.DummyLen = 0; end
                            while true

                                while true
                                    dummy = mean(TxSym0)*ones(par.DummyLen,1);
                                    TxSym = [TxSym0;dummy];
                                    par.TxSymLen = length(TxSym);
                                    Nsampl_tmp = par.TxSymLen*par.nSamp_AWG;
                                    if      floor(Nsampl_tmp)==Nsampl_tmp; break; % check if integer
                                    elseif  par.DummyLen>5e4; error('Can''t find suitable value for "TxNsymLen"!');
                                    else,   par.DummyLen = par.DummyLen + 1;
                                    end
                                end

                                %% Prepare data for AWG
                                % for Keysight AWG: total length must be multiple of 512
                                par.TxSymAvePow = mean(abs(TxSym).^2,1);
                                if par.nSamp > 1
                                    if mod(par.RClen,2)~=0, par.RClen = par.RClen + 1; end
                                    hTxfilter = rcosdesign(par.RCroff,par.RClen,par.nSamp,'sqrt'); % RRC filter
                                    TxSampl = upfirdn(TxSym,hTxfilter,par.nSamp,1);
                                    ss = par.RClen*par.nSamp/2;
                                    ee = ss + length(TxSym)*par.nSamp - 1;
                                    TxSampl = TxSampl(ss:ee); % cancel filter delay
                                    TxSampl = TxSampl - mean(TxSampl);
                                end
                                if par.nSamp_AWG ~= par.nSamp
                                    TxSampl = resample(TxSampl,par.AWG_Rsampl*10,par.nSamp*rs*10);
                                end
                                Nsym_cut = 10;
                                ss = round(Nsym_cut*par.nSamp_AWG);
                                ee = length(TxSampl) - round(Nsym_cut*par.nSamp_AWG);
                                meanV = mean(TxSampl);
                                TxSampl(1:ss-1) = meanV;
                                TxSampl(ee+1:end) = meanV;
                                TxSamplLen = length(TxSampl);
                                DummySamplLen = 512*(ceil(TxSamplLen/512))-TxSamplLen;
                                dummy = meanV*ones(DummySamplLen,1);
                                TxSig = [TxSampl;dummy];
                                TxSig = TxSig - mean(TxSig);
                                TxSig = sqrt(par.TxSymAvePow)*TxSig/sqrt(mean(abs(TxSig).^2,1));
                                TxSig = TxSig/max(abs(TxSig));

                                par.TxSigLen = length(TxSig); % [samples]
                                if  floor(par.TxSigLen/par.nSamp_AWG) == par.TxSigLen/par.nSamp_AWG % check if integer
                                    break;
                                else
                                    par.DummyLen = par.DummyLen + 1;
                                end
                            end

                            if flag_init == 1
                                flag_init = 0;
                                fprintf('Total samples to AWG: %d\n',par.TxSigLen);
                            end

                            %% Load DSO data
                            DSO_filename = sprintf('DSOdata_%s_%d',filename0,ii_end);
                            DSO_filepath = [metadata_folder,DSO_filename,'.bin'];
                            if ~isfile(DSO_filepath)
                                error('Missing DSO file: %s',DSO_filepath);
                            end
                            fidDSO = fopen(DSO_filepath,'r');
                            if fidDSO < 0
                                error('Could not open DSO file: %s',DSO_filepath);
                            end
                            DSO_data = fread(fidDSO,inf,'single');
                            fclose(fidDSO);

                            if plot_on == 1 && run == 1 && kkk == 1
                                set(f1,'Visible',1); clf(f1);
                            end

                            RxSig = resample(DSO_data,par.nSamp*rs,par.DSO_Rsampl/1e9);
                            RxSig = RxSig-mean(RxSig);

                            %% Plot Signal RF Spectrum
                            if plot_on == 1 && run == 1
                                figure(f1);
                                subplot(2,3,1); cla; hold on; box on; grid on;
                                [p1,ff1] = periodogram(DSO_data,[],[],par.DSO_Rsampl,'onesided','power');
                                [p2,ff2] = periodogram(RxSig,[],[],par.nSamp*rs*1e9,'onesided','power');
                                Pow1 = 10*log10(abs(p1));
                                Pow2 = 10*log10(abs(p2));
                                plot(ff1/1e9,Pow1-max(Pow1));
                                plot(ff2/1e9,Pow2-max(Pow2));
                                xlim([0,2*rs]); ylim([-100 10]);
                                legend('DSO','Rx resampled');
                                xlabel('Frequency (GHz)');
                                ylabel('Normalized RF power (dB)');
                                set(gca,'fontsize',12);
                                drawnow limitrate;
                            end

                            %% Normalization
                            RxSigAvePow = mean(abs(RxSig).^2,1);
                            RxSig = sqrt(par.TxSymAvePow)*RxSig/sqrt(RxSigAvePow);

                            %% Use saved synchronization
                            start_idx = par.start_frame(run);
                            if start_idx + par.Nsym*par.nSamp - 1 > length(RxSig)
                                error('Insufficient DSO samples after saved start frame: %s',DSO_filename);
                            end

                            %% Downsampling & Thresholding
                            RxSig = RxSig(start_idx:end);
                            RxSig = RxSig(1:floor(length(RxSig)/par.nSamp)*par.nSamp);
                            RxSym = reshape(RxSig,par.nSamp,[]);
                            RxSym = mean(RxSym,1);          % downsampling
                            RxSym = RxSym - mean(RxSym);
                            RxSym = RxSym/max(abs(RxSym));  % normalization
                            SpikeOut = RxSym(1:par.Nsym);
                            SpikeOut = reshape(SpikeOut,par.Nsym_per_point,[]);

                            %% Plot received waveform
                            if plot_on == 1 && run == 1
                                subplot(2,3,4); cla; hold on; box on; grid on;
                                Nplot = min(1.6e6,length(RxSig));
                                Rx_plot = RxSig(1:Nplot);
                                Rx_plot = Rx_plot - min(Rx_plot);
                                Rx_plot = Rx_plot/max(abs(Rx_plot));
                                Rx_plot = 2*Rx_plot - 1;
                                DClvl = mean(Rx_plot)*ones(size(Rx_plot));
                                plot(Rx_plot,'k');
                                plot(DClvl,'b--','LineWidth',1.2);
                                [Nhist,edges] = histcounts(Rx_plot,1e3,'Normalization','pdf');
                                centers = edges(1:end-1) + diff(edges)/2;
                                Nhist = smoothdata(Nhist,'gaussian',9);
                                Ns = length(Rx_plot);
                                pdf_width = 0.18*Ns;
                                x0 = 0.80*Ns;
                                Nhist_norm = Nhist/max(Nhist);
                                x_pdf = x0 + pdf_width*Nhist_norm;
                                plot(x_pdf,centers,'y','LineWidth',1.8);
                                xlim([0 Ns]); ylim([-1,1]);
                                ylabel('Normalized amplitude');
                                xlabel('Samples');
                                legend('Rx waveform','DC level','PDF','Location','nw');
                                set(gca,'fontsize',12);
                            end

                            %% Plot Detected Spikes - Raster plot
                            if plot_on == 1 && run == 1
                                figure(f1);
                                subplot(2,3,[2,5]); cla; hold on; box on; grid off;
                                ylabel('Dataset sample no.'); xlabel('Symbol no.');
                                spikes_plot = SpikeOut(:,1:min(50,size(SpikeOut,2))) > 0.0;
                                spikes_plot = (1:size(spikes_plot,2)).*spikes_plot;
                                spikes_plot(spikes_plot == 0) = NaN;
                                plot(spikes_plot,'.g','MarkerSize',3);
                                xlim([0.5,par.Nnodes+0.5]);
                                ylim([0.5,min(50.5,size(spikes_plot,2)+0.5)]);
                                set(gca,'fontsize',12,'color','k');
                                drawnow limitrate;
                            end

                            X_batch = SpikeOut(1:par.Nnodes,1:par.Nbatch);

                        else
                            X_batch = X_cache(:,ii:ii_end);
                        end

                        if strcmp(par.train_method,'MMSE')
                            X_train(:,ii:ii_end) = X_batch;
                        end
                        if kkk == 1
                            X_cache(:,ii:ii_end) = X_batch;
                        end

                        %% Training: SGD-Adam
                        if strcmp(par.train_method,'SGD-Adam')

                            % Initialize if first epoch
                            if kkk == 1 && run == 1
                                par.mW = zeros(size(par.W));
                                par.vW = zeros(size(par.W));
                                par.mB = zeros(size(par.bias));
                                par.vB = zeros(size(par.bias));
                                par.t_adam = 0;
                            end

                            CEL_tmp = 1000*ones(par.Nbatch,1);
                            MSE_tmp = CEL_tmp;

                            for xx = 1:par.Nbatch
                                X_tmp = X_batch(:,xx);
                                Z = par.W*X_tmp + par.bias;
                                Z = Z - max(Z);         % softmax stabilization
                                Y = exp(Z)/sum(exp(Z)); % multinomial (softmax) logistic regression

                                err = Y-T(:,xx);
                                MSE_tmp(xx) = mean(err.^2);
                                CEL_tmp(xx) = -sum(T(:,xx).*log(max(Y,realmin))); % categorical cross-entropy

                                if ii+xx-1 <= par.Ndata_train && ~par.freezeW

                                    % Compute gradients
                                    gradW = err*X_tmp.';
                                    gradB = err;

                                    if strcmp(par.train_method,'SGD-Adam') % Adam
                                        par.t_adam = par.t_adam + 1;

                                        % --- Adam: Weight updates ---
                                        par.mW = par.Adam_b1*par.mW + (1-par.Adam_b1)*gradW;
                                        par.vW = par.Adam_b2*par.vW + (1-par.Adam_b2)*(gradW.^2);

                                        mW_hat = par.mW/(1-par.Adam_b1^par.t_adam);
                                        vW_hat = par.vW/(1-par.Adam_b2^par.t_adam);

                                        if ~par.freezeW
                                            par.W = par.W - mu*mW_hat./(sqrt(vW_hat)+par.Adam_eps);
                                        end

                                        % --- Adam: Bias updates ---
                                        par.mB = par.Adam_b1*par.mB + (1-par.Adam_b1)*gradB;
                                        par.vB = par.Adam_b2*par.vB + (1-par.Adam_b2)*(gradB.^2);

                                        mB_hat = par.mB/(1-par.Adam_b1^par.t_adam);
                                        vB_hat = par.vB/(1-par.Adam_b2^par.t_adam);

                                        if ~par.freezeW
                                            par.bias = par.bias - mu*mB_hat./(sqrt(vB_hat)+par.Adam_eps);
                                        end

                                    else % SGD
                                        if ~par.freezeW
                                            par.W = par.W - mu*gradW;
                                            par.bias = par.bias - mu*gradB;
                                        end
                                    end

                                end
                            end

                            par.MSE(run) = mean(MSE_tmp,'omitnan');
                            par.CEL(run) = mean(CEL_tmp,'omitnan');

                            %% All-sample block accuracy diagnostic
                            Ncorr_all_run = 0;
                            Nall_run = par.Nbatch;
                            for xx_acc = 1:par.Nbatch
                                X_tmp_acc = X_batch(:,xx_acc);
                                Z_acc = par.W*X_tmp_acc + par.bias;
                                Z_acc = Z_acc - max(Z_acc);
                                Y_acc = exp(Z_acc)/sum(exp(Z_acc));

                                [~,iT_acc] = max(T(:,xx_acc));
                                [~,iY_acc] = max(Y_acc);

                                Ncorr_all_run = Ncorr_all_run + (iY_acc == iT_acc);
                            end
                            par.Acc_all_block(run) = 100*Ncorr_all_run/Nall_run;

                            %% High-CEL block diagnostic
                            if par.CEL(run)>1 && kkk == par.epochs
                                fprintf('High-CEL block: nodes=%d, run=%d, ii=%d, ii_end=%d, CEL=%.3f, Acc=%.1f%%\n', ...
                                    nodes, run, ii, ii_end, par.CEL(run), par.Acc_all_block(run));
                            end

                        end

                        if strcmp(par.train_method,'SGD-Adam')

                            %% Plot MSE/CEL Training
                            if plot_on == 1 && run == par.Nblocks && kkk == 1
                                figure(f1);
                                subplot(2,3,3); cla; hold on; grid on; box on;
                                xax = min((1:run)*par.Ndata_per_capture,par.Ndata_tot);
                                plot(xax,par.CEL(1:run),'k'); ylabel('Cross-Entropy Loss');
                                xlabel('Dataset samples'); xlim([1,ii_end]);
                                set(gca,'fontsize',12);
                                drawnow limitrate;
                            end

                            %% Inference
                            for xx = 1:par.Nbatch
                                if ii+xx-1 > par.Ndata_train
                                    X_tmp = X_batch(:,xx);
                                    Z = par.W*X_tmp + par.bias;
                                    Z = Z - max(Z);
                                    Y = exp(Z)/sum(exp(Z)); % multinomial (softmax) logistic regression

                                    test_idx = ii + xx - 1 - par.Ndata_train;
                                    Y_test(:,test_idx) = Y;

                                    [~,iT] = max(T(:,xx));
                                    [~,iY] = max(Y);
                                    par.confMat(iY,iT) = par.confMat(iY,iT) + 1;
                                end
                            end

                        end

                        ii = ii_end + 1;
                    end

                    fprintf('Epoch %d finished: Nruns = %d, last ii_end = %d, Nconf = %d\n', ...
                        kkk, run, ii_end, sum(par.confMat,'all'));

                    %% Training: MMSE (ridge regression)
                    if strcmp(par.train_method,'MMSE')

                        X_all = X_train;
                        T_all = T0;

                        %% Define effective training and test indices
                        idx_train = 1:par.Ndata_train;
                        idx_test  = par.Ndata_train+1:par.Ndata_tot;

                        %% Split training set into fit and validation sets
                        valStream = RandStream('mt19937ar','Seed',1234);
                        idx_fit = []; idx_val = [];
                        for cc = 1:par.Nclass
                            idx_class = idx_train(T_all(cc,idx_train) == 1);
                            idx_class = idx_class(randperm(valStream,length(idx_class)));
                            Nval_class = round(par.val_frac*length(idx_class));
                            Nval_class = min(Nval_class,length(idx_class)-1);
                            idx_val = [idx_val,idx_class(1:Nval_class)];
                            idx_fit = [idx_fit, idx_class(Nval_class+1:end)];
                        end

                        %% Sort indices back into original dataset order
                        idx_fit = sort(idx_fit);
                        idx_val = sort(idx_val);
                        fprintf('MMSE split: Nfit = %d, Nval = %d, Ntest = %d\n', ...
                            length(idx_fit),length(idx_val),length(idx_test));

                        fprintf('mean: %f, std: %f\n', mean(X_all(:,idx_fit),'all'), ...
                            std(X_all(:,idx_fit),1,'all'));

                        bestAcc = -Inf;
                        bestLambda = NaN;
                        X_aug = [X_all;ones(1,size(X_all,2))];

                        for lambda = par.lambda_range

                            R = X_aug(:,idx_fit)*X_aug(:,idx_fit).' + lambda*eye(size(X_aug,1));
                            R(end,end) = R(end,end)-lambda;
                            W_aug = (T_all(:,idx_fit)*X_aug(:,idx_fit).')/R;
                            W_tmp = W_aug(:,1:end-1);
                            bias_tmp = W_aug(:,end);

                            %% Validation inference
                            Z_val = W_tmp*X_all(:,idx_val) + bias_tmp;
                            Z_val = Z_val-max(Z_val,[],1);
                            Y_val = exp(Z_val)./sum(exp(Z_val),1);
                            T_val = T_all(:,idx_val);
                            err_val = Y_val-T_val;

                            %% Validation metrics
                            MSE_val = mean(err_val.^2,'all');
                            CEL_val = mean(-sum(T_val.*log(max(Y_val,realmin))));
                            confMat_val = zeros(par.Nclass,par.Nclass);
                            for xx = 1:length(idx_val)
                                [~,iT] = max(T_val(:,xx));
                                [~,iY] = max(Y_val(:,xx));
                                confMat_val(iY,iT) = confMat_val(iY,iT)+1;
                            end
                            Acc_val = 100*trace(confMat_val)/sum(confMat_val,'all');
                            fprintf( 'lambda = %.1e: Val Accuracy = %.1f%%, CEL = %.4f, MSE = %.4f\n', ...
                                lambda,Acc_val,CEL_val,MSE_val);

                            %% Select lambda from validation set only
                            if Acc_val > bestAcc
                                bestLambda = lambda;
                                bestAcc = Acc_val;
                            end

                        end

                        %% Store selected lambda
                        par.bestLambda = bestLambda;
                        fprintf('Selected lambda = %.1e: Validation Accuracy = %.1f%%\n', ...
                            par.bestLambda,bestAcc);

                        %% Refit readout using complete training set
                        lambda = par.bestLambda;
                        R = X_aug(:,idx_train)*X_aug(:,idx_train).' + lambda*eye(size(X_aug,1));
                        R(end,end) = R(end,end)-lambda;
                        W_aug = (T_all(:,idx_train)*X_aug(:,idx_train).')/R;
                        par.W = W_aug(:,1:end-1);
                        par.bias = W_aug(:,end);

                        %% Final inference on test set
                        Z_test = par.W*X_all(:,idx_test) + par.bias;
                        Z_test = Z_test-max(Z_test,[],1);
                        Y_test = exp(Z_test)./sum(exp(Z_test),1);
                        T_test = T_all(:,idx_test);
                        err_test = Y_test-T_test;

                        %% Final test metrics
                        par.MSE_epochs = mean(err_test.^2,'all');
                        par.CEL_epochs = mean(-sum(T_test.*log(max(Y_test,realmin))));
                        par.confMat = zeros(par.Nclass,par.Nclass);
                        for xx = 1:length(idx_test)
                            [~,iT] = max(T_test(:,xx));
                            [~,iY] = max(Y_test(:,xx));
                            par.confMat(iY,iT) = par.confMat(iY,iT)+1;
                        end
                        par.Accuracy = 100*trace(par.confMat)/sum(par.confMat,'all');

                        fprintf('Test: Accuracy = %.1f%%, CEL = %.4f, MSE = %.4f\n', ...
                            par.Accuracy,par.CEL_epochs,par.MSE_epochs);

                    else

                        %% Plot all DSO block diagnostics after selected epochs
                        if plot_on == 1 && (kkk == 1 || kkk == par.epochs)

                            Nruns = run;
                            run_ax = 1:Nruns;
                            CEL_plot  = par.CEL(run_ax);
                            Acc_plot  = par.Acc_all_block(run_ax);
                            if kkk == 1
                                Sync_plot = par.SyncPeakNorm_block(run_ax);
                            end

                            set(f2,'Visible',1); clf(f2); figure(f2);

                            subplot(3,1,1); hold on; box on; grid on;
                            plot(run_ax,CEL_plot,'k-o','MarkerSize',3);
                            ylabel('Block CEL');
                            title(sprintf('DSO block diagnostics: %s, epoch %d',filename0,kkk),'Interpreter','none');
                            set(gca,'fontsize',12);

                            subplot(3,1,2); hold on; box on; grid on;
                            plot(run_ax,Acc_plot,'k-o','MarkerSize',3);
                            ylabel('Block accuracy (%)');
                            ylim([0 100]);
                            set(gca,'fontsize',12);

                            if kkk == 1
                                subplot(3,1,3); hold on; box on; grid on;
                                plot(run_ax,Sync_plot,'k-o','MarkerSize',3);
                                ylabel('Sync peak');
                            end
                            xlabel('DSO block run');
                            set(gca,'fontsize',12);

                            drawnow limitrate;

                            fig_filename = [results_folder,'DSOblockDiagnostics_',filename0,'_epoch',num2str(kkk)];
                            exportgraphics(f2,[fig_filename,'.jpg'],'Resolution',200);
                        end

                        par.MSE_epochs(kkk) = mean(par.MSE(1:run),'omitnan');
                        par.CEL_epochs(kkk) = mean(par.CEL(1:run),'omitnan');
                        Nconf = sum(par.confMat,'all');
                        if Nconf > 0
                            par.Accuracy(kkk) = 100*trace(par.confMat)/Nconf;
                        else
                            par.Accuracy(kkk) = NaN;
                            warning('No test samples were evaluated in epoch %d.',kkk);
                        end
                        fprintf('#%d epoch: Accuracy = %.1f%%, CEL = %.4f, MSE = %.4f\n',kkk,par.Accuracy(kkk),par.CEL_epochs(kkk),par.MSE_epochs(kkk));
                    end

                    %% Plot confusion matrix
                    if plot_on == 1 && (kkk == 1 || kkk == par.epochs)
                        figure(f1);
                        subplot(2,3,6);

                        colsum = sum(par.confMat,1);
                        colsum(colsum == 0) = 1;
                        confMat_live = par.confMat./colsum*100;

                        h_cm = heatmap(confMat_live,'ColorbarVisible',0);
                        colormap("parula");
                        h_cm.XDisplayLabels = par.TaskLabels.';
                        h_cm.YDisplayLabels = par.TaskLabels.';
                        h_cm.YLabel = 'Prediction';
                        h_cm.XLabel = 'Actual targets';
                        h_cm.CellLabelFormat = '%.1f%%';
                        h_cm.FontSize = 10;

                        drawnow limitrate;
                    end

                end

                %% Save data
                if plot_on == 1

                    %% Plot error/accuracy vs. epochs
                    if par.epochs > 1
                        set(f3,'Visible',1); clf(f3); figure(f3);
                        hold on; grid on; box on;

                        yyaxis left;
                        plot(par.Accuracy,'k','LineWidth',1.2);
                        ylabel('Accuracy (%)');
                        xlabel('Epochs');
                        set(gca,'fontsize',12);

                        yyaxis right;
                        plot(par.CEL_epochs,'LineWidth',1.2); ylabel('Cross-Entropy Loss');
                        set(gca,'fontsize',12);

                        title(sprintf('Training performance: %s',filename0),'Interpreter','none');
                        drawnow limitrate;

                        fig_filename = [results_folder,'Training_',filename0];
                        exportgraphics(f3,[fig_filename,'.jpg'],'Resolution',200);
                    end

                    %% Plot last confusion matrix
                    colsum = sum(par.confMat,1);
                    colsum(colsum == 0) = 1;
                    confMat_plot = par.confMat./colsum*100;
                    set(f4,'Visible',1); clf(f4); figure(f4); box on;
                    h_cm = heatmap(confMat_plot,'ColorbarVisible',0);
                    colormap("parula");
                    h_cm.XDisplayLabels = par.TaskLabels.';
                    h_cm.YDisplayLabels = par.TaskLabels.';
                    h_cm.YLabel = 'Prediction';
                    h_cm.XLabel = 'Actual targets';
                    h_cm.CellLabelFormat = '%.1f%%';
                    h_cm.FontSize = 14;
                    title_str = sprintf('Confusion matrix: %s, Accuracy = %.1f%%', ...
                        strrep(filename0,'_','\_'), par.Accuracy(end));
                    h_cm.Title = title_str;

                    drawnow limitrate;

                    fig_filename = [results_folder,'ConfMat_',filename0];
                    exportgraphics(f4,[fig_filename,'.jpg'],'Resolution',200);

                end

                %% Save results
                Results_file = [results_folder,'Results_',filename0,'.mat'];
                save(Results_file,'par','Y_test','T_test','-v7.3');
            end
        end
    end
end

%% Results summary ---
for rs = par.Rs_range
    for vpp = par.Vpp_range

        Accuracy_all = NaN(length(par.Ib_range),length(par.Mask_len));

        %% Load all results
        for ib_idx = 1:length(par.Ib_range)
            ib = par.Ib_range(ib_idx);

            for nodes_idx = 1:length(par.Mask_len)
                nodes = par.Mask_len(nodes_idx);

                filename0 = sprintf('%.1fGBd_%.1fmA_%.2fVpp_%dnodes',rs,ib,vpp,nodes);
                Results_file = [results_folder,'Results_',filename0,'.mat'];

                if ~isfile(Results_file)
                    warning('Missing Results file: %s',Results_file);
                else
                    Sres = load(Results_file,'par');
                    Accuracy_all(ib_idx,nodes_idx) = Sres.par.Accuracy(end);
                end
            end
        end

        %% Grouped bar plot
        if plot_on == 1

            set(f5,'Visible',1); clf(f5); figure(f5);
            hold on; box on; grid on;
            bar(par.Mask_len,Accuracy_all.','grouped');
            xlabel('Number of nodes');
            ylabel('Accuracy (%)');
            if length(par.Mask_len) > 1
                dx = min(diff(sort(par.Mask_len)));
                xlim([min(par.Mask_len)-0.5*dx,max(par.Mask_len)+0.5*dx]);
            else
                xlim(par.Mask_len + [-0.5 0.5]);
            end
            ylim([round(min(Accuracy_all,[],'all'))-1 min(round(max(Accuracy_all,[],'all'))+1,100)]);
            legend(compose('%.1f mA',par.Ib_range),'Location','nw');
            set(gca,'fontsize',12,'Xtick',par.Mask_len);
            filename0_summary = sprintf('%.1fGBd_%.2fVpp',rs,vpp);
            title(sprintf('Accuracy vs nodes: %s',filename0_summary),'Interpreter','none');

            drawnow limitrate;

            fig_filename = [results_folder,'ResultsFinal_',filename0_summary];
            exportgraphics(f5,[fig_filename,'.jpg'],'Resolution',200);
        end
    end
end

%% Total elapsed time
elapsed_s = toc;
days  = floor(elapsed_s/86400);
hours = floor(mod(elapsed_s,86400)/3600);
mins  = floor(mod(elapsed_s,3600)/60);
secs  = mod(elapsed_s,60);
fprintf('Total elapsed time: %d days, %d hours, %d mins, %.1f secs\n', ...
    days, hours, mins, secs);

diary off;
