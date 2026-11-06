%% Reservoir Computing using Spiking-LD experiment code
% N.P. Diamantopoulos, Device Technology Labs, NTT, Inc., Japan
% last update: 2026-Nov-02
% -------------------------------------------------------------------------
clear; tic; clc; close all; rng(101);
fprintf('----------------------------------------------\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plot_on         = 1;                % {0:OFF,1:ON} summary plot
par.Task        = 'MackeyGlass';    % This review script is configured for "MackeyGlass" task only

par.Ib_range    = [4.3];            % [mA] (positive polarity only)
par.Rs_range    = [100];            % [GBaud]
par.Vpp_range   = [1.10];           % [Vpp]

par.RCroff      = 0.1;              % roll-off of raised-cosine filter
par.nSamp       = 8;                % [Sa/sym]
par.RClen       = 2;                % [symbols]

par.AWG_Rsampl_range = [100,130];   % [GHz] {min,max}
par.DSO_Rsampl  = 160e9;            % [Sa/s] DSO sampling rate

par.Mask_len    = [9];              % random mask length - vector
par.initT       = 2;                % [ns] null time before sending data points
par.Npre        = 25;               % pre-samples (incl. current) used for prediction

par.val_frac    = 0.20;             % fraction of training set used for validation
par.lambda_range= [0,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,1e2,1e3]; % for MMSE regularization

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Task configuration
switch char(par.Task)
    case 'MackeyGlass'
        task_folder = 'MG';
        dataset_name = 'MackeyGlass_dataset.mat';
        metadata_prefix = 'MG_';

    case 'SantaFe'
        warning('Santa Fe analysis is included for reference only.');
        task_folder = 'SantaFe';
        dataset_name = 'santafe_dataset.mat';
        metadata_prefix = 'SantaFe_'; 

    otherwise
        error('Task "%s" is not supported.',par.Task);
end

%% Set folders
script_folder = fileparts(mfilename('fullpath'));
metadata_folder = [fullfile(script_folder,[task_folder,'_metadata']),filesep];
results_folder  = [fullfile(script_folder,[task_folder,'_results']),filesep];

if ~isfolder(metadata_folder)
    error('Missing %s input folder: %s',par.Task,metadata_folder);
end
if ~isfolder(results_folder)
    mkdir(results_folder);
end

%% Save command-window log
log_file = [results_folder,'RunLog_',datestr(now,'yyyymmdd_HHMMSS'),'.txt'];
diary(log_file);
diary on;
fprintf('Log file:\n\t%s\n',log_file);

%% Initialize figures
if plot_on == 1
    f1 = figure('Name','Live update','Visible',0,'color','w');
    f2 = figure('Name','Prediction performance','Visible',0,'color','w');
end

%% Initialize
par.Ib_range = abs(par.Ib_range);       % ensure positive
par.Vpp_range = sort(par.Vpp_range);    % lowest Vpp first
par.nSamp = round(par.nSamp);           % ensure integrer
par.AWG_MaxDataLength = 2^19;

%% Load task data
dataset_file = fullfile(metadata_folder,dataset_name);
if ~isfile(dataset_file)
    error('Missing dataset: %s',dataset_file);
end
S = load(dataset_file,'X','T');
X = S.X;
T = S.T;
if      size(X,2) > par.Npre
    X = X(:,1:par.Npre);
elseif  size(X,2) < par.Npre
    par.Npre = size(X,2);
end
par.Npost = size(T,2);
par.Ndata_tot = size(T,1);
par.Ndata_test = round(par.Ndata_tot/4); % ~25% data used for testing
par.Ndata_train = par.Ndata_tot - par.Ndata_test;
fprintf('No. train data points: %d\n',par.Ndata_train);
fprintf('No. test data points: %d\n',par.Ndata_test);

%% Normalization
X0 = X-mean(X,'all');
X0 = X0/max(abs(X0),[],'all');
T0 = T-mean(T,'all');
T0 = T0/max(abs(T0),[],'all');
par.VarT = mean((T0-mean(T0)).^2);

%% Load consolidated experimental metadata
filename_meta = sprintf('%.1fGBd_%.1fmA_%.2fVpp_%dnodes', ...
    par.Rs_range(1),par.Ib_range(1),par.Vpp_range(1),par.Mask_len(1));

Metadata_file = fullfile(metadata_folder, ...
    [metadata_prefix,filename_meta,'.mat']);

if ~isfile(Metadata_file)
    error('Missing %s metadata: %s',par.Task,Metadata_file);
end
Metadata = load(Metadata_file);
fprintf('Loaded %s metadata: %s\n',par.Task,Metadata_file);

fprintf('\nData processing begins...\n');

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

        for nodes = par.Mask_len
            par.Nnodes = nodes*par.Npre; % total reservoir-state dimension

            %% Load projection mask
            if ~isfield(Metadata,'Mask_save')
                error('Mask_save not found in %s',Metadata_file);
            end
            par.Mask = Metadata.Mask_save;
            if ~isequal(size(par.Mask),[nodes,1])
                error('Incorrect mask dimensions in %s',Metadata_file);
            end
            fprintf('Loaded projection mask: %s\n',Metadata_file);

            %% Calculate no. sample points per DSO capture
            par.Nsym_per_point = par.Nnodes;
            T_per_point = par.Nsym_per_point/rs;                 % [ns]
            T_AWG_output = par.AWG_MaxDataLength/par.AWG_Rsampl; % [ns]
            par.Ndata_per_capture = floor(60/100*T_AWG_output/T_per_point);
            if par.Ndata_per_capture < 1
                error('AWG memory too small for one data point.');
            end
            fprintf('Data points per capture: %d\n', par.Ndata_per_capture);

            %% Initialization
            par.Nsym_init = round(par.initT*rs);
            Mask_tx = par.Mask/max(abs(par.Mask));

            %% Vpp loop
            for vpp = par.Vpp_range

                %% Set filename0
                filename0 = sprintf('%.1fGBd_%.1fmA_%.2fVpp_%dnodes',rs,ib,vpp,nodes);
                fprintf(['\nNext Run: ',filename0,'\n']);
                fprintf('-------------------------------------------\n');

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

                %% Initialize received reservoir-state matrix
                X_train = zeros(par.Nnodes,par.Ndata_tot);
                par.DummyLen = 0;

                %% Process dataset in AWG-sized blocks
                ii = 1; run = 0;
                while ii <= par.Ndata_tot

                    run = run + 1;

                    %% Define current data block
                    ii_end = min(ii+par.Ndata_per_capture-1,par.Ndata_tot);
                    indx_true = ii:ii_end;
                    Ntrue = numel(indx_true);

                    %% Pad final AWG block
                    Npad = par.Ndata_per_capture-Ntrue;
                    if Npad > 0
                        indx_pad = randi(par.Ndata_tot,1,Npad); % Pad AWG payload
                        indx_awg = [indx_true,indx_pad];
                    else
                        indx_awg = indx_true;
                    end

                    %% Apply mask to current block
                    X_block = X0(indx_awg,:);
                    U = reshape(X_block.',numel(X_block),1);
                    TxModSym = zeros(par.Ndata_per_capture*par.Nnodes,1);
                    for n = 1:par.Ndata_per_capture*par.Npre
                        ss = (n-1)*nodes+1;
                        TxModSym(ss:ss+nodes-1) = Mask_tx*U(n);
                    end
                    par.Nsym = length(TxModSym);

                    %% Add sync and initialization symbols
                    dummy = zeros(par.SyncDummySym*par.Sync_nSamp,1);
                    TxSym0 = [dummy; par.Sync_Ampl*par.Sync; zeros(par.Nsym_init,1); TxModSym];

                    %% Adjust pattern length
                    while true

                        while true
                            dummy = mean(TxSym0)*ones(par.DummyLen,1);
                            TxSym = [TxSym0;dummy];
                            par.TxSymLen = length(TxSym);
                            Nsampl_tmp = par.TxSymLen*par.nSamp_AWG;
                            if floor(Nsampl_tmp) == Nsampl_tmp
                                break;
                            elseif par.DummyLen > 5e4
                                error('Can''t find suitable value for "TxNsymLen"!');
                            else
                                par.DummyLen = par.DummyLen+1;
                            end
                        end

                        %% Prepare data for AWG
                        par.TxSymAvePow = mean(abs(TxSym).^2,1);
                        if par.nSamp > 1
                            if mod(par.RClen,2)~=0
                                par.RClen = par.RClen+1;
                            end
                            hTxfilter = rcosdesign(par.RCroff,par.RClen,par.nSamp,'sqrt');
                            TxSampl = upfirdn(TxSym,hTxfilter,par.nSamp,1);
                            ss = par.RClen*par.nSamp/2;
                            ee = ss + length(TxSym)*par.nSamp - 1;
                            TxSampl = TxSampl(ss:ee);
                            TxSampl = TxSampl-mean(TxSampl);
                        else
                            TxSampl = TxSym;
                        end
                        if par.nSamp_AWG ~= par.nSamp
                            TxSampl = resample(TxSampl, par.AWG_Rsampl*10, par.nSamp*rs*10);
                        end

                        Nsym_cut = 10;
                        ss = round(Nsym_cut*par.nSamp_AWG);
                        ee = length(TxSampl)-round(Nsym_cut*par.nSamp_AWG);
                        meanV = mean(TxSampl);
                        TxSampl(1:ss-1) = meanV;
                        TxSampl(ee+1:end) = meanV;
                        TxSamplLen = length(TxSampl);
                        DummySamplLen = 512*ceil(TxSamplLen/512)-TxSamplLen;
                        dummy = meanV*ones(DummySamplLen,1);
                        TxSig = [TxSampl;dummy];
                        TxSig = TxSig-mean(TxSig);
                        TxSig = sqrt(par.TxSymAvePow)*TxSig/sqrt(mean(abs(TxSig).^2,1));
                        TxSig = TxSig/max(abs(TxSig));
                        par.TxSigLen = length(TxSig);
                        if floor(par.TxSigLen/par.nSamp_AWG) == par.TxSigLen/par.nSamp_AWG
                            break;
                        else
                            par.DummyLen = par.DummyLen+1;
                        end

                    end

                    %% Block-specific DSO filename
                    DSO_filename = sprintf('DSOdata_%s_%d',filename0,ii_end);
                    fprintf('Run #%d: %d/%d samples loaded (idx: %d-%d, block size: %d)\n',run,ii_end,par.Ndata_tot,ii,ii_end,Ntrue);

                    %% Load previously captured block
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

                    %% Resample received signal
                    RxSig = resample(DSO_data, par.nSamp*rs, par.DSO_Rsampl/1e9);
                    RxSig = RxSig-mean(RxSig);
                    if plot_on == 1 && run == 1
                        RxSig_RF = RxSig;
                    end

                    %% Normalization
                    RxSigAvePow = mean(abs(RxSig).^2,1);
                    RxSig = sqrt(par.TxSymAvePow)*RxSig/sqrt(RxSigAvePow);

                    %% Use saved synchronization
                    fprintf('Loaded start point: %d, sync peak: %.4f\n', ...
                        par.start_frame(run),par.SyncPeakNorm_block(run));
                    if par.start_frame(run) + par.Nsym*par.nSamp - 1 > length(RxSig)
                        error('Insufficient DSO samples after saved start frame: %s',DSO_filename);
                    end

                    %% Downsampling
                    RxSig = RxSig(par.start_frame(run):end);
                    RxSig = RxSig(1:floor(length(RxSig)/par.nSamp)*par.nSamp);
                    RxSym = reshape(RxSig,par.nSamp,[]);
                    RxSym = mean(RxSym,1);          % downsampling
                    RxSym = RxSym - mean(RxSym);
                    RxSym = RxSym/max(abs(RxSym));  % normalization

                    %% Extract reservoir states
                    Nsym_payload = par.Nsym;
                    if length(RxSym) < Nsym_payload
                        error('Not enough received symbols in %s.',DSO_filename);
                    end
                    SpikeOut = RxSym(1:Nsym_payload);
                    SpikeOut = reshape(SpikeOut,par.Nnodes,par.Ndata_per_capture);
                    SpikeOut = SpikeOut(:,1:Ntrue);
                    X_train(:,indx_true) = SpikeOut;

                    %% Plot live diagnostics
                    if plot_on == 1 && run == 1

                        set(f1,'Visible',1); figure(f1);

                        %% RF spectrum
                        subplot(2,3,1); cla; hold on; box on; grid on;
                        [p1,ff1] = periodogram(DSO_data,[],[],par.DSO_Rsampl,'onesided','power');
                        [p2,ff2] = periodogram(RxSig_RF,[],[],par.nSamp*rs*1e9,'onesided','power');
                        Pow1 = 10*log10(max(p1,realmin));
                        Pow2 = 10*log10(max(p2,realmin));
                        plot(ff1/1e9,Pow1-max(Pow1));
                        plot(ff2/1e9,Pow2-max(Pow2));
                        xlim([0,2*rs]); ylim([-100 10]);
                        legend('DSO','Rx resampled','Location','best');
                        xlabel('Frequency (GHz)');
                        ylabel('Normalized RF power (dB)');
                        title('Received RF spectrum');
                        set(gca,'fontsize',12);

                        %% Received waveform + distribution
                        subplot(2,3,4); 
                        cla; hold on; box on; grid on;

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

                        %% Detected spikes - raster plot
                        subplot(2,3,[2 3 5 6]);
                        cla; hold on; box on; grid off;
                        ylabel('Dataset sample no.'); xlabel('Symbol no.');
                        spikes_plot = SpikeOut(:,1:min(50,size(SpikeOut,2))) > 0;
                        spikes_plot = (1:size(spikes_plot,2)).*spikes_plot;
                        spikes_plot(spikes_plot == 0) = NaN;
                        plot(spikes_plot,'.g','MarkerSize',3);
                        xlim([0.5,par.Nnodes+0.5]);
                        ylim([0.5,min(50.5,size(spikes_plot,2)+0.5)]);
                        set(gca,'fontsize',12,'color','k');

                        drawnow limitrate;

                    end

                    %% Advance to next block
                    ii = ii_end+1;

                end % while ii <= par.Ndata_tot

                %% Training: MMSE (ridge regression)
                X_all = X_train;    % [Nnodes x Ndata_tot]
                T_all = T0;         % [Ndata_tot x Npost]

                %% Define training and test indices
                idx_train = 1:par.Ndata_train;
                idx_test  = par.Ndata_train+1:par.Ndata_tot;

                %% Split training set into fit and validation sets
                Nval = round(par.val_frac*length(idx_train));
                Nval = max(1,min(Nval,length(idx_train)-1));
                idx_fit = idx_train(1:end-Nval);
                idx_val = idx_train(end-Nval+1:end);
                fprintf('\nMMSE split: Nfit = %d, Nval = %d, Ntest = %d\n', ...
                    length(idx_fit),length(idx_val),length(idx_test));

                fprintf('mean: %f, std: %f\n', mean(X_all(:,idx_fit),'all'), ...
                    std(X_all(:,idx_fit),1,'all'));

                bestValNMSE = Inf;
                bestLambda = NaN;
                X_aug = [X_all;ones(1,size(X_all,2))];

                for lambda = par.lambda_range

                    R = X_aug(:,idx_fit)*X_aug(:,idx_fit).' + lambda*eye(size(X_aug,1));
                    R(end,end) = R(end,end)-lambda;
                    W_aug = (T_all(idx_fit,:).'*X_aug(:,idx_fit).')/R;
                    W_tmp = W_aug(:,1:end-1);
                    bias_tmp = W_aug(:,end);

                    %% Validation inference
                    Y_val = W_tmp*X_all(:,idx_val) + bias_tmp;
                    T_val = T_all(idx_val,:).';
                    err_val = Y_val-T_val;

                    %% Validation NMSE for each prediction horizon
                    MSE_val = mean(err_val.^2,2).';
                    NMSE_val = MSE_val./par.VarT;
                    NMSE_val_mean = mean(NMSE_val);

                    fprintf('lambda = %.1e: NMSE mean = %.5f, i=1 = %.5f, i=%d = %.5f\n', ...
                        lambda,NMSE_val_mean,NMSE_val(1),par.Npost,NMSE_val(end));

                    %% Select lambda giving minimum validation NMSE
                    if NMSE_val_mean < bestValNMSE
                        bestValNMSE = NMSE_val_mean;
                        bestLambda = lambda;
                    end

                end

                %% Store selected lambda
                par.bestLambda = bestLambda;
                fprintf('Selected lambda = %.1e: Validation NMSE = %.5f\n', ...
                    par.bestLambda,bestValNMSE);

                %% Refit readout using complete training set
                lambda = par.bestLambda;
                R = X_aug(:,idx_train)*X_aug(:,idx_train).' + lambda*eye(size(X_aug,1));
                R(end,end) = R(end,end)-lambda;
                W_aug = (T_all(idx_train,:).'*X_aug(:,idx_train).')/R;
                par.W = W_aug(:,1:end-1);
                par.bias = W_aug(:,end);

                %% Final inference
                Y = (par.W*X_all + par.bias).';      % [Ndata_tot x Npost]
                Y_test = Y(idx_test,:);
                T_test = T_all(idx_test,:);
                err_test = Y_test-T_test;

                %% NMSE per DSO block
                par.NMSE_block = NaN(1,par.Nblocks);
                for run = 1:par.Nblocks
                    idx_block = (run-1)*par.Ndata_per_capture+1:min(run*par.Ndata_per_capture,par.Ndata_tot);
                    err_block = Y(idx_block,:) - T_all(idx_block,:);
                    MSE_block = mean(err_block.^2,1);          % [1 x Npost]
                    NMSE_block = MSE_block./par.VarT;          % [1 x Npost]
                    par.NMSE_block(run) = mean(NMSE_block);    % scalar mean over horizons
                    fprintf('DSO block #%d: idx %d-%d, NMSE = %.5f\n', ...
                        run,idx_block(1),idx_block(end),par.NMSE_block(run));
                end

                %% Final test metrics
                MSE_test = mean(err_test.^2,1);
                par.NMSE = MSE_test./par.VarT;
                fprintf('Test NMSE: mean = %.5f, i=1 = %.5f, i=%d = %.5f\n', ...
                    mean(par.NMSE),par.NMSE(1),par.Npost,par.NMSE(end));

                if plot_on == 1
                    set(f2,'Visible',1); figure(f2); clf;

                    %% Plot predicted time series
                    subplot(2,2,[1,2]); hold on; grid on; box on;
                    Ndata_plot = min(1e3,par.Ndata_test);
                    idx_plot = par.Ndata_tot-Ndata_plot+1:par.Ndata_tot;
                    Nshift = par.Npost-1;
                    xax = 1:Ndata_plot;
                    block_plot = ceil(idx_plot/par.Ndata_per_capture);
                    plot(xax,(Y(idx_plot,1)+1)/2,'b');
                    plot(xax(1:end-Nshift)+Nshift, (Y(idx_plot(1:end-Nshift),end)+1)/2,'r');
                    plot(xax,(T0(idx_plot,1)+1)/2,'k');
                    block_change = find(diff(block_plot) ~= 0);
                    for jj = block_change
                        xline(jj+0.5,'k--');
                    end
                    ylabel('Normalized amplitude'); ylim([0,1]);
                    xlabel('Data points'); xlim([1,Ndata_plot]);
                    legend('i_{post} = 1',sprintf('i_{post} = %d',par.Npost),'target','location','ne');
                    set(gca,'fontsize',12);

                    %% NMSE vs prediction interval
                    subplot(2,2,[3,4]);
                    hold on; grid on; box on;
                    plot(1:par.Npost,par.NMSE,'k-o','MarkerSize',4);
                    ylabel('NMSE');
                    xlabel('Future-current interval, {\iti}_{post}');
                    xlim([1,par.Npost]);
                    set(gca,'fontsize',12);

                    drawnow limitrate;
                end

                %% Save results
                Results_file = [results_folder,'Results_',filename0,'.mat'];
                save(Results_file,'par','X_train','Y_test','T_test', ...
                    'idx_train','idx_test','-v7.3');
                if plot_on == 1
                    exportgraphics(f2,[results_folder,'Prediction_',filename0,'.png'], ...
                        'Resolution',200);
                end

            end     % vpp
        end         % nodes
    end             % rs
end                 % ib

fprintf('\nTotal run-time: %.1f mins\n\n',toc/60);
diary off;
