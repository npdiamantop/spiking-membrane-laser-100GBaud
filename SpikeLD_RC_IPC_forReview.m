%% Reservoir Computing using Spiking-LD experiment code
% N.P.Diamantopoulos, Device Technology Labs, NTT, Inc., Japan
% last update: 2026-Oct-26
% -------------------------------------------------------------------------
clear; tic; clc; close all; rng(101);
fprintf('----------------------------------------------\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plot_on         = 1;                % {0:OFF,1:ON}

par.Ltot        = 1e4;              % total input length
par.L           = 5e3;              % signal length used for IPC calculations
par.D           = 20;               % max. delay for memory evaluation
par.M           = 7;                % max. order

par.Ib_range    = [4.3];            % [mA] (positive polarity only)
par.Rs_range    = [100];            % [GBaud]
par.Vpp_range   = [1.10];           % [Vpp]

par.RCroff      = 0.1;              % roll-off of raised-cosine filter
par.nSamp       = 8;                % [Sa/sym]
par.RClen       = 2;                % [symbols]

par.AWG_Rsampl_range = [100,130];   % [GHz] {min,max}
par.DSO_Rsampl  = 160e9;            % [Sa/s] DSO sampling rate

par.Mask_len    = [20];             % RC no. nodes (random mask is applied)
par.initT       = 2;                % [ns] null time before sending data points

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Set folders
script_folder = fileparts(mfilename('fullpath'));
metadata_folder = [fullfile(script_folder,'IPC_metadata'),filesep];
dso_folder      = metadata_folder;
results_folder     = [fullfile(script_folder,'IPC_results'),filesep];
if ~isfolder(metadata_folder)
    error('Missing IPC input folder: %s',metadata_folder);
end
if ~isfolder(results_folder)
    mkdir(results_folder);
end

%% Initialize figures
if plot_on == 1
    f1 = figure('Name','IPC diagnostics','Visible',0,'color','w');
    f2 = figure('Name','IPC vs nodes','Visible',0,'color','w');
end

%% Initialize
par.Ib_range = abs(par.Ib_range);       % ensure positive
par.Vpp_range = sort(par.Vpp_range);    % lowest Vpp first
par.nSamp = round(par.nSamp);           % ensure integrer

%% Input data
par.Linit = par.Ltot - par.L;
filename_meta = sprintf('%.1fGBd_%.1fmA_%.2fVpp_%dnodes', ...
    par.Rs_range(1),par.Ib_range(1),par.Vpp_range(1),par.Mask_len(1));
S = load([metadata_folder,'IPCmeta_',filename_meta,'.mat'],'u_save');
par.u = S.u_save(:);
assert(numel(par.u) == par.Ltot,'Incorrect IPC input length.');

%% Build delayed input matrix U
U = zeros(par.L,par.D);
for h = 1:par.D
    U(:,h) = par.u(par.Linit-h+(1:par.L));
end

%% Precompute Legendre polynomials
par.P = cell(par.M,1);
for m = 1:par.M
    par.P{m} = legendreP_matrix(m,U);  % [L×D]
end

%% Set capacity threshold
par.Cth = 1e2/par.L; % correlations threshold
fprintf('Data processing begins...\n');

%% Main loop
for ib = par.Ib_range

    for rs = par.Rs_range

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
        nSamp_AWG = par.AWG_Rsampl/rs;
        fprintf('AWG sampling rate was set to %f GHz\n',par.AWG_Rsampl);

        for nodes_idx = 1:length(par.Mask_len)
            nodes = par.Mask_len(nodes_idx);

            %% Load original IPC metadata
            assert(isscalar(par.Vpp_range), ...
                'This IPC reviewer example expects a single Vpp.');
            filename_meta = sprintf('%.1fGBd_%.1fmA_%.2fVpp_%dnodes', ...
                rs,ib,par.Vpp_range(1),nodes);
            Metadata_file = [metadata_folder,'IPCmeta_',filename_meta,'.mat'];
            if ~isfile(Metadata_file)
                error('Missing IPC metadata: %s',Metadata_file);
            end
            S = load(Metadata_file);
            assert(isequal(S.u_save(:),par.u(:)), ...
                'IPC input sequence does not match the metadata.');
            par.Mask = S.Mask_save;
            par.Sync = S.Sync_save;
            par.start_frame = S.start_frame_save;
            par.SyncRep      = S.SyncMeta.SyncRep;
            par.Sync_nSamp   = S.SyncMeta.Sync_nSamp;
            par.SyncLen      = S.SyncMeta.SyncLen;
            par.Sync_Rs      = S.SyncMeta.Sync_Rs;
            par.Sync_Ampl    = S.SyncMeta.Sync_Ampl;
            par.SyncDummySym = S.SyncMeta.SyncDummySym;
            assert(isequal(size(par.Mask),[nodes,1]), ...
                'Incorrect IPC mask dimensions.');
            assert(numel(par.Sync) == par.SyncRep*par.SyncLen*par.Sync_nSamp, ...
                'Incorrect IPC sync pattern length.');

            %% Apply projection mask
            TxModSym = reshape(par.Mask*par.u.',[],1);

            %% Initialize data
            TxModSym = TxModSym - mean(TxModSym);
            TxModSym = TxModSym/max(abs(TxModSym)); % Normalize
            par.Nsym = length(TxModSym);

            %% Define time series symbols
            par.Nsym_init = round(par.initT*rs); % no. initialization symbols
            dummy = zeros(par.SyncDummySym*par.Sync_nSamp,1);
            TxSym0 = [dummy;par.Sync_Ampl*par.Sync;zeros(par.Nsym_init,1);TxModSym];

            %% Adjust pattern length
            par.DummyLen = 0;
            while true

                while true
                    dummy = mean(TxSym0)*ones(par.DummyLen,1);
                    TxSym = [TxSym0;dummy];
                    par.TxSymLen = length(TxSym);
                    Nsampl_tmp = par.TxSymLen*nSamp_AWG;
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
                if nSamp_AWG ~= par.nSamp
                    TxSampl = resample(TxSampl,par.AWG_Rsampl*10,par.nSamp*rs*10);
                end
                Nsym_cut = 10;
                ss = round(Nsym_cut*nSamp_AWG);
                ee = length(TxSampl) - round(Nsym_cut*nSamp_AWG);
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
                if  floor(par.TxSigLen/nSamp_AWG) == par.TxSigLen/nSamp_AWG % check if integer
                    break;
                else
                    par.DummyLen = par.DummyLen + 1;
                end
            end

            fprintf('Total samples to AWG: %d\n',par.TxSigLen);

            %% Data processing loop ---
            if plot_on == 1; set(f1,'Visible',1); end

            for vpp = par.Vpp_range

                %% Select IPC diagnostic figure
                if plot_on == 1; figure(f1); end

                %% Setup folders
                filename0 = sprintf('%.1fGBd_%.1fmA_%.2fVpp_%dnodes',rs,ib,vpp,nodes);
                fprintf(['Processing data for: ',filename0,'\n']);
                if ~exist(results_folder,'dir'); mkdir(results_folder); end

                %% Load DSO data
                DSO_filename = sprintf('DSOdata_%s',filename0);
                DSO_filepath = [dso_folder,DSO_filename,'.bin'];
                fidDSO = fopen(DSO_filepath,'rb');
                if fidDSO < 0
                    error('Could not open DSO file: %s',DSO_filepath);
                end
                DSO_data = fread(fidDSO,inf,'single');
                fclose(fidDSO);

                RxSig = resample(DSO_data,par.nSamp*rs,par.DSO_Rsampl/1e9);
                RxSig = RxSig-mean(RxSig);

                %% Plot Signal RF Spectrum
                if plot_on == 1
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
                    set(gca,'fontsize',12); drawnow;
                end

                %% Normalization
                RxSigAvePow = mean(abs(RxSig).^2,1);
                RxSig = sqrt(par.TxSymAvePow)*RxSig/sqrt(RxSigAvePow);

                %% Use saved time synchronization
                if ~isscalar(par.start_frame) || ...
                        ~isfinite(par.start_frame) || ...
                        par.start_frame < 1 || ...
                        par.start_frame ~= round(par.start_frame) || ...
                        par.start_frame > length(RxSig)
                    error('Invalid saved start-frame position.');
                end
                fprintf('Loaded start point: %d\n',par.start_frame);
                if par.start_frame + par.Nsym*par.nSamp - 1 > length(RxSig)
                    error('Insufficient DSO samples after saved start frame.');
                end

                %% Downsampling & Thresholding
                RxSig = RxSig(par.start_frame:end);
                RxSig = RxSig(1:floor(length(RxSig)/par.nSamp)*par.nSamp);
                RxSym = reshape(RxSig,par.nSamp,[]);
                RxSym = mean(RxSym,1);          % downsampling
                RxSym = RxSym - mean(RxSym);
                RxSym = RxSym/max(abs(RxSym));  % normalization
                SpikeOut = RxSym(1:par.Nsym);
                SpikeOut = reshape(SpikeOut,nodes,[]);

                %% Plot received waveform
                if plot_on == 1
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
                if plot_on == 1
                    subplot(2,3,[2,5]); cla; hold on; box on; grid off;
                    ylabel('Dataset sample no.'); xlabel('Symbol no.');
                    spikes_plot = SpikeOut(:,1:min(50,size(SpikeOut,2))) > 0;
                    spikes_plot = (1:size(spikes_plot,2)).*spikes_plot;
                    spikes_plot(spikes_plot == 0) = NaN;
                    plot(spikes_plot,'.g','MarkerSize',3);
                    xlim([0.5,nodes+0.5]);
                    ylim([0.5,min(50.5,size(spikes_plot,2)+0.5)]);
                    set(gca,'fontsize',12,'color','k'); drawnow;
                end

                %% IPC calculation
                dd = 5; % Nominal zero-lag target index (h = dd)

                par.MC = zeros(1,par.M);
                par.C = cell(par.M,1);
                par.IPC = 0;
                X = SpikeOut.';  % [Ltot x N]
                X = 2*X(par.Linit+1-dd:end-dd,:)-1; % [0,1] -> [-1,1]

                %% --- Degree 1 ---
                Y = par.P{1}; % P1, [D]
                [par.MC(1),par.C{1}] = DambreCorrelationMatrix(X,Y,par.Cth);

                %% Print significant first-order input delays (nominal alignment)
                C1_idx = find(par.C{1} >= par.Cth);
                if isempty(C1_idx)
                    fprintf('C1 above cutoff %.4f: no delays\n',par.Cth);
                else
                    fprintf('C1 above %.4f: h=[%s], lag=[%s] input steps (%.0f ps/step), C=[%s]\n', ...
                        par.Cth,strtrim(sprintf('%d ',C1_idx)), ...
                        strtrim(sprintf('%d ',C1_idx-dd)),1000*nodes/rs, ...
                        strtrim(sprintf('%.3f ',par.C{1}(C1_idx))));
                end

                %% --- Degree 2 ---
                Ylist = zeros(par.L,par.D*(par.D+1)/factorial(2)); % [D(D+1)/2!]
                Ylist(:,1:par.D) = [par.P{2}]; % P2, [D]
                indx = par.D + 1;
                % P1*P1 (i>j), [D(D-1)/2]
                for i = 1:par.D
                    for j = 1:(i-1)
                        Ylist(:,indx) = par.P{1}(:,i).*par.P{1}(:,j);
                        indx = indx + 1;
                    end
                end
                [par.MC(2),par.C{2}] = DambreCorrelationMatrix(X,Ylist,par.Cth);

                %% --- Degree 3 ---
                Ylist = zeros(par.L,prod(par.D:par.D+2)/factorial(3)); % [D(D+1)(D+2)/3!]
                Ylist(:,1:par.D) = [par.P{3}]; % P3, [D]
                indx = par.D + 1;
                % P2*P1, [D(D-1)]
                for i = 1:par.D
                    for j = 1:par.D
                        if i ~= j
                            Ylist(:,indx) = par.P{2}(:,i).*par.P{1}(:,j);
                            indx = indx + 1;
                        end
                    end
                end
                % P1*P1*P1 (i>j>k), [D(D-1)(D-2)/6]
                for i = 1:par.D
                    for j = 1:(i-1)
                        for k = 1:(j-1)
                            Ylist(:,indx) = par.P{1}(:,i).*par.P{1}(:,j).*par.P{1}(:,k);
                            indx = indx + 1;
                        end
                    end
                end
                [par.MC(3),par.C{3}] = DambreCorrelationMatrix(X,Ylist,par.Cth);

                %% --- Degree 4 ---
                Ylist = zeros(par.L,prod(par.D:par.D+3)/factorial(4)); % [D(D+1)(D+2)(D+3)/4!]
                Ylist(:,1:par.D) = [par.P{4}]; % P4, [D]
                indx = par.D + 1;
                % P3*P1, [D(D-1)]
                for i = 1:par.D
                    for j = 1:par.D
                        if i ~= j
                            Ylist(:,indx) = par.P{3}(:,i).*par.P{1}(:,j);
                            indx = indx + 1;
                        end
                    end
                end
                % P2*P2 (i>j), [D(D-1)/2]
                for i = 1:par.D
                    for j = 1:(i-1)
                        Ylist(:,indx) = par.P{2}(:,i).*par.P{2}(:,j);
                        indx = indx + 1;
                    end
                end
                % P2*P1*P1 (j>k), [D(D-1)(D-2)/2]
                for i = 1:par.D
                    for j = 1:par.D
                        for k = 1:(j-1)
                            if i~=j && i~=k
                                Ylist(:,indx) = par.P{2}(:,i).*par.P{1}(:,j).*par.P{1}(:,k);
                                indx = indx + 1;
                            end
                        end
                    end
                end
                % P1*P1*P1*P1 (i>j>k>l), [D(D-1)(D-2)(D-3)/24]
                for i = 1:par.D
                    for j = 1:(i-1)
                        for k = 1:(j-1)
                            for l = 1:(k-1)
                                Ylist(:,indx) = par.P{1}(:,i).*par.P{1}(:,j).*par.P{1}(:,k).*par.P{1}(:,l);
                                indx = indx + 1;
                            end
                        end
                    end
                end
                [par.MC(4),par.C{4}] = DambreCorrelationMatrix(X,Ylist,par.Cth);

                %% --- Degree 5 ---
                Ylist = zeros(par.L,prod(par.D:par.D+4)/factorial(5)); % [D(D+1)(D+2)(D+3)(D+4)/5!]
                Ylist(:,1:par.D) = [par.P{5}]; % P5, [D]
                indx = par.D + 1;
                % P4*P1, [D(D-1)]
                for i = 1:par.D
                    for j = 1:par.D
                        if i ~= j
                            Ylist(:,indx) = par.P{4}(:,i).*par.P{1}(:,j);
                            indx = indx + 1;
                        end
                    end
                end
                % P3*P2, [D(D-1)]
                for i = 1:par.D
                    for j = 1:par.D
                        if i ~= j
                            Ylist(:,indx) = par.P{3}(:,i).*par.P{2}(:,j);
                            indx = indx + 1;
                        end
                    end
                end
                % P3*P1*P1 (j>k), [D(D-1)(D-2)/2]
                for i = 1:par.D
                    for j = 1:par.D
                        for k = 1:(j-1)
                            if i~=j && i~=k
                                Ylist(:,indx) = par.P{3}(:,i).*par.P{1}(:,j).*par.P{1}(:,k);
                                indx = indx + 1;
                            end
                        end
                    end
                end
                % P2*P2*P1 (i>j), [D(D-1)(D-2)/2]
                for i = 1:par.D
                    for j = 1:(i-1)
                        for k = 1:par.D
                            if k~=i && k~=j
                                Ylist(:,indx) = par.P{2}(:,i).*par.P{2}(:,j).*par.P{1}(:,k);
                                indx = indx + 1;
                            end
                        end
                    end
                end
                % P2*P1*P1*P1 (j>k>l), [D(D-1)(D-2)(D-3)/6]
                for i = 1:par.D
                    for j = 1:par.D
                        for k = 1:(j-1)
                            for l = 1:(k-1)
                                if i~=j && i~=k && i~=l
                                    Ylist(:,indx) = par.P{2}(:,i).*par.P{1}(:,j).*par.P{1}(:,k).*par.P{1}(:,l);
                                    indx = indx + 1;
                                end
                            end
                        end
                    end
                end
                % P1*P1*P1*P1*P1 (i>j>k>l>m), [D(D-1)(D-2)(D-3)(D-4)/120]
                for i = 1:par.D
                    for j = 1:(i-1)
                        for k = 1:(j-1)
                            for l = 1:(k-1)
                                for m = 1:(l-1)
                                    Ylist(:,indx) = par.P{1}(:,i).*par.P{1}(:,j).*par.P{1}(:,k).*par.P{1}(:,l).*par.P{1}(:,m);
                                    indx = indx + 1;
                                end
                            end
                        end
                    end
                end
                [par.MC(5),par.C{5}] = DambreCorrelationMatrix(X,Ylist,par.Cth);

                %% --- Degree 6 ---
                %Ylist = zeros(par.L,prod(par.D:par.D+5)/factorial(6)); % [D(D+1)(D+2)(D+3)(D+4)(D+5)/6!]
                Ylist = zeros(par.L,par.D + 2*par.D*(par.D-1) + par.D*(par.D-1)/2); % restricted basis
                Ylist(:,1:par.D) = [par.P{6}]; % P6, [D]
                indx = par.D + 1;
                % P5*P1, [D(D-1)]
                for i = 1:par.D
                    for j = 1:par.D
                        if i ~= j
                            Ylist(:,indx) = par.P{5}(:,i).*par.P{1}(:,j);
                            indx = indx + 1;
                        end
                    end
                end
                % P4*P2, [D(D-1)]
                for i = 1:par.D
                    for j = 1:par.D
                        if i ~= j
                            Ylist(:,indx) = par.P{4}(:,i).*par.P{2}(:,j);
                            indx = indx + 1;
                        end
                    end
                end
                % P3*P3 (i>j), [D(D-1)/2]
                for i = 1:par.D
                    for j = 1:(i-1)
                        Ylist(:,indx) = par.P{3}(:,i).*par.P{3}(:,j);
                        indx = indx + 1;
                    end
                end
                % Higher-dimensional interactions intentionally omitted
                assert(indx-1 == size(Ylist,2),'Degree-6 target count mismatch.');
                [par.MC(6),par.C{6}] = DambreCorrelationMatrix(X,Ylist,par.Cth);

                %% --- Degree 7 ---
                %Ylist = zeros(par.L,3*par.D*(par.D-1)); % [D(D+1)(D+2)(D+3)(D+4)(D+5)(D+6)/7!]
                Ylist = zeros(par.L,par.D + 3*par.D*(par.D-1)); % restricted basis
                Ylist(:,1:par.D) = [par.P{7}]; % P7, [D]
                indx = par.D + 1;
                % P6*P1, [D(D-1)]
                for i = 1:par.D
                    for j = 1:par.D
                        if i ~= j
                            Ylist(:,indx) = par.P{6}(:,i).*par.P{1}(:,j);
                            indx = indx + 1;
                        end
                    end
                end
                % P5*P2, [D(D-1)]
                for i = 1:par.D
                    for j = 1:par.D
                        if i ~= j
                            Ylist(:,indx) = par.P{5}(:,i).*par.P{2}(:,j);
                            indx = indx + 1;
                        end
                    end
                end
                % P4*P3, [D(D-1)]
                for i = 1:par.D
                    for j = 1:par.D
                        if i ~= j
                            Ylist(:,indx) = par.P{4}(:,i).*par.P{3}(:,j);
                            indx = indx + 1;
                        end
                    end
                end
                % Higher-dimensional interactions intentionally omitted
                assert(indx-1 == size(Ylist,2),'Degree-7 target count mismatch.');
                [par.MC(7),par.C{7}] = DambreCorrelationMatrix(X,Ylist,par.Cth);

                %% Plot Correlations
                if plot_on == 1
                    subplot(2,3,3); cla; hold on; box on; grid on;
                    xlabel('Delay steps'); ylabel('Correlations');
                    set(gca,'fontsize',12); xlim([1,par.D]);
                    for m = 1:par.M
                        %plot(par.C{m}/max(abs(par.C{m})));
                        plot(par.C{m});
                    end
                    legend('1st-order','2nd-order','3rd-order','4th-order','5th-order','6th-order','7th-order');
                end

                %% Print results
                fprintf('MC results by degree:\n');
                for m = 1:par.M
                    fprintf('  MC%d = %.4f\n',m,par.MC(m));
                end
                par.IPC = sum(par.MC);
                fprintf('IPC = %.4f\n',par.IPC);

                %% Plot MCs
                if plot_on == 1
                    subplot(2,3,6);
                    labels = ["1st","2nd","3rd","4th","5th","6th","7th"];
                    d = donutchart(par.MC,labels);
                    d.Labels = d.Names + '\newline(' + round(d.Data,2) + ')';
                    set(gca,'fontsize',12);
                end

                %% Save results
                Results_file = [results_folder,'Results_',filename0,'.mat'];
                save(Results_file,'par','SpikeOut');
                if plot_on == 1
                    exportgraphics(f1,[results_folder,'IPC_',filename0,'.png'], 'Resolution',200);
                end

            end
        end
    end
end

%% Results summary ---
for rs = par.Rs_range
    for ib = par.Ib_range
        for vpp = par.Vpp_range

            %% Load results
            MC_all = zeros(length(par.Mask_len),par.M);
            for nodes_idx = 1:length(par.Mask_len)
                nodes = par.Mask_len(nodes_idx);
                filename0 = sprintf('%.1fGBd_%.1fmA_%.2fVpp_%dnodes',rs,ib,vpp,nodes);
                Sres = load([results_folder,'Results_',filename0,'.mat'],'par');
                MC_all(nodes_idx,:) = Sres.par.MC;
            end

            %% Plot results
            if plot_on == 1
                set(f2,'Visible',1); figure(f2); clf(f2);
                hold on; box on; grid on;
                bar(par.Mask_len,MC_all,'stacked');
                xlabel('Number of nodes');
                xlim([par.Mask_len(1)-0.5,par.Mask_len(1)+0.5]);
                ylabel("Memory capacity");
                legend("1st","2nd","3rd","4th","5th","6th","7th",'location','northoutside',"Orientation","horizontal");
                set(gca,'fontsize',12); drawnow;

                filename0 = sprintf('%.1fGBd_%.1fmA_%.2fVpp',rs,ib,vpp);
                exportgraphics(f2,[results_folder,'IPCsummary_',filename0,'.png'], 'Resolution',200);
            end

        end
    end
end

fprintf('\nTotal run-time: %.1f mins\n\n',toc/60);

%--------------------------------------------------------------------------
function P = legendreP_matrix(d,X)
%% Vectorized Legendre polynomial evaluation for matrix X
% d: degree
% X: [L x D] L: data length, D: max. delay
% P: [L x D]
%--------------------------------------------------------------------------
P = legendre(d,X,'norm');
P = squeeze(P(1,:,:));
end

%--------------------------------------------------------------------------
function [MC,C] = DambreCorrelationMatrix(X,Y,Cth)
%% Dambre capacities (correlations) between RC states X and targets Y
% J. Dambre, et al., Sci Rep 2, 514 (2012)
%   X:  [L x N]  L: data length, N: no. nodes
%   Y:  [L x T]  T: no. targets
%   C:  [1 x T]
%   MC: [1 x 1]
%--------------------------------------------------------------------------

X = X - mean(X,1);
Y = Y - mean(Y,1);
X = X./vecnorm(X);
Y = Y./vecnorm(Y);

W = pinv(X.'*X)*X.'*Y; % [NxT] % Least-Squares solution
Y_hat = X*W;           % [LxT] % linear readout

C = sum(Y.*Y_hat,1).^2./(sum(Y.^2,1).*sum(Y_hat.^2,1));
C(~isfinite(C)) = 0;

C(C<Cth) = 0;
MC = sum(C);

end
