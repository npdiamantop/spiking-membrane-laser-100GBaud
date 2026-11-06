**Supplementary Code for manuscript**:
N.-P. Diamantopoulos et al., “Ultrafast spiking membrane laser neuromorphic hardware processing 100 GBaud input streams,” submitted to Nature Communications

Subject to NTT SOFTWARE LICENSE AGREEMENT.

The MATLAB scripts analyze experimentally measured responses of a silicon-integrated membrane-feedback laser to 100-GBaud electrical input streams. This release covers information-processing capacity (IPC), MNIST handwritten-digit classification, and Mackey–Glass time-series prediction. The classification and time-series prediction scripts are configured for reproducing the MNIST and Mackey–Glass results, respectively. The corresponding Two-Moons and Santa Fe experimental recordings and metadata are not included in the reviewer package.

### Repository contents

Main script files:

1. SpikeLD_RC_IPC_forReview.m
2. SpikeLD_RC_Classification_forReview.m
3. SpikeLD_RC_TimeSeriesPrediction_forReview.m

Dataset preparation and loading utilities:

- tools/MNIST_dataset_load.m
- tools/MackeyGlass_series.m
- tools/SantaFe_series.m
- tools/TwoMoons_dataset_generation.m

The MNIST dataset-loading utility requires the original MNIST IDX files, while the Santa Fe utility requires the original SantaFeA.dat and SantaFeA2.dat files. These utilities generate reference datasets and do not reproduce the experimentally measured reservoir responses. The original experimental datasets and corresponding measurement metadata should be used when reproducing the reported results.

The classification and time-series prediction scripts also contain task-specific code for Two-Moons and Santa Fe, respectively, for reference purposes. The corresponding experimental recordings and metadata are not provided, and these configurations are not supported for end-to-end reproduction in this release.

### Experimental data traces and metadata access

Experimental oscilloscope recordings, measurement metadata, datasets, and MMSE-initialized readout parameters are not included in this repository. For manuscript verification, these files are made available separately to authorized Nature Communications editors and reviewers through restricted, time-limited NTT R&D Box access, in accordance with institutional intellectual-property and information-security requirements. Access details are provided through the editorial correspondence.

### Licensing

The NTT-owned MATLAB source code is provided under the NTT Software License Agreement for Evaluation included with this package.

The loadMNISTImages and loadMNISTLabels routines incorporated with modifications in tools/MNIST_dataset_load.m originate from the Stanford UFLDL software distribution and are subject to the MIT License. Their copyright attribution and full license text are provided in THIRD_PARTY_NOTICES.txt.

The NTT Software License Agreement governs the non-OSS components, while the applicable MIT License governs the identified third-party components.

Public availability on GitHub does not constitute permission to use, modify, redistribute, or commercially exploit the NTT-owned software beyond the rights granted by NTT.

### Software requirements

- MATLAB R2023b or later is recommended for the scripts' plotting functionality ("donutchart" in the IPC script).
- Signal Processing Toolbox for resampling, raised-cosine filter design, and spectrum analysis ("resample", "upfirdn", "rcosdesign", "periodogram").
- Image Processing Toolbox is needed only if the optional MNIST utility is modified to resize images; no resizing is required for the supplied 28 × 28 dataset.
- A 64-bit MATLAB installation and sufficient RAM/disk space for the full MNIST recordings and reservoir-state matrices are recommended.

The scripts process existing measured data; they do not control the signal generator (AWG), oscilloscope, or laser hardware. No additional third-party MATLAB function files are required by the three main scripts.

### Tested environment

- Operating system: Microsoft Windows 11 (64-bit).
- MATLAB release tested: R2026a. MATLAB R2023b or later is recommended, but other releases have not been independently verified for this package.
- Required toolbox: Signal Processing Toolbox.
- Non-standard hardware: None required for offline MATLAB analysis. The experimental data were acquired using specialized photonic hardware, including the membrane feedback laser, arbitrary waveform generator, photodetector and oscilloscope.

### Getting started

1. Obtain the corresponding experimental-data package using the separately approved access procedure.
2. Extract the data into the folders shown below, alongside the MATLAB scripts. Keep the original filenames and capitalization.
3. Open MATLAB, change to the repository folder, and run the desired main script.
4. Examine the Command Window output and the automatically created results folder.

The scripts determine input paths relative to their own ".m" file locations; you do not need to edit the base paths when the folder structure is preserved.

### Installation and execution time

No installation or compilation is required beyond a working MATLAB installation with the required toolbox. Download or clone the repository, place the corresponding experimental datasets in the specified directories, and run the MATLAB scripts.

- **Typical setup time:** Approximately 2–5 minutes on a standard desktop computer with MATLAB already installed, excluding MATLAB installation and experimental-data downloads (estimated).
- **Expected execution time:** Approximately a few minutes to a few hours, depending on the analysis, experimental dataset size and available computing resources (indicative estimate; not a controlled desktop benchmark). A standalone demonstration dataset is not included; the corresponding experimental data are provided separately to authorized reviewers.

### Expected directory structure

```text
SupplementaryCode/
|
|-- README.md
|-- NTT_SOFTWARE_LICENSE_AGREEMENT.pdf
|-- THIRD_PARTY_NOTICES.txt
|
|-- SpikeLD_RC_IPC_forReview.m
|-- SpikeLD_RC_Classification_forReview.m
|-- SpikeLD_RC_TimeSeriesPrediction_forReview.m
|
|-- tools/
|   |-- MNIST_dataset_load.m
|   |-- MackeyGlass_series.m
|   |-- SantaFe_series.m
|   `-- TwoMoons_dataset_generation.m
|
|-- IPC_metadata/                [Provided separately]
|   |-- IPCmeta_100.0GBd_4.3mA_1.10Vpp_20nodes.mat
|   `-- DSOdata_100.0GBd_4.3mA_1.10Vpp_20nodes.bin
|
|-- MNIST_metadata/              [Provided separately]
|   |-- mnist_dataset_28x28.mat
|   |-- MNISTmeta_100.0GBd_4.4mA_1.10Vpp_16nodes.mat
|   |-- W_100.0GBd_4.4mA_1.10Vpp_16nodes.mat
|   |-- bias_100.0GBd_4.4mA_1.10Vpp_16nodes.mat
|   `-- DSOdata_100.0GBd_4.4mA_1.10Vpp_16nodes_<block_end_index>.bin
|
`-- MG_metadata/                 [Provided separately]
    |-- MackeyGlass_dataset.mat
    |-- MG_100.0GBd_4.3mA_1.10Vpp_9nodes.mat
    `-- DSOdata_100.0GBd_4.3mA_1.10Vpp_9nodes_<block_end_index>.bin

```

The "<block_end_index>" segment is the original cumulative sample index for each recorded DSO block. There are multiple files with different indices; retain the provided names rather than literally using the angle-bracket placeholder. The exact number of files depends on the data/block settings.

### Run from MATLAB

Run one analysis at a time:

1. run('SpikeLD_RC_IPC_forReview.m');
2. run('SpikeLD_RC_Classification_forReview.m');
3. run('SpikeLD_RC_TimeSeriesPrediction_forReview.m');

Each script starts by clearing the workspace and closing existing figures; run them sequentially rather than relying on variables from another script. Change the "par.*" settings near the start of a script only if you also have the matching measurements and metadata files. To reduce figure generation, set "plot_on = 0" at the start of the relevant script.

### Analyses and expected results

1. Information-processing capacity (IPC)

Script: "SpikeLD_RC_IPC_forReview.m"  

Default condition: 100 GBaud; 4.3 mA bias; 1.10 Vpp drive; 20-symbol temporal mask.

Outputs: "IPC_results/Results_100.0GBd_4.3mA_1.10Vpp_20nodes.mat" (including "par.MC" and "par.IPC"), together with IPC diagnostic and summary PNG figures when plotting is enabled. The Command Window reports "MC1" through "MC7" and total "IPC".

The script uses a stored random input sequence ("u_save", 10,000 original input samples), with 5,000 samples assigned to the initial interval and 5,000 to capacity evaluation. It forms delayed Legendre-polynomial targets up to degree 7 and a maximum input delay of 20 steps.

Manuscript comparison: Fig. 5a,b (order-resolved IPC and single-delay correlations). No single numerical total IPC is specified here as a pass/fail threshold; compare the degree-resolved results with the corresponding experimental condition in the paper. One delay step is one original random input sample, corresponding to 200 ps for the 20-symbol mask at 100 GBaud, not one 10-ps transmitted symbol.

2. MNIST handwritten-digit classification

Script: "SpikeLD_RC_Classification_forReview.m"  

Default condition: 100 GBaud; 4.4 mA bias; 1.10 Vpp; mask length 16; 28 × 28 input images (3,136 measured reservoir features per image). The dataset contains 60,000 training and 10,000 test images.

Outputs: "MNIST_results/Results_100.0GBd_4.4mA_1.10Vpp_16nodes.mat" (including "par.Accuracy", "par.confMat", "Y_test", and "T_test"), a timestamped "RunLog_...txt", and training/confusion-matrix/accuracy figures when plotting is enabled. The Command Window reports accuracy, cross-entropy loss (CEL), and mean-square error (MSE).

Readout options (set at the beginning of the script):
- Evaluate the saved MMSE-trained readout weights without updating them: set "par.train_method = 'SGD-Adam'", "par.retrain = 0", and "par.freezeW = 1". This evaluates the MMSE-initialized readout using the measured reservoir states, without Adam fine-tuning. The resulting accuracy does not necessarily reproduce the reported 92.6%, which was obtained after Adam optimization.
- Fine-tune from saved MMSE-initialized weights: set "par.train_method = 'SGD-Adam'", "par.retrain = 0", "par.freezeW = 0", and "par.epochs = 100". The saved weight files contain the MMSE initialization, not already-final Adam weights, when reproducing the two-stage training procedure.
- MMSE-only regression: set "par.train_method = 'MMSE'". The script computes the regression weights in "par.W" and "par.bias". In this source version, those values are not automatically exported as "W_...mat" and "bias_...mat"; they must be saved separately if they will initialize a subsequent Adam run.

Setting "par.retrain = 1" with unfrozen Adam training uses random initialization and is not the published MMSE-then-Adam training procedure.

Manuscript reference: 92.6% classification accuracy on the complete 10,000-image MNIST test set under the weak-spiking, 16-symbol-mask condition (Fig. 4e). This is the manuscript result; a rerun should be checked against it using the matching measured data and training/readout settings. 

3. Mackey–Glass time-series prediction

Script: "SpikeLD_RC_TimeSeriesPrediction_forReview.m"  

Default condition: 100 GBaud; 4.3 mA bias; 1.10 Vpp; mask length 9; 25-sample input windows. The supplied target dataset should contain the 10 future prediction horizons used in the paper.

Outputs: "MG_results/Results_100.0GBd_4.3mA_1.10Vpp_9nodes.mat", a timestamped "RunLog_...txt", and "Prediction_100.0GBd_4.3mA_1.10Vpp_9nodes.png" when plotting is enabled. The result structure includes "par.NMSE", the test NMSE for each prediction horizon, and the file also contains "Y_test" and "T_test".

Manuscript reference: minimum NMSE approximately 0.0059, with NMSE below 0.017 through ten future samples for the reported experimental condition (Fig. 5d,f). The exact comparison should be made per prediction horizon, not solely by comparing the average NMSE printed in the Command Window.

### Common issues

- "Missing ... input folder": Extract the data folders directly beside the ".m" files.
- "Missing ... dataset" or "Missing ... metadata": Match filenames, capitalization, bias, drive voltage, and mask length to the script settings.
- "Missing DSO file": Download all block-indexed recordings for MNIST/Mackey–Glass; do not rename the block indices.
- "Invalid start-frame metadata" or "Insufficient DSO samples": Ensure the metadata and recordings came from the same acquisition configuration and are complete.
- Missing "W_...mat" or "bias_...mat": The MMSE-initialized readout parameters are required when par.retrain = 0. Use the corresponding files supplied with the experimental data, or run MMSE regression and explicitly save the resulting weights and bias before using them for Adam initialization.
- Undefined "rcosdesign", "upfirdn", or "resample": Check the installed MATLAB release and toolboxes.
- Undefined "donutchart": Use MATLAB R2023b or later, or set "plot_on = 0" in the IPC script. 

The measurements, random projection masks, synchronization positions, and applicable readout weights must correspond to the same experimental acquisition. Combining independent regenerated data with unrelated DSO recordings generally does not reproduce the reported results.

### Contact and citation

For scientific questions or requests concerning the associated experimental files, please contact the corresponding author identified in the manuscript.
