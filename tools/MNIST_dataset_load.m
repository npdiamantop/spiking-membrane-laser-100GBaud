%% MNIST dataset
% [1] Y. LeCun, C. Cortes, & C.J.C. Burges, "The MNIST database of handwritten digits". 
%       Available at https://web.archive.org/web/20200430193701/http://yann.lecun.com/exdb/mnist/ 
%       (accessed 23 June 2026).
% ------------------------------------------------------------------------
clear; clc; close all; rng(101);

new_size = [];%[12 12]; % leave empty [] if no resize

% Use Inf for full MNIST !!!
Ntrain_per_class = Inf; %160;
Ntest_per_class  = Inf; %40;

TrainData = loadMNISTImages("train-images.idx3-ubyte",new_size);
TestData = loadMNISTImages("t10k-images.idx3-ubyte",new_size);
TrainLabels_raw = loadMNISTLabels("train-labels.idx1-ubyte");
TestLabels_raw = loadMNISTLabels("t10k-labels.idx1-ubyte");
TrainLabels = onehot(TrainLabels_raw);
TestLabels = onehot(TestLabels_raw);
Labels = [{"0"};{"1"};{"2"};{"3"};{"4"};{"5"};{"6"};{"7"};{"8"};{"9"}];

%% Select subset, class-balanced
Nclass = numel(Labels);
if ~isinf(Ntrain_per_class)
    idx_train_keep = [];
    for cc = 0:Nclass-1
        idx_cc = find(TrainLabels_raw == cc);
        if numel(idx_cc) < Ntrain_per_class
            error('Class %d has only %d training samples, requested %d.', ...
                cc, numel(idx_cc), Ntrain_per_class);
        end
        idx_cc = idx_cc(randperm(numel(idx_cc))); % randomize within class
        idx_train_keep = [idx_train_keep; idx_cc(1:Ntrain_per_class)];
    end
    idx_train_keep = idx_train_keep(randperm(numel(idx_train_keep))); % shuffle
    TrainData   = TrainData(:,:,idx_train_keep);
    TrainLabels = TrainLabels(:,idx_train_keep);
end

if ~isinf(Ntest_per_class)
    idx_test_keep = [];
    for cc = 0:Nclass-1
        idx_cc = find(TestLabels_raw == cc);
        if numel(idx_cc) < Ntest_per_class
            error('Class %d has only %d test samples, requested %d.', ...
                cc, numel(idx_cc), Ntest_per_class);
        end
        idx_cc = idx_cc(randperm(numel(idx_cc))); % randomize within class
        idx_test_keep = [idx_test_keep; idx_cc(1:Ntest_per_class)];
    end
    idx_test_keep = idx_test_keep(randperm(numel(idx_test_keep))); % shuffle
    TestData   = TestData(:,:,idx_test_keep);
    TestLabels = TestLabels(:,idx_test_keep);
end

fprintf('MNIST subset loaded:\n');
fprintf('TrainData size: %d x %d x %d\n',size(TrainData,1),size(TrainData,2),size(TrainData,3));
fprintf('TestData size : %d x %d x %d\n',size(TestData,1),size(TestData,2),size(TestData,3));
fprintf('Training samples per class: %s\n',mat2str(sum(TrainLabels,2).'));
fprintf('Test samples per class    : %s\n',mat2str(sum(TestLabels,2).'));

if ~isempty(new_size)
    save(sprintf("mnist_dataset_%dx%d.mat",new_size(1),new_size(2)),"TestLabels","TrainLabels","TestData","TrainData","Labels");
else
    save("mnist_dataset_28x28.mat","TestLabels","TrainLabels","TestData","TrainData","Labels");
end

%%-------------------------------------------------------------------------

function images = loadMNISTImages(filename,new_size)
%loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
%the raw MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);

numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

images = fread(fp, inf, 'unsigned char');
images = reshape(images, numCols, numRows, numImages);
images = permute(images,[2 1 3]);
fclose(fp);

images = double(images)/255;    % [0,1] norm
%images = images > 0.5;         % {0,1} binarization
if ~isempty(new_size)
    images = resize(images,new_size); % resize using bilinear interpolation
end

% Reshape to #pixels x #examples
%images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));

end

function labels = loadMNISTLabels(filename)
%loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
%the labels for the MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', filename, '']);

numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');

labels = fread(fp, inf, 'unsigned char').'; % 1 x N

assert(size(labels,2) == numLabels, 'Mismatch in label count');

fclose(fp);

end

%% resize using bilinear interpolation
function resized_images = resize(images,new_size)
    num_images = size(images,3);
    resized_images = zeros([new_size(1), new_size(2), num_images]);
    for i = 1:num_images
        img = images(:, :, i);
        resized_img = imresize(img, new_size, 'bilinear');
        resized_images(:, :, i) = resized_img;
    end
end

%% make one-hot target matrix
function targets_out = onehot(targets_in)
    Nclass = 10;
    Ndata = length(targets_in);
    targets_out = zeros(Nclass,Ndata);
    for no = 0:Nclass-1
        targets_out(no+1,targets_in==no) = 1; % one-hot vector
    end
end
