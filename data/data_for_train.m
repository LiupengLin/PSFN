clear;
clc;
train_dir = '\traindata';

%% Polarimetric Mode of Single Polarimetric SAR Image
% polmode = 'HH';
% polmode = 'HV';
polmode = 'VV';

%% Data Serial Number
serial = '_20200105';

%% Path Setting
lr_polsar_info = dir(fullfile(train_dir,'lr_polsar','*.mat'));
hr_sar_info = dir(fullfile(train_dir,'hr_sar',polmode,'*.mat'));
hr_polsar_info = dir(fullfile(train_dir,'hr_polsar','*.mat'));

lr_polsar = fullfile({lr_polsar_info.folder},{lr_polsar_info.name});
hr_sar = fullfile({hr_sar_info.folder},{hr_sar_info.name});
hr_polsar = fullfile({hr_polsar_info.folder},{hr_polsar_info.name});

savepath = fullfile(train_dir,strcat('train_',polmode,serial,'.h5'));

%% Size Setting for Train Data 
scale = 2;
size_hr = 40;
stride_hr = 20;
size_lr = size_hr/scale;
stride_lr = stride_hr/scale;

%% Parameter Initialization
lr_polsar_data = zeros(size_lr, size_lr, 9, 1);
hr_sar_data = zeros(size_hr, size_hr, 1, 1);
hr_polsar_data = zeros(size_hr, size_hr, 9, 1);
count = 0;
num = length(hr_polsar);

%% Data Generate
for para1 = 1:num
    load(lr_polsar{para1});
    load(hr_sar{para1});
    load(hr_polsar{para1});
    [hei,wid,cha] = size(hr_polsar_train);
    for parax = 1:1:floor((hei-size_hr)/stride_hr)+1
        for paray = 1:1:floor((wid-size_hr)/stride_hr)+1
            subim_lr_polsar = lr_polsar_train(1+(parax-1)*stride_lr:size_lr+(parax-1)*stride_lr,1+(paray-1)*stride_lr:size_lr+(paray-1)*stride_lr,:);
            subim_hr_sar = hr_sar_train(1+(parax-1)*stride_hr:size_hr+(parax-1)*stride_hr,1+(paray-1)*stride_hr:size_hr+(paray-1)*stride_hr,:);
            subim_hr_polsar = hr_polsar_train(1+(parax-1)*stride_hr:size_hr+(parax-1)*stride_hr,1+(paray-1)*stride_hr:size_hr+(paray-1)*stride_hr,:);
            count=count+1;
            lr_polsar_data(:, :, :, count) = subim_lr_polsar;
            hr_sar_data(:, :, :, count) = subim_hr_sar;
            hr_polsar_data(:, :, :, count) = subim_hr_polsar;
        end
    end
end

order = randperm(count);
lr_polsar_data = lr_polsar_data(:, :, :, order);
hr_sar_data = hr_sar_data(:, :, :, order);
hr_polsar_data = hr_polsar_data(:, :, :, order); 

%% writing to HDF5
chunksz = 64;
created_flag = false;
totalct = 0;
for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata1 = lr_polsar_data(:,:,:,last_read+1:last_read+chunksz);
    batchdata2 = hr_sar_data(:,:,:,last_read+1:last_read+chunksz);
    batchlabs = hr_polsar_data(:,:,:,last_read+1:last_read+chunksz);
    startloc = struct('dat1',[1,1,1,totalct+1], 'dat2',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata1, batchdata2, batchlabs, ~created_flag, startloc, chunksz);
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath);