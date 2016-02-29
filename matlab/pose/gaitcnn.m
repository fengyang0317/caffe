addpath('../')
caffe.reset_all();
modelDefFile='../../examples/pose/deploy.prototxt';
modelFile='../../examples/pose/posenet_train_iter_75000.caffemodel';
%modelDefFile='../../examples/pose/xxu31/test.prototxt';
%modelFile='../../examples/pose/xxu32/chalearn_train_step1_iter_10000.caffemodel';
dims = [256 256];
layerName = 'conv5_fusion';
%layerName = 'conv8';
caffe.set_mode_gpu();
gpu_id = 0;  
caffe.set_device(gpu_id);
net = caffe.Net(modelDefFile, modelFile, 'test');

% inputDir = '/home/yfeng23/lab/dataset/gait/DatasetB/';
% files = importdata([inputDir 'croplist']);
% if ~exist([inputDir 'mean_' layerName], 'dir')
%     mkdir([inputDir 'mean_' layerName]);
% end

%inputDir = '/home/yfeng23/lab/dataset/gait/DatasetB/crop/001/nm-01/090/';
inputDir = 'images/';
files = dir([inputDir '*.png']);

mean_v = zeros(1,1,3,'single');
mean_v(1) = 105;
mean_v(2) = 115;
mean_v(3) = 119;

for ind=1:length(files)
    %imFile = files{ind};
    imFile = files(ind).name;
    [a, b, c]=fileparts(imFile);
	fprintf('file: %s\n', imFile);
    
    img = imread([inputDir imFile]);
    [m,n,~] = size(img);
    if m < n
        img = imcrop(img, [(n-m)/2, 1, m-1, m-1]);
    end
    im = img;
    img = single(img);
    img = img(:,:,[3, 2, 1]);
    img = bsxfun(@minus, img, mean(mean(img)));
    img = imresize(img, dims);
    img = permute(img, [2 1 3]);

    net.forward({img});
    features = net.blobs(layerName).get_data();
    features = permute(features, [2 1 3]);
    %save([inputDir 'mean_' layerName '/' b],'features');
    
    figure(1)
    img = single(rgb2gray(im))/255;
    img = img + imresize(sum(features, 3), size(img));
    imshow(img);
    figure(2)
    feat = ones(64, 65, 16);
    feat(:,1:64,:)=features;
    feat = reshape(feat, 64, []);
    feat = imresize(feat, size(feat) * 2);
    imshow(feat);
    drawnow
    pause
end
