addpath('../')
caffe.reset_all();
modelDefFile='../../examples/pose/deploy.prototxt';
modelFile='../../examples/pose/pose4mc12joints_10000.caffemodel';
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

inputDir = 'images/';
files = dir([inputDir '*.png']);

mean_v = zeros(1,1,3,'single');
mean_v(1) = 84.5695;
mean_v(2) = 91.5907;
mean_v(3) = 139.752;

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
    figure(1)
    imshow(img);
    img = single(img);
    img = img(:,:,[3, 2, 1]);
    img = bsxfun(@minus, img, mean(mean(img)));
    img = imresize(img, dims);
    img = permute(img, [2 1 3]);

    net.forward({img});
    features = net.blobs(layerName).get_data();
    features = permute(features, [2 1 3]);
    %save([inputDir 'mean_' layerName '/' b],'features');
    
    figure(2)
    feat = ones(64, 65, 12);
    feat(:,1:64,:)=features;
    feat = reshape(feat, 64, []);
    feat = imresize(feat, size(feat) * 2);
    imshow(feat);
    drawnow
    pause
end
