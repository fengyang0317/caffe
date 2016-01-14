addpath('../')
caffe.reset_all();
modelDefFile='../../examples/pose/deploy.prototxt';
modelFile='../../examples/pose/posenet_train_iter_20000.caffemodel';
dims = [256 256];
%layerName = 'conv5_fusion';
layerName = 'conv8';
caffe.set_mode_gpu();
gpu_id = 0;  
caffe.set_device(gpu_id);
net = caffe.Net(modelDefFile, modelFile, 'test');

inputDir = '/home/yfeng23/lab/dataset/gait/DatasetB/';
files = importdata([inputDir 'croplist']);

if ~exist([inputDir 'mean_' layerName], 'dir')
    mkdir([inputDir 'mean_' layerName]);
end

%inputDir = 'images/';
%files = dir([inputDir '*.png']);

for ind=1:length(files)
    imFile = files{ind};
    %imFile = files(ind).name;
    [a, b, c]=fileparts(imFile);
	fprintf('file: %s\n', imFile);
    
    img = imread([inputDir imFile]);
    img = rgb2gray(img);
    %figure(1)
    %imshow(img);
    %img = imread('images/00000000.jpg');
    %img = imread('images/out096.png');
    img = single(img);
    [m,n,~] = size(img);
    img = img - 112.512;
    if size(img,1)>size(img,2)
        img = padarray(img,[0,floor(size(img,1)-size(img,2))/2,0]);
    end
    img = imresize(img, dims);
    %img = img(:, :, [3, 2, 1]);
    img = permute(img, [2 1 3]);

    net.forward({img});
    features = net.blobs(layerName).get_data();
    features = permute(features, [2 1 3]);
    save([inputDir 'mean_' layerName '/' b],'features');
    
%     figure(2)
%     feat = ones(32, 33, 12);
%     feat(:,1:32,:)=features;
%     feat = reshape(feat, 32, []);
%     feat = imresize(feat, size(feat) * 4);
%     imshow(feat);
%     drawnow
end
