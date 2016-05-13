clear
addpath('../')
caffe.reset_all();
modelDefFile='../../examples/pose/deploy.prototxt';
modelFile='../../examples/pose/pose4gait.caffemodel';
dims = [256 256];
%layerName = 'conv5_fusion';
layerName = {'conv8', 'conv5_fusion'};%, 'conv7', 'conv4_fusion'};
caffe.set_mode_gpu();
gpu_id = 0;  
caffe.set_device(gpu_id);
net = caffe.Net(modelDefFile, modelFile, 'test');

inputDir = '/home/yfeng23/lab/dataset/gait/DatasetB/';
for ln = 1:length(layerName)
    if ~exist([inputDir layerName{ln}], 'dir')
        mkdir([inputDir layerName{ln}]);
    end
end

for person = 112:112
    fprintf(1, '%d starting\n', person);
    for walk = 6:6
        angle = 72;
        con = 0;
        for ln = 1:length(layerName)
            if exist(sprintf('%s%s/%03d-nm-%02d-%03d.mat',inputDir, layerName{ln},...
                person, walk, angle), 'file')
                con = con + 1;
            end
        end
        %if con == length(layerName)
        %    continue
        %end
        ad = sprintf('%scrop/%03d/nm-%02d/%03d', inputDir, person, walk, angle);
        png = dir([ad '/*.png']);
        num = length(png);
        frame_id = zeros(num, 1);
        cfeat = cell(1, length(layerName));
        for l=1:num
            img=imread([ad '/' png(l).name]);
            %img = rgb2gray(img);
            img = single(img);
            [m,n,~] = size(img);
            assert(m==n);
            img = img - 112;
%             if size(img,1)>size(img,2)
%                 img = padarray(img,[0,floor(size(img,1)-size(img,2))/2,0]);
%             end
            img = imresize(img, dims);
            %img = img(:, :, [3, 2, 1]);
            img = permute(img, [2 1 3]);

            net.forward({img});
            for ln = 1:length(layerName)
                features = net.blobs(layerName{ln}).get_data();
                features = permute(features, [2 1 3]);
                features = reshape(features, [1 size(features)]);
                cfeat{ln} = [cfeat{ln}; features];
            end
            frame_id(l) = str2double(png(l).name(15:17));
        end
        for ln = 1:length(layerName)
            feeat = cfeat{ln};
            save(sprintf('%s%s/%03d-nm-%02d-%03d',inputDir, layerName{ln},...
                person, walk, angle), 'frame_id', 'feat');
        end
    end
end
