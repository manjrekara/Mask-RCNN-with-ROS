% Read the image for inference

%img = imread(img_curr);
img = imread("test1.png");


% Define the target size of the image for inference
targetSize = [700 700 3];

% Resize the image maintaining the aspect ratio and scaling the largest
% dimension to the target size.
imgSize = size(img);
[~, maxDim] = max(imgSize);
resizeSize = [NaN NaN]; 
resizeSize(maxDim) = targetSize(maxDim);

img = imresize(img, resizeSize);

trainImgSize = [800 800 3];
classNames = {'person', 'car', 'background'};
numClasses = 2;

params = createMaskRCNNConfig(trainImgSize, numClasses, classNames);

executionEnvironment = "cpu";

datadir = tempdir;
url = 'https://www.mathworks.com/supportfiles/vision/data/maskrcnn_pretrained_person_car.mat';

helper.downloadTrainedMaskRCNN(url, datadir);

pretrained = load(fullfile(datadir, 'maskrcnn_pretrained_person_car.mat'));
net = pretrained.net;

maskSubnet = helper.extractMaskNetwork(net);

% detect the objects and their masks
[boxes, scores, labels, masks] = detectMaskRCNN(net, maskSubnet, img, params, executionEnvironment);

% Visualize Predictions

% Overlay the detected masks on the image using the insertObjectMask
% function.
if(isempty(masks))
    overlayedImage = img;
else
    overlayedImage = insertObjectMask(img, masks);
end
figure, imshow(overlayedImage)

% Show the bounding boxes and labels on the objects
showShape("rectangle", gather(boxes), "Label", labels, "LineColor",'g')