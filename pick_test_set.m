truth = dlmread('./imagenet12/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt');
net = load( './imagenet12/imagenet-vgg-f.mat');

% pick random 10 images for each of the 1000 classes
image_list = zeros(10, 1000); 
for j = 1:1000
    indClass = find(truth == j);
    image_list(:,j) = indClass(randperm(length(indClass), 10));
end

% each batch is one random example from each class 
for k = 1:10
    batch = image_list(k,:);
    filename = sprintf('batch%d.mat',k);
    save_batch(batch, filename, net.meta);
end

    