function scores = classify_image_imagenet(net, batch_name)

% loads batch, an array of size  224 x 224 x 3 x num_images
new_batch = load(batch_name);

% run the CNN
res = vl_simplenn(net, new_batch.batch);

% output class score for each image and each class 
scores = squeeze(res(end).x)' ;
