function save_batch(file_list, filename, net_meta)
% given list image filenames, load each image, preprocess it, and stack them all together.
% save batch of images as .mat file

number_images = length(file_list);
batch = single(zeros(224, 224, 3, number_images));
for j = 1:number_images
    image_name = sprintf('../matconvnet-1.0-beta16/examples/data/imagenet12/images2012val/ILSVRC2012_val_%08d.JPEG',file_list(j));
    batch(:,:,:,j) = preprocess_image(image_name, net_meta);
end
save(filename, 'batch');

end

