function im_ = preprocess_image(filename, net_meta)
% preprocess image for ImageNet network


im = imread(filename) ;

% if grayscale, convert to RGB
if size(im,3) == 1
    im = gray2rgb(im);
end

% resize image to size expected by network and convert to single
im_ = imresize(single(im), net_meta.normalization.imageSize(1:2)) ;

% check if resizing worked 
if ((size(im_,1) ~= size(net_meta.normalization.averageImage,1)) || ...
        (size(im_,2) ~= size(net_meta.normalization.averageImage,2))) ...
        || (size(im_,3) ~= size(net_meta.normalization.averageImage,3))
    % if failed to switch to right size, skip
    % TODO: figure out better solution
    fprintf('size problem on image %s\n',filename)
    scores = zeros(length(net_meta.classes.name),1);
    total_num_damaged = 0;
    keyboard
    return
end

% subtract average image 
im_ = im_ - net_meta.normalization.averageImage ;


end

