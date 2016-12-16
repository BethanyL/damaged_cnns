function features = get_facial_features(faceDet, convNet, batch_name)
% returns matrix of size num_images x num_features
% gives us the features of each image by running it through the CNN

% loads batch, cell array of length num_images
new_batch = load(batch_name);

num_images = length(new_batch.batch);
num_features = 4096; % should find this programmatically 
features = zeros(num_images, num_features);

for j = 1:num_images
    img = new_batch.batch{j};
    
    % first, detect face and crop image
    det = faceDet.detect(img);

    if isempty(det)
        det = [1; 1; size(img,2); size(img,1)];
    end

    crop = lib.face_proc.faceCrop.crop(img,det(1:4,1));

    % now ready to send through CNN
    feat = convNet.simpleNN(crop);
    features(j,:) = feat/norm(feat);
end