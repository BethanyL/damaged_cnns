function save_batch_face(file_list, filename)

% file list filled with indices into images/PersonEachImage (1:30281)

faces = load('../data/faceData/FacesInTheWild');

number_images = length(file_list);

batch = cell(number_images, 1);
for j = 1:number_images
    image_name = strcat('../matconvnet-1.0-beta16/data/faceData/',faces.metaData{file_list(j)}.fileName);
    batch{j} = imread(image_name);
end
save(filename, 'batch');

end

