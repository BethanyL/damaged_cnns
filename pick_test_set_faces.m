% pick 500 random people and 5 images of each of them 
% each batch has one of the 500 people

faces = load('../data/faceData/FacesInTheWild');
% faces.lexicon: list of people, 1 x 14808 cell, i.e. 'Scott Ritter'
% faces.metaData: list of images, 1 x 30281 cell, i.e. ...
% faces.metaData{1}.fileName = % '2002/09/28/img_828.0.ppm' 
% and faces.metaData{1}.clusterNum = 1

numPeople = length(faces.lexicon);
numImgs = length(faces.metaData);

% for each image, gives clusterNum 
% 30281 entries, ranging -1 to 14,807
% 9,958 unique clusters 
% 11% (3369) are -1 
load('../data/faceData/PI.mat','PersonEachImage');

countImgs = zeros(numPeople,1);
for j = 1:numPeople
    countImgs(j) = length(find(PersonEachImage == j));
end
% sum(countImgs) = 26912 because that's how many images have clusterNum
% other than -1

% pick 500 random people who have at least 5 images of them 
our_num_images = 5; 
our_num_people = 50; 
candidatePeople = find(countImgs >= our_num_images); % there are 907 candidates
ourSetPeople = randperm(length(candidatePeople), our_num_people); 
ourSetPeople = candidatePeople(ourSetPeople);
save('ourSetPeople','ourSetPeople');


% pick random 5 images for each of the 500 classes
% fill image_list with indices into images/PersonEachImage (1:30281)
image_list = zeros(our_num_images, our_num_people); 
for j = 1:our_num_people
    p = ourSetPeople(j);
    indClass = find(PersonEachImage == p);
    image_list(:,j) = indClass(randperm(length(indClass), our_num_images));
end

for k = 1:our_num_images
    batch = image_list(k,:);
    filename = sprintf('prebatch%d_faces.mat',k);
    save_batch_face(batch, filename);
end

    