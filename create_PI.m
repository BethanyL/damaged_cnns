% Creates vector PersonEachImage
% for each image, the vector contains the class number for the person in the image 

faces = load('../faceData/FacesInTheWild');
numPeople = length(faces.lexicon);
numImgs = length(faces.metaData);
PersonEachImage = zeros(numImgs,1);
for j = 1:numImgs
    p = faces.metaData{j}.clusterNum;
    PersonEachImage(j) = p;
end
save('PI.mat','PersonEachImage');