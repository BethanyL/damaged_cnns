function net =  setup_network(net_file_name, face_flag)

root = fileparts(fileparts(mfilename('fullpath'))) ;
addpath(fullfile(root, 'matlab')) ;
addpath(fullfile(root, 'matlab', 'mex')) ;
addpath(fullfile(root, 'matlab', 'simplenn')) ;
addpath(fullfile(root, 'matlab', 'xtest')) ;

% load the pre-trained CNN and dataset (already evened out) 
if face_flag
    net = lib.face_feats.convNet(net_file_name);
    net.net.layers = net.net.layers(1:end-2);
else
    net = load(net_file_name);
    net = vl_simplenn_tidy(net) ;
end