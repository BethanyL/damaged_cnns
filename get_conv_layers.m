function wts = get_conv_layers(net, conv_layers, face_flag)
%
% returns weights from network (from convolutional layers)
%
% INPUTS:
%
% net
%           CNN (Matconvnet format)
%
% conv_layers
%           vector list of which layers in the CNN are convolutional (contain 
%           weights we want to damage)
%
% face_flag
%           1 if using VGG-face, 0 otherwise
%
% OUTPUTS:
%
% wts
%           cell array of weights from each convolutional layer
%

if face_flag
    net = net.net;
end

wts = cell(length(conv_layers), 1);
for j = 1:length(conv_layers)
    
    wts{j} = net.layers{conv_layers(j)}.weights{1};
end
