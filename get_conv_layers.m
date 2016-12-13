function wts = get_conv_layers(net, conv_layers, face_flag)

if face_flag
    net = net.net;
end

wts = cell(length(conv_layers), 1);
for j = 1:length(conv_layers)
    
    wts{j} = net.layers{conv_layers(j)}.weights{1};
end
