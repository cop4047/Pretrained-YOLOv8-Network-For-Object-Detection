function output = preprocessYOLOv8Input(net, image)
    % Get the input size of the network.
    inputSize = net.Layers(1).InputSize;
    % Apply Preprocessing on the input image.
    image = imresize(image, inputSize(1:2));
    output = single(rescale(image));
end