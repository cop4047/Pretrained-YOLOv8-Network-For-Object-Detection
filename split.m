src_images = 'D:/Downloads/mini/yolo/images';
src_labels = 'D:/Downloads/mini/labels';
out_root   = 'D:/Downloads/mini/split';

% Create output folders
for split = ["train","val"]
    mkdir(fullfile(out_root, 'images', split));
    mkdir(fullfile(out_root, 'labels', split));
end

% Get all images
imgs = dir(fullfile(src_images, '*.jpg'));  % change to *.png if needed
idx  = randperm(numel(imgs));
n_train = round(0.8 * numel(imgs));

for i = 1:numel(imgs)
    if i <= n_train
        split = 'train';
    else
        split = 'val';
    end
    
    % Copy image
    copyfile(fullfile(src_images, imgs(idx(i)).name), ...
             fullfile(out_root, 'images', split, imgs(idx(i)).name));
    
    % Copy matching label
    [~, stem] = fileparts(imgs(idx(i)).name);
    label_file = fullfile(src_labels, stem + ".txt");
    if isfile(label_file)
        copyfile(label_file, fullfile(out_root, 'labels', split, stem + ".txt"));
    end
end

disp("Done! Train: " + n_train + "  Val: " + (numel(imgs)-n_train))