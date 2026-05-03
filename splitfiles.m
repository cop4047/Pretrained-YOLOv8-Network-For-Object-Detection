src_images = 'D:/Downloads/mini/yolo/images';
src_labels = 'D:/Downloads/mini/yolo/labels';

train_images = 'D:/Downloads/mini/yolo/images/train';
val_images   = 'D:/Downloads/mini/yolo/images/val';
train_labels = 'D:/Downloads/mini/yolo/labels/train';
val_labels   = 'D:/Downloads/mini/yolo/labels/val';

mkdir(train_images); mkdir(val_images);
mkdir(train_labels); mkdir(val_labels);

imgs = dir(fullfile(src_images, '*.jpg'));
idx  = randperm(numel(imgs));
n_train = round(0.8 * numel(imgs));

for i = 1:numel(imgs)
    img_name = imgs(idx(i)).name;
    [~, stem] = fileparts(img_name);

    if i <= n_train
        dst_img = train_images;
        dst_lbl = train_labels;
    else
        dst_img = val_images;
        dst_lbl = val_labels;
    end

    movefile(fullfile(src_images, img_name),        fullfile(dst_img, img_name));
    lbl = fullfile(src_labels, stem + ".txt");
    if isfile(lbl)
        movefile(lbl, fullfile(dst_lbl, stem + ".txt"));
    end
end

disp("Done! Train: " + n_train + "  Val: " + (numel(imgs) - n_train))