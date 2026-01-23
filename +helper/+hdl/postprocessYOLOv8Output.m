function [bboxes,scores,labels] = postprocessYOLOv8Output(hwprediction, img, numClasses, classNames)
imageSize = size(img);
[bboxes,scores,labelIds] = helper.postprocess(hwprediction, ...
    imageSize, imageSize, numClasses);

bboxes = gather(bboxes);
scores = gather(scores);
labelIds = gather(labelIds);

% Map labelIds back to labels.
labels = classNames(labelIds);
end