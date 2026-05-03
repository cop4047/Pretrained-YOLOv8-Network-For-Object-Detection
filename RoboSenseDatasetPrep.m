%% =====================================================================
%% ROBOSENSE DATASET PREPARATION FOR LANE DETECTION
%% =====================================================================
%%
%% This utility prepares RoboSense data for:
%% 1. Lane segmentation model fine-tuning
%% 2. Domain adaptation from CULane → RoboSense
%% 3. Validation testing
%%
%% Usage:
%%   >> robosense_prep = RoboSenseDatasetPrep('path/to/robosense_data');
%%   >> [frames, labels] = robosense_prep.loadDataset();
%%   >> robosense_prep.visualizeFrames(frames(1:10));
%%   >> robosense_prep.exportForTraining('./export_dir');

classdef RoboSenseDatasetPrep
    % Prepare RoboSense dataset for autonomous vehicle CV
    
    properties
        datasetPath        % Root directory of RoboSense data
        imageDir           % Image directory
        annotationDir      % Annotation directory (if exists)
        calibDir           % Calibration directory
        outputDir          % Output directory for processed data
        
        % Data statistics
        numFrames
        frameWidth
        frameHeight
        hasAnnotations     % Boolean: dataset has lane annotations
    end
    
    methods
        function obj = RoboSenseDatasetPrep(datasetPath, outputDir)
            % Initialize dataset processor
            %
            % Inputs:
            %   datasetPath: Path to RoboSense dataset root
            %   outputDir: Where to save processed data (optional)
            
            if nargin < 1
                error('RoboSenseDatasetPrep requires dataset path');
            end
            
            obj.datasetPath = datasetPath;
            obj.imageDir = fullfile(datasetPath, 'image');
            obj.annotationDir = fullfile(datasetPath, 'annotations');
            obj.calibDir = fullfile(datasetPath, 'calib');
            
            if nargin < 2
                obj.outputDir = fullfile(datasetPath, 'processed');
            else
                obj.outputDir = outputDir;
            end
            
            mkdir(obj.outputDir);
            
            % Check dataset structure
            obj.inspectDataset();
        end
        
        function inspectDataset(obj)
            % Check RoboSense dataset structure and contents
            
            fprintf('=== RoboSense Dataset Inspection ===\n\n');
            
            % Check image directory
            if isfolder(obj.imageDir)
                imageFiles = dir(fullfile(obj.imageDir, '*.png'));
                obj.numFrames = length(imageFiles);
                fprintf('✓ Image directory found: %d frames\n', obj.numFrames);
                
                % Sample first image to get dimensions
                if obj.numFrames > 0
                    sampleImg = imread(fullfile(obj.imageDir, imageFiles(1).name));
                    obj.frameHeight = size(sampleImg, 1);
                    obj.frameWidth = size(sampleImg, 2);
                    fprintf('  Resolution: %d × %d\n', obj.frameWidth, obj.frameHeight);
                end
            else
                warning('Image directory not found at: %s', obj.imageDir);
                obj.numFrames = 0;
            end
            
            % Check annotations
            if isfolder(obj.annotationDir)
                annoFiles = dir(fullfile(obj.annotationDir, '*.json'));
                obj.hasAnnotations = length(annoFiles) > 0;
                if obj.hasAnnotations
                    fprintf('✓ Annotation directory found: %d annotation files\n', length(annoFiles));
                else
                    fprintf('⚠ Annotation directory exists but is empty\n');
                end
            else
                fprintf('⚠ No annotation directory found\n');
                obj.hasAnnotations = false;
            end
            
            % Check calibration
            if isfolder(obj.calibDir)
                calibFiles = dir(obj.calibDir);
                fprintf('✓ Calibration directory found: %d files\n', length(calibFiles) - 2);
            else
                fprintf('⚠ No calibration directory found\n');
            end
            
            fprintf('\n');
        end
        
        function [frames, labels, frameNames] = loadDataset(obj, maxFrames)
            % Load RoboSense images and annotations
            %
            % Outputs:
            %   frames: Cell array of RGB images
            %   labels: Cell array of lane masks (if available)
            %   frameNames: Original filenames
            
            if nargin < 2
                maxFrames = obj.numFrames;  % Load all
            else
                maxFrames = min(maxFrames, obj.numFrames);
            end
            
            fprintf('Loading %d frames from RoboSense...\n', maxFrames);
            
            imageFiles = dir(fullfile(obj.imageDir, '*.png'));
            imageFiles = imageFiles(1:maxFrames);
            
            frames = cell(maxFrames, 1);
            labels = cell(maxFrames, 1);
            frameNames = cell(maxFrames, 1);
            
            for i = 1:maxFrames
                % Load image
                imgPath = fullfile(obj.imageDir, imageFiles(i).name);
                frames{i} = imread(imgPath);
                frameNames{i} = imageFiles(i).name;
                
                % Try to load annotation (lane mask)
                if obj.hasAnnotations
                    % Look for corresponding JSON annotation
                    [~, baseName, ~] = fileparts(imageFiles(i).name);
                    jsonPath = fullfile(obj.annotationDir, [baseName '.json']);
                    
                    if isfile(jsonPath)
                        labels{i} = obj.loadLaneMask(jsonPath, frames{i});
                    else
                        % Generate placeholder (will need manual annotation)
                        labels{i} = false(size(frames{i}, 1), size(frames{i}, 2));
                    end
                else
                    % No annotations - placeholders
                    labels{i} = [];
                end
                
                if mod(i, 100) == 0
                    fprintf('  Loaded %d/%d frames\n', i, maxFrames);
                end
            end
            
            fprintf('✓ Loading complete\n\n');
        end
        
        function laneMask = loadLaneMask(obj, jsonPath, referenceImage)
            % Parse lane annotation from JSON
            % Supports multiple annotation formats
            
            % Read JSON
            try
                jsonText = fileread(jsonPath);
                data = jsondecode(jsonText);
            catch
                warning('Could not parse JSON at: %s', jsonPath);
                laneMask = [];
                return;
            end
            
            % Initialize mask
            h = size(referenceImage, 1);
            w = size(referenceImage, 2);
            laneMask = false(h, w);
            
            % Format 1: Polygon points
            if isfield(data, 'lanes') && isstruct(data.lanes)
                lanes = data.lanes;
                for i = 1:length(lanes)
                    if isfield(lanes(i), 'points')
                        points = lanes(i).points;
                        if ~isempty(points)
                            % Convert points to mask
                            laneMask = obj.pointsToMask(laneMask, points);
                        end
                    end
                end
            end
            
            % Format 2: Pixel-level mask
            if isfield(data, 'mask_rle')
                laneMask = obj.decodeMaskRLE(data.mask_rle, h, w);
            end
            
            % Format 3: Direct binary mask
            if isfield(data, 'lane_mask')
                laneMask = logical(data.lane_mask);
            end
        end
        
        function mask = pointsToMask(~, mask, points)
            % Convert point polygon to binary mask
            
            if size(points, 1) < 3
                return;
            end
            
            % Extract x, y coordinates
            if isstruct(points)
                x = [points.x];
                y = [points.y];
            elseif size(points, 2) == 2
                x = points(:, 1);
                y = points(:, 2);
            else
                return;
            end
            
            % Create polygon mask
            try
                [row, col] = meshgrid(1:size(mask, 2), 1:size(mask, 1));
                in = inpolygon(row, col, x, y);
                mask(in) = true;
            catch
                % Fallback if polygonal masking fails
            end
        end
        
        function mask = decodeMaskRLE(~, rle, h, w)
            % Decode run-length encoded mask
            
            mask = false(h, w);
            
            if ~ischar(rle) && ~isstring(rle)
                return;
            end
            
            % Simple RLE format: count1:value1,count2:value2,...
            parts = split(string(rle), ',');
            idx = 1;
            
            for i = 1:length(parts)
                if contains(parts(i), ':')
                    segment = split(parts(i), ':');
                    count = str2double(segment(1));
                    value = str2double(segment(2));
                    
                    endIdx = min(idx + count - 1, h * w);
                    if value == 1
                        mask(idx:endIdx) = true;
                    end
                    idx = endIdx + 1;
                    
                    if idx > h * w
                        break;
                    end
                end
            end
        end
        
        function normalizedFrames = preprocessFrames(obj, frames)
            % Preprocess frames for training
            % - Resize to 384×384
            % - Normalize (ImageNet stats)
            % - Handle different input formats
            
            targetSize = [384, 384];
            normalizedFrames = cell(length(frames), 1);
            
            fprintf('Preprocessing %d frames to %d×%d...\n', ...
                length(frames), targetSize(1), targetSize(2));
            
            for i = 1:length(frames)
                frame = frames{i};
                
                % Resize
                if ~isequal(size(frame, 1:2), targetSize)
                    frame = imresize(frame, targetSize);
                end
                
                % Convert to single precision
                frame = im2single(frame);
                
                % Normalize (ImageNet statistics)
                meanVals = reshape([0.485, 0.456, 0.406], 1, 1, 3);
                stdVals = reshape([0.229, 0.224, 0.225], 1, 1, 3);
                
                frame = (frame - meanVals) ./ stdVals;
                
                normalizedFrames{i} = frame;
                
                if mod(i, 100) == 0
                    fprintf('  Preprocessed %d/%d\n', i, length(frames));
                end
            end
            
            fprintf('✓ Preprocessing complete\n\n');
        end
        
        function resizedLabels = resizeLabels(obj, labels, targetSize)
            % Resize lane masks to target resolution
            
            targetSize = [384, 384];
            resizedLabels = cell(length(labels), 1);
            
            for i = 1:length(labels)
                if ~isempty(labels{i})
                    resizedLabels{i} = imresize(labels{i}, targetSize, 'nearest') > 0.5;
                else
                    resizedLabels{i} = false(targetSize);
                end
            end
        end
        
        function [train_frames, train_labels, val_frames, val_labels] = ...
                trainValSplit(obj, frames, labels, valRatio)
            % Split dataset into training and validation
            
            if nargin < 4
                valRatio = 0.2;  % 80/20 split
            end
            
            n = length(frames);
            nVal = round(n * valRatio);
            
            % Random permutation
            perm = randperm(n);
            valIdx = perm(1:nVal);
            trainIdx = perm(nVal+1:end);
            
            train_frames = frames(trainIdx);
            train_labels = labels(trainIdx);
            val_frames = frames(valIdx);
            val_labels = labels(valIdx);
            
            fprintf('Dataset split:\n');
            fprintf('  Training: %d frames\n', length(train_frames));
            fprintf('  Validation: %d frames\n', length(val_frames));
            fprintf('\n');
        end
        
        function visualizeFrames(obj, frames, labels)
            % Visualize frames with lane masks
            
            if nargin < 3
                labels = {};
            end
            
            numFrames = length(frames);
            numCols = min(4, numFrames);
            numRows = ceil(numFrames / numCols);
            
            figure('Name', 'RoboSense Dataset Visualization', ...
                   'NumberTitle', 'off', ...
                   'Position', [100 100 1200 800]);
            
            for i = 1:numFrames
                subplot(numRows, numCols, i);
                
                imshow(frames{i});
                
                % Overlay mask if available
                if ~isempty(labels) && ~isempty(labels{i})
                    hold on;
                    mask = labels{i};
                    
                    % Create semi-transparent overlay
                    overlay = false(size(mask, 1), size(mask, 2), 3);
                    overlay(:, :, 2) = mask;  % Green channel
                    
                    h = imshow(overlay);
                    set(h, 'AlphaData', 0.3);
                    hold off;
                end
                
                title(sprintf('Frame %d', i));
            end
        end
        
        function exportForTraining(obj, exportDir, frames, labels)
            % Export dataset in PyTorch format for training
            %
            % Creates directory structure:
            % exportDir/
            % ├── images/
            % │   ├── train/
            % │   └── val/
            % └── labels/
            %     ├── train/
            %     └── val/
            
            if nargin < 2
                exportDir = obj.outputDir;
            end
            
            % Create directories
            trainImgDir = fullfile(exportDir, 'images', 'train');
            valImgDir = fullfile(exportDir, 'images', 'val');
            trainLabelDir = fullfile(exportDir, 'labels', 'train');
            valLabelDir = fullfile(exportDir, 'labels', 'val');
            
            mkdir(trainImgDir);
            mkdir(valImgDir);
            mkdir(trainLabelDir);
            mkdir(valLabelDir);
            
            fprintf('Exporting to: %s\n\n', exportDir);
            
            % Split data
            [train_frames, train_labels, val_frames, val_labels] = ...
                obj.trainValSplit(frames, labels, 0.2);
            
            % Export training data
            fprintf('Exporting training set (%d frames)...\n', length(train_frames));
            for i = 1:length(train_frames)
                imgPath = fullfile(trainImgDir, sprintf('frame_%06d.png', i));
                imwrite(train_frames{i}, imgPath);
                
                if ~isempty(train_labels{i})
                    maskPath = fullfile(trainLabelDir, sprintf('frame_%06d.png', i));
                    imwrite(uint8(train_labels{i}) * 255, maskPath);
                end
                
                if mod(i, 100) == 0
                    fprintf('  Exported %d/%d\n', i, length(train_frames));
                end
            end
            
            % Export validation data
            fprintf('Exporting validation set (%d frames)...\n', length(val_frames));
            for i = 1:length(val_frames)
                imgPath = fullfile(valImgDir, sprintf('frame_%06d.png', i));
                imwrite(val_frames{i}, imgPath);
                
                if ~isempty(val_labels{i})
                    maskPath = fullfile(valLabelDir, sprintf('frame_%06d.png', i));
                    imwrite(uint8(val_labels{i}) * 255, maskPath);
                end
                
                if mod(i, 50) == 0
                    fprintf('  Exported %d/%d\n', i, length(val_frames));
                end
            end
            
            fprintf('\n✓ Export complete\n');
            fprintf('Ready for training with PyTorch/TensorFlow\n');
        end
        
        function analyzeDataset(obj, frames, labels)
            % Analyze dataset statistics
            
            fprintf('=== Dataset Analysis ===\n\n');
            
            % Lane pixel statistics
            if ~isempty(labels) && ~all(cellfun(@isempty, labels))
                lanePixels = [];
                for i = 1:length(labels)
                    if ~isempty(labels{i})
                        lanePixels = [lanePixels; ...
                            sum(labels{i}(:)) / numel(labels{i})];
                    end
                end
                
                if ~isempty(lanePixels)
                    fprintf('Lane Coverage:\n');
                    fprintf('  Mean: %.2f%%\n', mean(lanePixels) * 100);
                    fprintf('  Min: %.2f%%\n', min(lanePixels) * 100);
                    fprintf('  Max: %.2f%%\n', max(lanePixels) * 100);
                    fprintf('  Std: %.2f%%\n\n', std(lanePixels) * 100);
                end
            end
            
            % Image statistics
            imgBrightness = [];
            imgContrast = [];
            
            for i = 1:min(length(frames), 100)  % Sample first 100
                gray = rgb2gray(frames{i});
                imgBrightness = [imgBrightness; mean(gray(:))];
                imgContrast = [imgContrast; std(double(gray(:)))];
            end
            
            fprintf('Image Statistics (sample of %d):\n', length(imgBrightness));
            fprintf('  Brightness: %.1f ± %.1f\n', mean(imgBrightness), std(imgBrightness));
            fprintf('  Contrast (std): %.1f ± %.1f\n', mean(imgContrast), std(imgContrast));
            fprintf('\n');
        end
    end
end

%% =====================================================================
%% EXAMPLE USAGE
%% =====================================================================

% Example: Load and prepare RoboSense data
% 
% prep = RoboSenseDatasetPrep('path/to/robosense_data', ...
%                             'path/to/output');
% 
% % Load data
% [frames, labels, names] = prep.loadDataset(500);  % First 500 frames
% 
% % Preprocess
% frames_normalized = prep.preprocessFrames(frames);
% labels_resized = prep.resizeLabels(labels);
% 
% % Analyze
% prep.analyzeDataset(frames, labels);
% 
% % Visualize
% prep.visualizeFrames(frames(1:16), labels(1:16));
% 
% % Export for training
% prep.exportForTraining('robosense_dataset_export', ...
%                       frames_normalized, labels_resized);
