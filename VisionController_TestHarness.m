%% AUTONOMOUS VISION CONTROLLER - STANDALONE TEST HARNESS
% Test and validate the vision pipeline independent of Simulink
% Useful for debugging, performance profiling, and validation

clear; clc; close all;

%% =====================================================================
%% TEST CONFIGURATION
%% =====================================================================

TEST_MODE = 'video';  % 'video', 'image_sequence', 'synthetic', or 'single_image'

% YOLOv8 detection model (your custom trained one)
yolo_custom_pt = fullfile(pwd, 'runs/detect/train8/weights/best.pt');

% Pre-trained YOLOv8 for traffic (download separately)
yolo_traffic_pt = fullfile(pwd,'models\yolov8n.mat');  % or yolov8m.pt

% Lane segmentation (created via Python script)
lane_segmentation_onnx = fullfile(pwd, 'onnx_models/lane_segmentation.onnx');

% Directories
results_dir = fullfile(pwd, 'results');

fprintf('YOLOv8 custom model: %s\n', yolo_custom_pt);
fprintf('YOLOv8 traffic model: %s\n', yolo_traffic_pt);
fprintf('Lane segmentation ONNX: %s\n', lane_segmentation_onnx);
LANE_ONNX_PATH = 'C:\Users\alros\Documents\GitHub\Pretrained-YOLOv8-Network-For-Object-Detection\onnx_models\lane_segmentation.onnx';
YOLO_DETECT_PATH = 'C:\Users\alros\Documents\GitHub\Pretrained-YOLOv8-Network-For-Object-Detection\runs\detect\train8\weights\best.pt';
YOLO_TRAFFIC_PATH = 'C:\Users\alros\Documents\GitHub\Pretrained-YOLOv8-Network-For-Object-Detection\models\yolov8n.mat';  % Download this pre-trained model
%% =====================================================================
%% INITIALIZE VISION CONTROLLER
%% =====================================================================

fprintf('Initializing vision controller...\n');

try
    visionCtrl = AutonomousVisionController(...
        LANE_ONNX_PATH, ...
        YOLO_DETECT_PATH, ...
        YOLO_TRAFFIC_PATH, ...
        []);
    fprintf('✓ Vision controller initialized\n\n');
catch ME
    fprintf('✗ Failed to initialize: %s\n', ME.message);
    return;
end

%% =====================================================================
%% SELECT TEST MODE
%% =====================================================================

switch TEST_MODE
    
    %% =========================================================
    %% TEST 1: SINGLE IMAGE
    %% =========================================================
    case 'single_image'
        fprintf('=== TEST 1: SINGLE IMAGE ===\n\n');
        
        if ~isfile(SINGLE_IMAGE)
            fprintf('Image not found: %s\n', SINGLE_IMAGE);
            fprintf('Creating synthetic test image...\n');
            frame = createSyntheticFrame(1080, 1920);
        else
            frame = imread(SINGLE_IMAGE);
        end
        
        % Process
        fprintf('Processing frame...\n');
        output = visionCtrl.processFrame(frame);
        
        % Display results
        fprintf('\nResults:\n');
        fprintf('  Lane segmentation: %dx%d binary mask\n', ...
            size(output.laneSegmentation, 1), size(output.laneSegmentation, 2));
        fprintf('  Lane centerline: [%.1f, %.1f]\n', ...
            output.laneCenterline(1), output.laneCenterline(2));
        fprintf('  Traffic light: %s\n', output.trafficLightColor);
        fprintf('  Speed limit: %d mph\n', output.speedLimit);
        fprintf('  Allowed to drive: %d\n', output.isAllowedToDrive);
        fprintf('  Processing time: %.2f ms\n\n', output.processingTime);
        
        % Visualize
        visionCtrl.visualizeOutput(frame, output);
        
    %% =========================================================
    %% TEST 2: VIDEO FILE
    %% =========================================================
    case 'video'
        fprintf('=== TEST 2: VIDEO FILE ===\n\n');
        VIDEO_PATH = "D:\Downloads\SensorDetectionSimulation.mp4";
        if ~isfile(VIDEO_PATH)
            fprintf('Video not found: %s\n', VIDEO_PATH);
            fprintf('Skipping video test\n');
            return;
        end
        
        % Open video
        videoReader = VideoReader(VIDEO_PATH);
        totalFrames = round(videoReader.Duration * videoReader.FrameRate);
        
        fprintf('Video: %s\n', VIDEO_PATH);
        fprintf('Frames: %d\n', totalFrames);
        fprintf('Resolution: %dx%d\n', videoReader.Width, videoReader.Height);
        fprintf('FPS: %.1f\n\n', videoReader.FrameRate);
        
        % Test on first 100 frames
        maxFrames = min(100, totalFrames);
        
        % Storage for results
        results = struct(...
            'frameIdx', [], ...
            'trafficLight', {}, ...
            'speedLimit', [], ...
            'allowedToDrive', [], ...
            'processingTime', [], ...
            'carsDetected', [], ...
            'pedestriansDetected', []);
        
        % Process frames
        fprintf('Processing frames...\n');
        for i = 1:maxFrames
            if hasFrame(videoReader)
                frame = readFrame(videoReader);
                output = visionCtrl.processFrame(frame);
                
                % Store results
                results.frameIdx(i) = i;
                results.trafficLight{i} = output.trafficLightColor;
                results.speedLimit(i) = output.speedLimit;
                results.allowedToDrive(i) = output.isAllowedToDrive;
                results.processingTime(i) = output.processingTime;
                results.carsDetected(i) = length(output.cars);
                results.pedestriansDetected(i) = length(output.pedestrians);
                
                if mod(i, 10) == 0
                    fprintf('  Frame %d/%d (%.1f FPS)\n', i, maxFrames, ...
                        mean(1000./results.processingTime(1:i)));
                end
            else
                break;
            end
        end
        
        % Statistics
        fprintf('\n=== Video Processing Results ===\n\n');
        fprintf('Frames processed: %d\n', i-1);
        fprintf('Avg processing time: %.2f ms\n', mean(results.processingTime));
        fprintf('Max processing time: %.2f ms\n', max(results.processingTime));
        fprintf('Min processing time: %.2f ms\n', min(results.processingTime));
        fprintf('Avg FPS: %.1f\n', 1000/mean(results.processingTime));
        fprintf('\nTraffic light distribution:\n');
        tl_counts = countTrafficLights(results.trafficLight);
        fprintf('  Red: %d frames\n', tl_counts.red);
        fprintf('  Green: %d frames\n', tl_counts.green);
        fprintf('  Yellow: %d frames\n', tl_counts.yellow);
        fprintf('  Unknown: %d frames\n', tl_counts.unknown);
        fprintf('\nSafety violations:\n');
        fprintf('  Frames NOT allowed to drive: %d (%.1f%%)\n', ...
            sum(~results.allowedToDrive), ...
            100*sum(~results.allowedToDrive)/(i-1));
        fprintf('  Avg cars detected: %.1f\n', mean(results.carsDetected));
        fprintf('  Avg pedestrians detected: %.1f\n', mean(results.pedestriansDetected));
        
        % Save results
        saveTestResults(results, LOG_FILE);
        
        % Visualization
        plotVideoResults(results);
        
    %% =========================================================
    %% TEST 3: IMAGE SEQUENCE
    %% =========================================================
    case 'image_sequence'
        fprintf('=== TEST 3: IMAGE SEQUENCE ===\n\n');
        
        imageFiles = dir(fullfile(IMAGE_DIR, '*.jpg'));
        imageFiles = [imageFiles; dir(fullfile(IMAGE_DIR, '*.png'))];
        
        if isempty(imageFiles)
            fprintf('No images found in: %s\n', IMAGE_DIR);
            return;
        end
        
        fprintf('Found %d images\n\n', length(imageFiles));
        
        % Storage
        results = struct(...
            'filename', {}, ...
            'trafficLight', {}, ...
            'speedLimit', [], ...
            'allowedToDrive', [], ...
            'processingTime', []);
        
        % Process each image
        for i = 1:length(imageFiles)
            imagePath = fullfile(imageFiles(i).folder, imageFiles(i).name);
            frame = imread(imagePath);
            output = visionCtrl.processFrame(frame);
            
            results.filename{i} = imageFiles(i).name;
            results.trafficLight{i} = output.trafficLightColor;
            results.speedLimit(i) = output.speedLimit;
            results.allowedToDrive(i) = output.isAllowedToDrive;
            results.processingTime(i) = output.processingTime;
            
            fprintf('[%d/%d] %s\n', i, length(imageFiles), imageFiles(i).name);
            fprintf('      Traffic: %s | Speed: %d mph | Drive: %d\n', ...
                output.trafficLightColor, output.speedLimit, output.isAllowedToDrive);
        end
        
        % Summary
        fprintf('\n=== Image Sequence Results ===\n\n');
        fprintf('Images processed: %d\n', length(imageFiles));
        fprintf('Avg processing: %.2f ms\n', mean(results.processingTime));
        fprintf('Violations: %d\n', sum(~results.allowedToDrive));
        
        saveTestResults(results, LOG_FILE);
        
    %% =========================================================
    %% TEST 4: SYNTHETIC FRAMES
    %% =========================================================
    case 'synthetic'
        fprintf('=== TEST 4: SYNTHETIC FRAME TEST ===\n\n');
        
        numFrames = 50;
        
        results = struct(...
            'frameIdx', 1:numFrames, ...
            'trafficLight', {}, ...
            'speedLimit', [], ...
            'allowedToDrive', [], ...
            'processingTime', []);
        
        fprintf('Generating %d synthetic frames...\n\n', numFrames);
        
        for i = 1:numFrames
            % Create synthetic frame with varying conditions
            frame = createSyntheticFrame(1080, 1920);
            output = visionCtrl.processFrame(frame);
            
            results.trafficLight{i} = output.trafficLightColor;
            results.speedLimit(i) = output.speedLimit;
            results.allowedToDrive(i) = output.isAllowedToDrive;
            results.processingTime(i) = output.processingTime;
            
            if mod(i, 10) == 0
                fprintf('Frame %d/%d - FPS: %.1f\n', i, numFrames, ...
                    1000/mean(results.processingTime(1:i)));
            end
        end
        
        % Summary
        fprintf('\n=== Synthetic Frame Test Results ===\n\n');
        fprintf('Frames: %d\n', numFrames);
        fprintf('Avg processing: %.2f ms\n', mean(results.processingTime));
        fprintf('Max processing: %.2f ms\n', max(results.processingTime));
        fprintf('Min processing: %.2f ms\n', min(results.processingTime));
        fprintf('Avg FPS: %.1f\n', 1000/mean(results.processingTime));
        
        saveTestResults(results, LOG_FILE);
        
        % Plot
        plotSyntheticResults(results);
end

%% =====================================================================
%% PERFORMANCE DIAGNOSTICS
%% =====================================================================

fprintf('\n=== Vision Controller Diagnostics ===\n\n');
diag = visionCtrl.getPerformanceDiagnostics();
fprintf('Avg processing time: %.2f ms\n', diag.avgProcessingTime);
fprintf('FPS achieved: %.1f\n', diag.fps);
fprintf('Total cars detected: %d\n', diag.detectionCounts.cars);
fprintf('Total pedestrians: %d\n', diag.detectionCounts.pedestrians);

fprintf('\n✓ Test complete. Results saved to: %s\n', RESULTS_DIR);

%% =====================================================================
%% HELPER FUNCTIONS
%% =====================================================================

function frame = createSyntheticFrame(height, width)
    % Create a synthetic test frame (road with lane markings)
    frame = 255 * ones(height, width, 3, 'uint8');
    
    % Add road texture
    frame(height/2:end, :, :) = 100;
    
    % Add lane markings (white dashed lines)
    lane_y = height/2 + [0:50:height/2];
    lane_x1 = round(width * 0.35);
    lane_x2 = round(width * 0.65);
    
    for y = lane_y
        if y < height
            frame(y:min(y+10, height), lane_x1-5:lane_x1+5, :) = 255;
            frame(y:min(y+10, height), lane_x2-5:lane_x2+5, :) = 255;
        end
    end
    
    % Add some noise
    frame = frame + int16(randn(height, width, 3) * 5);
    frame = uint8(max(0, min(255, double(frame))));
end

function tl_counts = countTrafficLights(tl_array)
    tl_counts.red = sum(strcmp(tl_array, 'red'));
    tl_counts.green = sum(strcmp(tl_array, 'green'));
    tl_counts.yellow = sum(strcmp(tl_array, 'yellow'));
    tl_counts.unknown = sum(strcmp(tl_array, 'unknown'));
end

function saveTestResults(results, logPath)
    % Save test results to file
    fid = fopen(logPath, 'w');
    fprintf(fid, '=== VISION CONTROLLER TEST RESULTS ===\n\n');
    fprintf(fid, 'Timestamp: %s\n', datetime('now'));
    fprintf(fid, 'Frames processed: %d\n\n', length(results.frameIdx));
    
    if isfield(results, 'processingTime') && ~isempty(results.processingTime)
        fprintf(fid, 'Processing Time (ms):\n');
        fprintf(fid, '  Avg: %.2f\n', mean(results.processingTime));
        fprintf(fid, '  Max: %.2f\n', max(results.processingTime));
        fprintf(fid, '  Min: %.2f\n', min(results.processingTime));
        fprintf(fid, '  FPS: %.1f\n\n', 1000/mean(results.processingTime));
    end
    
    fprintf(fid, 'Traffic Light Distribution:\n');
    tl_counts = countTrafficLights(results.trafficLight);
    fprintf(fid, '  Red: %d\n', tl_counts.red);
    fprintf(fid, '  Green: %d\n', tl_counts.green);
    fprintf(fid, '  Yellow: %d\n', tl_counts.yellow);
    fprintf(fid, '  Unknown: %d\n\n', tl_counts.unknown);
    
    fprintf(fid, 'Safety:\n');
    fprintf(fid, '  Allowed to drive: %d frames\n', sum(results.allowedToDrive));
    fprintf(fid, '  NOT allowed to drive: %d frames\n', sum(~results.allowedToDrive));
    
    fclose(fid);
    fprintf('Results saved to: %s\n', logPath);
end

function plotVideoResults(results)
    figure('Name', 'Video Test Results', 'NumberTitle', 'off');
    
    % Processing time
    subplot(2, 2, 1);
    plot(results.frameIdx, results.processingTime, 'LineWidth', 1.5);
    ylabel('Processing time (ms)');
    xlabel('Frame');
    title('Processing Time per Frame');
    grid on;
    
    % Speed limit over time
    subplot(2, 2, 2);
    plot(results.frameIdx, results.speedLimit, 'LineWidth', 1.5);
    ylabel('Speed limit (mph)');
    xlabel('Frame');
    title('Speed Limit Variation');
    grid on;
    
    % Traffic light timeline
    subplot(2, 2, 3);
    tl_numeric = zeros(1, length(results.trafficLight));
    for i = 1:length(results.trafficLight)
        switch results.trafficLight{i}
            case 'red'
                tl_numeric(i) = 3;
            case 'yellow'
                tl_numeric(i) = 2;
            case 'green'
                tl_numeric(i) = 1;
            otherwise
                tl_numeric(i) = 0;
        end
    end
    plot(results.frameIdx, tl_numeric, 'o', 'MarkerSize', 3);
    set(gca, 'YTick', [0 1 2 3]);
    set(gca, 'YTickLabel', {'Unknown', 'Green', 'Yellow', 'Red'});
    xlabel('Frame');
    title('Traffic Light State');
    grid on;
    
    % Drive permission
    subplot(2, 2, 4);
    plot(results.frameIdx, results.allowedToDrive, 'LineWidth', 1.5);
    set(gca, 'YLim', [-0.1 1.1]);
    set(gca, 'YTick', [0 1]);
    set(gca, 'YTickLabel', {'STOP', 'GO'});
    xlabel('Frame');
    title('Drive Permission');
    grid on;
end

function plotSyntheticResults(results)
    figure('Name', 'Synthetic Frame Test', 'NumberTitle', 'off');
    
    subplot(1, 2, 1);
    plot(results.frameIdx, results.processingTime, '-o', 'LineWidth', 2, 'MarkerSize', 4);
    xlabel('Frame');
    ylabel('Processing time (ms)');
    title('Processing Time (Synthetic Frames)');
    grid on;
    
    subplot(1, 2, 2);
    histogram(results.processingTime, 20);
    xlabel('Processing time (ms)');
    ylabel('Count');
    title('Processing Time Distribution');
    grid on;
end
