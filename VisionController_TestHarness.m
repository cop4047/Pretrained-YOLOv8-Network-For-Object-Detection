%% =====================================================================
%% AUTONOMOUS VEHICLE VISION CONTROLLER - MAIN LOOP
%% =====================================================================
%% 
%% This script demonstrates the complete autonomous driving vision pipeline
%% Works in MATLAB with or without ONNX models
%% 
%% Modes:
%% 1. Live camera (if available)
%% 2. Test video file
%% 3. Synthetic test frames
%%

clear; clc; close all;

%% =====================================================================
%% CONFIGURATION
%% =====================================================================

% Model paths (adjust to your setup)
LANE_MODEL_PATH = './lane_segmentation.onnx';      % Your ONNX model path
YOLO_DETECT_PATH = '';                              % YOLOv8 detection model
YOLO_TRAFFIC_PATH = '';                             % YOLOv8 traffic model
CAMERA_CALIB_PATH = '';                             % Camera calibration

% Processing parameters
INPUT_MODE = 'synthetic';  % Options: 'camera', 'video', 'synthetic'
VIDEO_FILE = 'autonomous_drive.mp4';  % If INPUT_MODE = 'video'

% Output options
SAVE_RESULTS = false;
OUTPUT_DIR = './results';
VISUALIZE = true;
VERBOSE = true;

% Create output directory
if SAVE_RESULTS
    mkdir(OUTPUT_DIR);
end

%% =====================================================================
%% INITIALIZE VISION CONTROLLER
%% =====================================================================

fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('AUTONOMOUS VEHICLE VISION SYSTEM\n');
fprintf('%s\n', repmat('=', 1, 70));

% Create controller
visionCtrl = AutonomousVisionController(...
    LANE_MODEL_PATH, ...
    YOLO_DETECT_PATH, ...
    YOLO_TRAFFIC_PATH, ...
    CAMERA_CALIB_PATH);

fprintf('\n✓ Vision controller initialized\n');

%% =====================================================================
%% SETUP INPUT SOURCE
%% =====================================================================

fprintf('\nSetting up input source: %s\n', upper(INPUT_MODE));

switch lower(INPUT_MODE)
    case 'camera'
        % Live camera input
        try
            cam = webcam;
            getImageSource = @() snapshot(cam);
            numFrames = 300;  % Process 300 frames
            fprintf('✓ Camera initialized (will process %d frames)\n', numFrames);
        catch ME
            warning('Camera not available: %s', ME.message);
            fprintf('Falling back to synthetic test frames\n');
            INPUT_MODE = 'synthetic';
        end
        
    case 'video'
        % Video file input
        if isfile(VIDEO_FILE)
            vr = VideoReader(VIDEO_FILE);
            numFrames = vr.NumFrames;
            fprintf('✓ Video loaded: %s (%d frames)\n', VIDEO_FILE, numFrames);
            frameIdx = 0;
            getImageSource = @() readFrame(vr);
        else
            warning('Video file not found: %s', VIDEO_FILE);
            fprintf('Falling back to synthetic test frames\n');
            INPUT_MODE = 'synthetic';
        end
        
    case 'synthetic'
        % Synthetic test frames
        fprintf('✓ Generating synthetic test frames\n');
        numFrames = 30;
end

%% =====================================================================
%% MAIN PROCESSING LOOP
%% =====================================================================

fprintf('\nStarting main processing loop...\n');
fprintf('%s\n\n', repmat('-', 1, 70));

% Storage for results
results = struct();
results.frameNum = [];
results.trafficLight = {};
results.speedLimit = [];
results.allowedToDrive = [];
results.processingTime = [];

% Timing variables
frameCount = 0;
startTime = tic;

% Main loop
for frameIdx = 1:numFrames
    
    % Get image frame
    if strcmp(INPUT_MODE, 'synthetic')
        % Generate synthetic test frame
        rgbFrame = generateTestFrame(frameIdx, numFrames);
    else
        % Get frame from source
        try
            rgbFrame = getImageSource();
        catch
            fprintf('Reached end of input stream\n');
            break;
        end
    end
    
    % Ensure RGB format
    if size(rgbFrame, 3) == 1
        rgbFrame = repmat(rgbFrame, [1, 1, 3]);
    end
    
    % Process frame
    output = visionCtrl.processFrame(rgbFrame);
    frameCount = frameCount + 1;
    
    % Store results
    results.frameNum = [results.frameNum, frameIdx];
    results.trafficLight{frameIdx} = output.trafficLightColor;
    results.speedLimit = [results.speedLimit, output.speedLimit];
    results.allowedToDrive = [results.allowedToDrive, output.isAllowedToDrive];
    results.processingTime = [results.processingTime, output.processingTime];
    
    % Print progress
    if VERBOSE && mod(frameIdx, 5) == 0
        fprintf('[Frame %3d / %d] TL: %-8s | Speed: %3d mph | %s | %.1f ms\n', ...
            frameIdx, numFrames, ...
            output.trafficLightColor, ...
            output.speedLimit, ...
            iif(output.isAllowedToDrive, 'GO   ', 'STOP '), ...
            output.processingTime);
    end
    
    % Visualize
    if VISUALIZE && mod(frameIdx, 1) == 0  % Show every frame
        visionCtrl.visualizeOutput(rgbFrame, output);
        drawnow;
        pause(0.01);  % Small pause for visualization
    end
    
    % Save frame results if requested
    if SAVE_RESULTS
        savefig(fullfile(OUTPUT_DIR, sprintf('frame_%04d.fig', frameIdx)));
        close all;
    end
end

% Calculate elapsed time
elapsedTime = toc(startTime);

%% =====================================================================
%% RESULTS SUMMARY
%% =====================================================================

fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('PROCESSING SUMMARY\n');
fprintf('%s\n\n', repmat('=', 1, 70));

% Get diagnostics
diag = visionCtrl.getPerformanceDiagnostics();

fprintf('Frames Processed:          %d\n', frameCount);
fprintf('Total Time:                %.2f seconds\n', elapsedTime);
fprintf('Average FPS:               %.2f\n', frameCount / elapsedTime);
fprintf('Avg Processing Time/Frame: %.2f ms\n', diag.avgProcessingTime_ms);
fprintf('Min Processing Time:       %.2f ms\n', diag.minProcessingTime_ms);
fprintf('Max Processing Time:       %.2f ms\n', diag.maxProcessingTime_ms);
fprintf('\n');

% Traffic light statistics
fprintf('TRAFFIC LIGHT STATISTICS:\n');
fprintf('--------------------------\n');
[trafficCounts, trafficStates] = hist(categorical(results.trafficLight), ...
    {'red', 'yellow', 'green', 'unknown'});
for i = 1:length(trafficStates)
    pct = 100 * trafficCounts(i) / frameCount;
    fprintf('  %s:  %4d frames (%5.1f%%)\n', ...
        char(trafficStates(i)), trafficCounts(i), pct);
end

fprintf('\n');

% Speed limit statistics
fprintf('SPEED LIMIT DISTRIBUTION:\n');
fprintf('--------------------------\n');
fprintf('  Avg Speed Limit: %.1f mph\n', mean(results.speedLimit));
fprintf('  Min Speed Limit: %d mph\n', min(results.speedLimit));
fprintf('  Max Speed Limit: %d mph\n', max(results.speedLimit));
fprintf('  Stopped (0 mph): %d frames (%.1f%%)\n', ...
    sum(results.speedLimit == 0), 100*sum(results.speedLimit == 0)/frameCount);

fprintf('\n');

% Drive allowance statistics
fprintf('DRIVE ALLOWANCE:\n');
fprintf('--------------------------\n');
allowed = sum(results.allowedToDrive);
fprintf('  Allowed to Drive: %d frames (%.1f%%)\n', allowed, 100*allowed/frameCount);
fprintf('  Must Stop:        %d frames (%.1f%%)\n', frameCount-allowed, 100*(frameCount-allowed)/frameCount);

fprintf('\n');

% Network status
fprintf('SYSTEM STATUS:\n');
fprintf('--------------------------\n');
fprintf('  Lane Network Loaded:    %s\n', iif(diag.laneNetworkLoaded, '✓ Yes', '✗ No (using HSV fallback)'));
fprintf('  GPU Available:          %s\n', iif(diag.gpuAvailable, '✓ Yes', '✗ No'));

fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('Processing complete!\n');
fprintf('%s\n\n', repmat('=', 1, 70));

%% =====================================================================
%% PLOT RESULTS
%% =====================================================================

figure('Name', 'Processing Results', 'NumberTitle', 'off', 'Position', [100, 100, 1200, 800]);

% Processing time over frames
subplot(2, 2, 1);
plot(results.frameNum, results.processingTime, 'b-', 'LineWidth', 1.5);
hold on;
yline(diag.avgProcessingTime_ms, 'r--', 'LineWidth', 1.5);
xlabel('Frame Number');
ylabel('Processing Time (ms)');
title('Processing Time per Frame');
grid on;
legend('Time', 'Average');

% Speed limit over frames
subplot(2, 2, 2);
plot(results.frameNum, results.speedLimit, 'g-', 'LineWidth', 2);
xlabel('Frame Number');
ylabel('Speed Limit (mph)');
title('Speed Limit Over Time');
grid on;
ylim([-5, 40]);

% Traffic light state transitions
subplot(2, 2, 3);
trafficLightNums = zeros(size(results.trafficLight));
for i = 1:length(results.trafficLight)
    switch results.trafficLight{i}
        case 'red'
            trafficLightNums(i) = 1;
        case 'yellow'
            trafficLightNums(i) = 2;
        case 'green'
            trafficLightNums(i) = 3;
        otherwise
            trafficLightNums(i) = 0;
    end
end
plot(results.frameNum, trafficLightNums, 'o-', 'LineWidth', 2, 'MarkerSize', 4);
set(gca, 'YTick', [0, 1, 2, 3]);
set(gca, 'YTickLabel', {'Unknown', 'Red', 'Yellow', 'Green'});
xlabel('Frame Number');
ylabel('Traffic Light State');
title('Traffic Light Detection Over Time');
grid on;

% Drive allowance
subplot(2, 2, 4);
allowanceNums = results.allowedToDrive;
area(results.frameNum, allowanceNums, 'FaceColor', [0, 0.5, 0], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
hold on;
area(results.frameNum, 1-allowanceNums, 'FaceColor', [1, 0, 0], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
set(gca, 'YTick', [0, 1]);
set(gca, 'YTickLabel', {'Stop', 'Go'});
xlabel('Frame Number');
ylabel('Drive Status');
title('Drive Allowance Over Time');
grid on;
ylim([-0.1, 1.1]);

sgtitle(sprintf('Vision Pipeline Results (%d frames)', frameCount), ...
    'FontSize', 14, 'FontWeight', 'bold');

%% =====================================================================
%% SAVE RESULTS
%% =====================================================================

if SAVE_RESULTS
    % Save results table
    resultsTable = table(...
        results.frameNum', ...
        cellfun(@char, results.trafficLight, 'UniformOutput', false)', ...
        results.speedLimit', ...
        results.allowedToDrive', ...
        results.processingTime', ...
        'VariableNames', {'Frame', 'TrafficLight', 'SpeedLimit_mph', 'AllowedToDrive', 'ProcessingTime_ms'});
    
    writetable(resultsTable, fullfile(OUTPUT_DIR, 'results.csv'));
    fprintf('✓ Results saved to: %s\n', fullfile(OUTPUT_DIR, 'results.csv'));
end

fprintf('\nVision system test completed successfully!\n\n');

%% =====================================================================
%% HELPER FUNCTIONS
%% =====================================================================

function frame = generateTestFrame(frameIdx, totalFrames)
    % Generate synthetic test frame for demonstration
    % Includes lane markings, traffic light simulation, etc.
    
    height = 480;
    width = 640;
    
    % Create base road (gray)
    frame = uint8(ones(height, width, 3) * 80);
    
    % Add lane markings (white)
    laneY = round(height/2);
    frame(laneY-10:laneY+10, :, :) = 255;
    
    % Add road boundaries
    frame(1:20, :, :) = 60;
    frame(height-20:height, :, :) = 60;
    
    % Simulate traffic light in corner (changes over time)
    tlX = 50:100;
    tlY = 50:100;
    phase = mod(frameIdx, 30);
    if phase < 10
        % Red
        frame(tlY, tlX, 1) = 255;  % Red channel
        frame(tlY, tlX, 2:3) = 100;
    elseif phase < 20
        % Yellow
        frame(tlY, tlX, 1:2) = 255;  % Red + Green
        frame(tlY, tlX, 3) = 100;
    else
        % Green
        frame(tlY, tlX, 1) = 100;
        frame(tlY, tlX, 2) = 255;  % Green channel
        frame(tlY, tlX, 3) = 100;
    end
    
    % Add some variation for visual interest
    noise = uint8(randn(height, width, 3) * 5);
    frame = uint8(double(frame) + noise);
    frame = max(min(frame, 255), 0);
end

function out = iif(condition, trueVal, falseVal)
    % Inline if-else helper function
    if condition
        out = trueVal;
    else
        out = falseVal;
    end
end