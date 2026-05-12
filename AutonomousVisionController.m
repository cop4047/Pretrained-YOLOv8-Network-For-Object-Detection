classdef AutonomousVisionController
    % AUTONOMOUSVISIONCONTROLLER
    % Comprehensive CV pipeline for UK autonomous vehicle navigation
    % Outputs driveable lanes, traffic state, and vehicle detections for MPC
    %
    % Key Features:
    % - ONNX-based lane segmentation (pre-trained, UK-compliant)
    % - Simulated YOLOv8 object detection (cars, pedestrians)
    % - HSV-based traffic light color classification
    % - UK Traffic Law enforcement (traffic lights, speed limits, lane rules)
    % - Real-time output for Model Predictive Control
    %
    % USAGE:
    %   visionCtrl = AutonomousVisionController(laneModelPath, yoloDetectPath, ...
    %                                           yoloTrafficPath, cameraCalibPath);
    %   output = visionCtrl.processFrame(rgbFrame);
    
    properties
        % Deep Learning Models
        laneSegmentationNet        % Loaded ONNX network for lane segmentation
        laneNetLoaded = false      % Flag for successful ONNX load
        
        % Camera calibration
        cameraParams               % intrinsic/extrinsic parameters
        
        % Processing parameters
        frameHeight = 1080
        frameWidth = 1920
        segmentationHeight = 384
        segmentationWidth = 384
        detectionConfidenceThreshold = 0.5
        
        % UK Traffic Law State
        trafficLightState          % 'red', 'green', 'yellow', 'unknown'
        speedLimitMPH              % Current speed limit
        isAllowedToDrive           % Boolean: safe to proceed
        
        % Lane tracking
        laneHistory                % Rolling buffer for lane smoothing
        laneHistorySize = 5
        
        % Performance monitoring
        processingTimes            % Frame processing timings
        detectionCounts            % Running tally of detections
        
        % Execution environment
        ExecutionEnvironment = 'auto'  % 'cpu', 'gpu', or 'auto'
        UseGPU = false
    end
    
    methods
        function obj = AutonomousVisionController(laneModelPath, yoloDetectPath, yoloTrafficPath, cameraCalibPath)
            % Initialize the vision controller
            %
            % Inputs:
            %   laneModelPath:    Path to ONNX lane segmentation model
            %   yoloDetectPath:   Path to YOLOv8 detection model (.pt or .onnx)
            %   yoloTrafficPath:  Path to YOLOv8 traffic model (.pt or .onnx)
            %   cameraCalibPath:  Path to camera calibration file (optional)
            
            disp('[AutonomousVisionController] Initializing CV pipeline...');
            
            % Check for GPU availability
            try
                gpuDevice;
                obj.UseGPU = canUseGPU;
            catch
                obj.UseGPU = false;
            end
            
            if obj.UseGPU
                disp('[Vision] GPU available - will use GPU acceleration');
            else
                disp('[Vision] GPU not available - using CPU');
            end
            
            % Load lane segmentation ONNX model
            disp('[Vision] Loading lane segmentation ONNX network...');
            try
                if isfile(laneModelPath)
                    % Try modern syntax first (R2023b+)
                    try
                        obj.laneSegmentationNet = importNetworkFromONNX(laneModelPath);
                    catch
                        % Fallback for older MATLAB versions
                        obj.laneSegmentationNet = importONNXNetwork(laneModelPath);
                    end
                    obj.laneNetLoaded = true;
                    disp('[Vision] ✓ Lane segmentation ONNX loaded successfully');
                else
                    warning('[Vision] Lane model file not found: %s', laneModelPath);
                    obj.laneNetLoaded = false;
                end
            catch ME
                warning('[Vision] Failed to load ONNX network: %s', ME.message);
                disp('[Vision] ⚠ Lane segmentation unavailable - will use fallback method');
                obj.laneNetLoaded = false;
            end
            
            % Load camera parameters if provided
            if nargin > 3 && ~isempty(cameraCalibPath) && isfile(cameraCalibPath)
                try
                    load(cameraCalibPath, 'cameraParams');
                    obj.cameraParams = cameraParams;
                    disp('[Vision] ✓ Camera calibration loaded');
                catch
                    disp('[Vision] ⚠ Could not load camera calibration');
                end
            end
            
            % Initialize UK traffic law state
            obj.trafficLightState = 'unknown';
            obj.speedLimitMPH = 30; % Default urban speed limit
            obj.isAllowedToDrive = false;
            
            % Initialize lane history for smoothing
            obj.laneHistory = cell(obj.laneHistorySize, 1);
            
            % Initialize performance tracking
            obj.processingTimes = [];
            obj.detectionCounts = struct('cars', 0, 'pedestrians', 0, 'trafficLights', 0);
            
            disp('[Vision] ✓ Initialization complete');
        end
        
        function output = processFrame(obj, rgbFrame)
            % Process a single video frame through the full pipeline
            %
            % Input:
            %   rgbFrame: RGB image (height × width × 3), uint8 or single
            %
            % Output:
            %   output: struct with fields:
            %       - laneSegmentation: Binary mask of driveable lane
            %       - laneCenterline: (x, y) coordinates of lane center
            %       - cars: cell array of detection structs {class, bbox, confidence}
            %       - pedestrians: cell array of detection structs
            %       - trafficLightColor: 'red'|'green'|'yellow'|'unknown'
            %       - speedLimit: Current speed limit (mph)
            %       - isAllowedToDrive: Boolean safety flag
            %       - processingTime: Execution time (ms)
            
            tic;
            
            % Validate input
            if ~ismatrix(rgbFrame) || size(rgbFrame, 3) ~= 3
                error('Input must be RGB image (height × width × 3)');
            end
            
            % =====================================================
            % STEP 1: Lane Segmentation
            % =====================================================
            [laneSegmentation, laneCenterline] = obj.segmentDrivableLane(rgbFrame);
            
            % =====================================================
            % STEP 2: Object Detection (Simulated YOLOv8)
            % =====================================================
            [carDetections, pedestrianDetections] = obj.detectObjects(rgbFrame);
            
            % =====================================================
            % STEP 3: Traffic Light Detection & Color Classification
            % =====================================================
            trafficLightColor = obj.classifyTrafficLightColor(rgbFrame);
            obj.trafficLightState = trafficLightColor;
            
            % =====================================================
            % STEP 4: UK Traffic Law Compliance Check
            % =====================================================
            [isAllowedToDrive, speedLimit] = obj.evaluateTrafficLawCompliance(...
                laneSegmentation, trafficLightColor, pedestrianDetections);
            obj.isAllowedToDrive = isAllowedToDrive;
            obj.speedLimitMPH = speedLimit;
            
            % =====================================================
            % STEP 5: Package Output for MPC
            % =====================================================
            processingTime = toc * 1000; % Convert to milliseconds
            obj.processingTimes = [obj.processingTimes, processingTime];
            
            output = struct(...
                'laneSegmentation', laneSegmentation, ...
                'laneCenterline', laneCenterline, ...
                'cars', {carDetections}, ...
                'pedestrians', {pedestrianDetections}, ...
                'trafficLightColor', trafficLightColor, ...
                'speedLimit', speedLimit, ...
                'isAllowedToDrive', isAllowedToDrive, ...
                'processingTime', processingTime);
        end
        
        function [laneSegmentation, laneCenterline] = segmentDrivableLane(obj, rgbFrame)
            % Segment the driveable lane using ONNX network or fallback method
            %
            % Returns:
            %   laneSegmentation: Binary mask (same size as input)
            %   laneCenterline: [x_center, y_center] of lane
            
            if obj.laneNetLoaded
                % Use ONNX network
                laneSegmentation = obj.segmentWithONNX(rgbFrame);
            else
                % Use fallback: simple white line detection via HSV
                laneSegmentation = obj.segmentWithHSV(rgbFrame);
            end
            
            % Extract lane centerline
            laneCenterline = obj.extractLaneCenterline(laneSegmentation);
            
            % Smooth centerline with history
            obj.laneHistory = [obj.laneHistory(2:end), {laneCenterline}];
            laneCenterline = obj.smoothLaneCenterline();
        end
        
        function segMask = segmentWithONNX(obj, rgbFrame)
            % Run lane segmentation via ONNX network
            
            try
                % Resize for network input
                imgResized = imresize(rgbFrame, [obj.segmentationHeight, obj.segmentationWidth]);
                
                % Normalize to single precision
                imgNorm = im2single(imgResized);
                
                % ImageNet normalization
                meanVals = reshape([0.485, 0.456, 0.406], [1, 1, 3]);
                stdVals = reshape([0.229, 0.224, 0.225], [1, 1, 3]);
                imgNorm = (imgNorm - meanVals) ./ stdVals;
                
                % Prepare input: add batch dimension [H, W, C, B]
                imgBatch = permute(imgNorm, [1, 2, 3]);
                imgBatch = repmat(imgBatch, [1, 1, 1, 1]);
                
                % Run prediction
                output = predict(obj.laneSegmentationNet, imgBatch);
                
                % Extract segmentation mask
                if iscell(output)
                    segMaskResized = output{1};
                else
                    segMaskResized = output;
                end
                
                % Handle different output formats
                if isdlarray(segMaskResized)
                    segMaskResized = extractdata(segMaskResized);
                end
                
                % Squeeze to 2D if needed
                segMaskResized = squeeze(segMaskResized);
                
                % Threshold to binary and resize back
                segMaskBinary = segMaskResized > 0.5;
                segMask = imresize(segMaskBinary, [size(rgbFrame, 1), size(rgbFrame, 2)]) > 0.5;
                
            catch ME
                warning('[Vision] ONNX inference failed: %s. Using HSV fallback.', ME.message);
                segMask = obj.segmentWithHSV(rgbFrame);
            end
        end
        
        function segMask = segmentWithHSV(obj, rgbFrame)
            % Fallback: detect bright white/light colors (lane markings)
            
            % Convert to HSV
            hsvImg = rgb2hsv(im2single(rgbFrame));
            
            % Extract channels
            S = hsvImg(:, :, 2);  % Saturation
            V = hsvImg(:, :, 3);  % Value
            
            % Lane markings are typically white (low saturation, high value)
            segMask = (S < 0.1) & (V > 0.7);
            
            % Morphological cleaning
            segMask = imclose(segMask, strel('disk', 5));
            segMask = imopen(segMask, strel('disk', 3));
        end
        
        function [carDetections, pedestrianDetections] = detectObjects(obj, rgbFrame)
            % Detect objects (cars, pedestrians) using simplified detection
            % In production: replace with actual YOLOv8 via Python
            %
            % For now: return empty detections (user can integrate real YOLOv8)
            % To use real YOLOv8:
            %   1. Export model to ONNX
            %   2. Create wrapper function: detectWithYOLO()
            %   3. Or use Python: pyrun("from ultralytics import YOLO; ...")
            
            carDetections = {};
            pedestrianDetections = {};
            
            % Placeholder: would call YOLOv8 here
            % Example structure:
            % carDetections{1} = struct('bbox', [x1 y1 x2 y2], 'confidence', 0.95, 'class', 'car');
            % pedestrianDetections{1} = struct('bbox', [x1 y1 x2 y2], 'confidence', 0.87, 'class', 'pedestrian');
        end
        
        function color = classifyTrafficLightColor(obj, rgbFrame)
            % Classify traffic light color using HSV color space
            %
            % Output:
            %   color: 'red', 'green', 'yellow', or 'unknown'
            
            % Convert to HSV
            if isa(rgbFrame, 'uint8')
                imgSingle = im2single(rgbFrame);
            else
                imgSingle = rgbFrame;
            end
            
            hsvImg = rgb2hsv(imgSingle);
            
            % Extract H, S, V channels (normalized [0, 1])
            H = hsvImg(:, :, 1);
            S = hsvImg(:, :, 2);
            V = hsvImg(:, :, 3);
            
            % Define color ranges
            % Red: H ~= [0-0.05, 0.95-1.0], high S, high V
            redMask = ((H < 0.05 | H > 0.95) & S > 0.3 & V > 0.3);
            
            % Yellow: H ~= [0.15-0.2], high S, high V
            yellowMask = (H > 0.10 & H < 0.20 & S > 0.3 & V > 0.3);
            
            % Green: H ~= [0.3-0.4], high S, high V
            greenMask = (H > 0.25 & H < 0.45 & S > 0.3 & V > 0.3);
            
            % Count pixels in each color
            redCount = sum(redMask(:));
            yellowCount = sum(yellowMask(:));
            greenCount = sum(greenMask(:));
            
            % Determine dominant color
            [maxCount, maxIdx] = max([redCount, yellowCount, greenCount]);
            
            if maxCount < 50  % Threshold for valid detection
                color = 'unknown';
            else
                switch maxIdx
                    case 1
                        color = 'red';
                    case 2
                        color = 'yellow';
                    case 3
                        color = 'green';
                    otherwise
                        color = 'unknown';
                end
            end
        end
        
        function [isAllowedToDrive, speedLimit] = evaluateTrafficLawCompliance(obj, laneSegmentation, trafficLightColor, pedestrianDetections)
            % Evaluate UK traffic law compliance
            %
            % UK Rules implemented:
            % 1. Must not drive on red traffic light
            % 2. Must not drive if pedestrians in lane
            % 3. Must stay within lane markings
            % 4. Speed limits enforced (default: 30 mph urban)
            %
            % Output:
            %   isAllowedToDrive: Boolean
            %   speedLimit: Current speed limit (mph)
            
            isAllowedToDrive = true;
            speedLimit = 30;  % Default UK urban speed limit
            
            % Rule 1: Traffic light state
            switch obj.trafficLightState
                case 'red'
                    isAllowedToDrive = false;
                    speedLimit = 0;
                    return;
                case 'yellow'
                    speedLimit = 10;
                    isAllowedToDrive = true;
                case 'green'
                    speedLimit = 30;
                    isAllowedToDrive = true;
                case 'unknown'
                    speedLimit = 15;
                    isAllowedToDrive = false;
            end
            
            % Rule 2: Check for pedestrians in lane
            if ~isempty(pedestrianDetections)
                isAllowedToDrive = false;
                speedLimit = 0;
            end
            
            % Rule 3: Lane coherence check
            laneArea = sum(laneSegmentation(:));
            maxArea = numel(laneSegmentation);
            if laneArea < 0.05 * maxArea  % Less than 5% of image is lane
                isAllowedToDrive = false;
                speedLimit = 0;
            end
        end
        
        function centerline = extractLaneCenterline(~, laneSegmentation)
            % Extract centerline from binary lane mask
            % Returns [x_center, y_center] of lane
            
            if ~any(laneSegmentation(:))
                % No lane detected
                centerline = [NaN, NaN];
                return;
            end
            
            % Find connected components
            CC = bwconncomp(laneSegmentation);
            if CC.NumObjects == 0
                centerline = [NaN, NaN];
                return;
            end
            
            % Get largest component (main lane)
            [~, idx] = max(cellfun(@length, CC.PixelIdxList));
            largestLane = CC.PixelIdxList{idx};
            
            % Get coordinates
            [y, x] = ind2sub(size(laneSegmentation), largestLane);
            
            % Simple centerline: mean position
            centerline = [mean(x), mean(y)];
        end
        
        function smoothed = smoothLaneCenterline(obj)
            % Smooth centerline using temporal history
            validHistory = obj.laneHistory(~cellfun(@isempty, obj.laneHistory));
            
            if isempty(validHistory)
                smoothed = [NaN, NaN];
                return;
            end
            
            positions = cell2mat(validHistory);
            smoothed = mean(positions, 1);
        end
        
        function visualizeOutput(obj, rgbFrame, output)
            % Visualize pipeline output for debugging
            figure('Name', 'Autonomous Vision Pipeline', 'NumberTitle', 'off', 'Position', [100, 100, 1400, 900]);
            
            % Original frame
            subplot(2, 3, 1);
            imshow(rgbFrame);
            title('Input Frame', 'FontSize', 12, 'FontWeight', 'bold');
            
            % Lane segmentation overlay
            subplot(2, 3, 2);
            imshow(rgbFrame);
            hold on;
            B = bwboundaries(output.laneSegmentation);
            for k = 1:length(B)
                boundary = B{k};
                plot(boundary(:, 2), boundary(:, 1), 'g', 'LineWidth', 2);
            end
            if ~isnan(output.laneCenterline(1))
                plot(output.laneCenterline(1), output.laneCenterline(2), 'r*', 'MarkerSize', 15);
            end
            title('Lane Segmentation', 'FontSize', 12, 'FontWeight', 'bold');
            hold off;
            
            % Traffic light state
            subplot(2, 3, 3);
            axh = gca;
            set(axh, 'Color', 'white');
            switch output.trafficLightColor
                case 'red'
                    tlColor = [1, 0, 0];
                case 'yellow'
                    tlColor = [1, 1, 0];
                case 'green'
                    tlColor = [0, 1, 0];
                otherwise
                    tlColor = [0.5, 0.5, 0.5];
            end
            rectangle(axh, 'Position', [0.2, 0.2, 0.6, 0.6], 'FaceColor', tlColor, 'EdgeColor', 'black', 'LineWidth', 2);
            text(0.5, 0.05, sprintf('Traffic: %s', upper(output.trafficLightColor)), ...
                'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
            axis off;
            xlim([0, 1]);
            ylim([0, 1]);
            
            % Drive allowance
            subplot(2, 3, 4);
            axh = gca;
            if output.isAllowedToDrive
                set(axh, 'Color', [0, 0.5, 0]);
                text_str = 'GO';
                text_color = 'white';
            else
                set(axh, 'Color', [1, 0, 0]);
                text_str = 'STOP';
                text_color = 'white';
            end
            text(0.5, 0.5, text_str, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                'FontSize', 28, 'FontWeight', 'bold', 'Color', text_color);
            axis off;
            xlim([0, 1]);
            ylim([0, 1]);
            
            % Speed limit
            subplot(2, 3, 5);
            axh = gca;
            set(axh, 'Color', 'white');
            text(0.5, 0.5, sprintf('%d mph', output.speedLimit), ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                'FontSize', 24, 'FontWeight', 'bold');
            text(0.5, 0.1, 'Speed Limit', 'HorizontalAlignment', 'center', 'FontSize', 10);
            axis off;
            xlim([0, 1]);
            ylim([0, 1]);
            
            % Processing stats
            subplot(2, 3, 6);
            axh = gca;
            set(axh, 'Color', 'white');
            stats_str = sprintf(['Processing Time: %.1f ms\n' ...
                                'FPS (estimated): %.1f\n' ...
                                'Cars Detected: %d\n' ...
                                'Pedestrians: %d'], ...
                output.processingTime, ...
                1000/max(output.processingTime, 1), ...
                length(output.cars), ...
                length(output.pedestrians));
            text(0.1, 0.5, stats_str, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', ...
                'FontSize', 11, 'FontFamily', 'monospaced');
            axis off;
            xlim([0, 1]);
            ylim([0, 1]);
        end
        
        function diagnostics = getPerformanceDiagnostics(obj)
            % Return performance metrics
            if isempty(obj.processingTimes)
                avgTime = 0;
                maxTime = 0;
                minTime = 0;
                fps = 0;
            else
                avgTime = mean(obj.processingTimes);
                maxTime = max(obj.processingTimes);
                minTime = min(obj.processingTimes);
                fps = 1000 / avgTime;
            end
            
            diagnostics = struct(...
                'avgProcessingTime_ms', avgTime, ...
                'maxProcessingTime_ms', maxTime, ...
                'minProcessingTime_ms', minTime, ...
                'estimatedFPS', fps, ...
                'detectionCounts', obj.detectionCounts, ...
                'laneNetworkLoaded', obj.laneNetLoaded, ...
                'gpuAvailable', obj.UseGPU);
        end
    end
end

% =========================================================================
% HELPER FUNCTION: RGB to HSV conversion
% =========================================================================
function hsvImg = rgb2hsv(rgbImg)
    % Convert RGB image to HSV color space
    % Input: RGB image with values in [0, 1]
    % Output: HSV image with H, S, V in [0, 1]
    
    r = rgbImg(:, :, 1);
    g = rgbImg(:, :, 2);
    b = rgbImg(:, :, 3);
    
    % Max and min values
    maxC = max(cat(3, r, g, b), [], 3);
    minC = min(cat(3, r, g, b), [], 3);
    
    % Value
    v = maxC;
    
    % Saturation
    delta = maxC - minC;
    s = zeros(size(maxC));
    mask = maxC > 0;
    s(mask) = delta(mask) ./ maxC(mask);
    
    % Hue
    h = zeros(size(maxC));
    
    % Red is dominant
    mask_r = (maxC == r) & (delta > 0);
    h(mask_r) = mod((g(mask_r) - b(mask_r)) ./ delta(mask_r), 6) / 6;
    
    % Green is dominant
    mask_g = (maxC == g) & (delta > 0);
    h(mask_g) = ((b(mask_g) - r(mask_g)) ./ delta(mask_g) + 2) / 6;
    
    % Blue is dominant
    mask_b = (maxC == b) & (delta > 0);
    h(mask_b) = ((r(mask_b) - g(mask_b)) ./ delta(mask_b) + 4) / 6;
    
    % Combine into HSV image
    hsvImg = cat(3, h, s, v);
end