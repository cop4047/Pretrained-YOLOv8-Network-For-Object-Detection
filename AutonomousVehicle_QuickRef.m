%% QUICK REFERENCE: AUTONOMOUS VEHICLE CV PIPELINE
%% UK Traffic Law Compliance + MATLAB MPC Integration
%%
%% This is a condensed guide for quick lookups
%% For detailed setup, see: AutonomousVehicle_Setup.m

%% =====================================================================
%% 1. QUICK START (5 minutes)
%% =====================================================================

% Initialize vision controller
visionCtrl = AutonomousVisionController(...
    'models/onnx/lane_segmentation.onnx', ...
    'models/yolov8_custom.pt', ...
    'models/yolov8_traffic.pt', ...
    []);

% Process a single frame
frame = imread('test_image.jpg');
output = visionCtrl.processFrame(frame);

% Check if allowed to drive
if output.isAllowedToDrive
    fprintf('Speed limit: %d mph\n', output.speedLimit);
    fprintf('Traffic light: %s\n', output.trafficLightColor);
else
    fprintf('STOP - Not allowed to drive\n');
end

%% =====================================================================
%% 2. KEY OUTPUTS STRUCTURE
%% =====================================================================

% output contains:
%  .laneSegmentation       - Binary mask (height × width)
%  .laneCenterline         - [x_center, y_center] coordinates
%  .cars                   - Bounding boxes of detected cars
%  .pedestrians            - Bounding boxes of pedestrians
%  .trafficLights          - Traffic light detections
%  .trafficLightColor      - 'red' | 'green' | 'yellow' | 'unknown'
%  .speedLimit             - Speed limit in mph
%  .isAllowedToDrive       - Boolean safety flag
%  .processingTime         - Frame time in milliseconds

%% =====================================================================
%% 3. UK TRAFFIC LAW RULES (Implemented)
%% =====================================================================

% Rule 1: Traffic Lights
%   RED    → isAllowedToDrive = false, speedLimit = 0
%   YELLOW → isAllowedToDrive = true,  speedLimit = 10
%   GREEN  → isAllowedToDrive = true,  speedLimit = 30
%   UNKNOWN→ isAllowedToDrive = false, speedLimit = 0 (conservative)

% Rule 2: Pedestrians
%   IF pedestrians_in_lane → isAllowedToDrive = false

% Rule 3: Lane Coherence
%   IF lane_area < 10% → isAllowedToDrive = false

% Rule 4: Speed Limits (Defaults)
%   Urban (default)    : 30 mph
%   Dual carriageway   : 60 mph
%   Motorway           : 70 mph
%   Residential/School : 20 mph

%% =====================================================================
%% 4. TRAFFIC LIGHT COLOR DETECTION (HSV Ranges)
%% =====================================================================

% RED:    H ∈ [0-10°, 170-180°], S > 100, V > 100
% YELLOW: H ∈ [20-30°],          S > 100, V > 100
% GREEN:  H ∈ [40-80°],          S > 100, V > 100

% Note: HSV in MATLAB
%   H: 0-1 (0° = red, 0.33 = green, 0.67 = blue)
%   S: 0-1 (0 = gray, 1 = saturated)
%   V: 0-1 (0 = black, 1 = bright)

%% =====================================================================
%% 5. LANE SEGMENTATION MODEL SPECS
%% =====================================================================

% Architecture: FCNN (Fully Convolutional Neural Network)
% Backbone: VGG16 (pre-trained ImageNet)
% Input size: 384×384×3
% Output size: 384×384×1 (binary mask)
% Expected accuracy: 95.62% (from paper)
% Processing time: 10-30 ms (GPU), 50-100 ms (CPU)

% Training details:
% - Optimizer: Adam
% - Loss function: Binary cross-entropy
% - Dataset: CULane or Tusimple
% - Fine-tuning on RoboSense recommended

%% =====================================================================
%% 6. YOLOV8 DETECTION INTEGRATION
%% =====================================================================

% For custom YOLOv8 models:
% 1. Train on dataset with cars/pedestrians
% 2. Export to ONNX:
%    python yolo_to_onnx_converter.py --yolo-model your_model.pt

% For traffic lights/signs (pre-trained):
% Download: https://github.com/ultralytics/yolov8
% Default pre-trained handles common objects

% Typical class outputs:
%   car, truck, bus, motorcycle, bicycle
%   person, dog, cat
%   traffic light, stop sign, speed limit sign

%% =====================================================================
%% 7. MPC CONTROL SETUP
%% =====================================================================

% MPC Configuration
mpc_controller = mpc;
mpc_controller.PredictionHorizon = 20;
mpc_controller.ControlHorizon = 5;
mpc_controller.MV.Min = -30;   % Steering angle bounds
mpc_controller.MV.Max = 30;
mpc_controller.Weights.OutputVariables = [5 0.1 1];

% Usage:
steering_command = mpc_controller(state, reference_trajectory);

% State format: [lateral_position, heading_angle, lateral_velocity]
% Reference: trajectory over prediction horizon (3 × 20)

%% =====================================================================
%% 8. ROBOSENSE DATASET INTEGRATION
%% =====================================================================

% RoboSense directory structure:
% robosense_data/
% ├── calib/                 (camera calibration)
% ├── image/                 (RGB frames)
% │   ├── 1613616137123.png
% │   ├── 1613616137153.png
% │   └── ...
% ├── lidar/                 (point cloud data)
% └── annotations/ (if available)
%     ├── lanes.json
%     └── objects.json

% Load RoboSense data:
image_dir = 'robosense_data/image/';
image_files = dir([image_dir '/*.png']);

for i = 1:length(image_files)
    frame = imread(fullfile(image_dir, image_files(i).name));
    output = visionCtrl.processFrame(frame);
    % Process...
end

%% =====================================================================
%% 9. CAMERA CALIBRATION (Optional)
%% =====================================================================

% If you have camera parameters:
% load('camera_calibration.mat', 'cameraParams');
% visionCtrl.cameraParams = cameraParams;

% Get intrinsic matrix:
% K = cameraParams.IntrinsicMatrix;

% Use for:
% - 3D reconstruction
% - Perspective transform
% - Lane geometry estimation
% - Obstacle distance estimation

%% =====================================================================
%% 10. PERFORMANCE MONITORING
%% =====================================================================

% Get diagnostics
diag = visionCtrl.getPerformanceDiagnostics();

fprintf('Performance Metrics:\n');
fprintf('  Avg processing time: %.1f ms\n', diag.avgProcessingTime);
fprintf('  FPS achieved: %.1f\n', diag.fps);
fprintf('  Total cars detected: %d\n', diag.detectionCounts.cars);
fprintf('  Total pedestrians detected: %d\n', diag.detectionCounts.pedestrians);

%% =====================================================================
%% 11. COMMON PARAMETERS TO TUNE
%% =====================================================================

% Lane detection threshold
lane_threshold = 0.5;  % Probability threshold [0-1]

% Detection confidence
detection_confidence = 0.5;  % YOLOv8 confidence [0-1]

% Traffic light color thresholds (pixels)
min_color_pixels = 10;  % Minimum pixels for color detection

% Speed limit adjustments
speed_mult_yellow = 0.33;  % Yellow = 0.33 × green speed
speed_mult_pedestrian = 0.17;  % Pedestrian = 0.17 × normal speed

% Lane smoothing (temporal)
lane_history_size = 5;  % Frames to average

%% =====================================================================
%% 12. TROUBLESHOOTING QUICK FIXES
%% =====================================================================

% Problem: Lane not detected
% → Check lighting conditions
% → Verify camera calibration
% → Adjust lane_threshold
% → Check ONNX model compatibility

% Problem: Traffic light always 'unknown'
% → Check HSV ranges (RGB vs BGR conversion)
% → Verify bounding box from YOLOv8
% → Check ROI extraction logic

% Problem: MPC steering commands unstable
% → Reduce control gains
% → Increase prediction horizon
% → Smooth reference trajectory
% → Check state measurements

% Problem: Slow processing (<10 FPS)
% → Enable GPU: gpuDevice
% → Reduce input resolution
% → Use quantized INT8 models
% → Profile code: profile viewer

%% =====================================================================
%% 13. FILE LOCATIONS
%% =====================================================================

% Source code:
%   AutonomousVisionController.m         - Main vision pipeline
%   AutonomousVehicleMPC_MainLoop.m      - Control loop
%   AutonomousVehicle_Setup.m            - Detailed setup
%   yolo_to_onnx_converter.py            - Model conversion
%   AutonomousVehicle_QuickRef.m         - This file

% Models:
%   models/onnx/lane_segmentation.onnx   - ONNX lane model
%   models/yolov8_custom.pt              - Custom detection
%   models/yolov8_traffic.pt             - Traffic detection

% Data:
%   data/robosense/                      - RoboSense dataset
%   data/test_videos/                    - Test footage
%   results/                              - Output logs/visualizations

%% =====================================================================
%% 14. DATASET SOURCES
%% =====================================================================

fprintf('Recommended Datasets for Lane Detection:\n\n');
fprintf('1. CULane (Public Road)\n');
fprintf('   - 133K images, multiple lanes\n');
fprintf('   - Download: https://xingangpan.github.io/projects/CULane.html\n\n');

fprintf('2. Tusimple (Highway)\n');
fprintf('   - 6.4K images, highway lanes\n');
fprintf('   - Download: https://github.com/tusimple/tusimple-benchmark\n\n');

fprintf('3. BDD100K\n');
fprintf('   - 100K images, diverse scenarios\n');
fprintf('   - Download: https://bdd-data.berkeley.edu/\n\n');

fprintf('4. RoboSense (Your dataset)\n');
fprintf('   - Use as fine-tuning/validation\n\n');

fprintf('For YOLOv8 Training:\n');
fprintf('   - COCO (cars, pedestrians, etc)\n');
fprintf('   - Custom dataset with annotations\n\n');

%% =====================================================================
%% 15. TYPICAL WORKFLOW
%% =====================================================================

fprintf('\nTypical Usage Workflow:\n\n');
fprintf('Step 1: Setup (one-time)\n');
fprintf('  >> AutonomousVehicle_Setup\n');
fprintf('  >> system(''python yolo_to_onnx_converter.py --mode all'')\n\n');

fprintf('Step 2: Initialize\n');
fprintf('  >> visionCtrl = AutonomousVisionController(...);\n\n');

fprintf('Step 3: Process Video\n');
fprintf('  >> AutonomousVehicleMPC_MainLoop\n\n');

fprintf('Step 4: Analyze Results\n');
fprintf('  >> diag = visionCtrl.getPerformanceDiagnostics();\n\n');

fprintf('Step 5: Visualize\n');
fprintf('  >> visionCtrl.visualizeOutput(frame, output);\n\n');

%% =====================================================================
%% 16. PERFORMANCE BENCHMARKS
%% =====================================================================

fprintf('\nExpected Performance on GPU (NVIDIA RTX 3080):\n');
fprintf('  Lane segmentation (384×384): 10-15 ms\n');
fprintf('  YOLOv8 detection (640×640):  15-25 ms\n');
fprintf('  Traffic light color:         1-2 ms\n');
fprintf('  MPC computation:             5-10 ms\n');
fprintf('  Total per frame:             30-50 ms\n');
fprintf('  → Achievable FPS:            20-30 FPS\n\n');

fprintf('Expected Performance on CPU:\n');
fprintf('  Lane segmentation (384×384): 50-100 ms\n');
fprintf('  YOLOv8 detection (640×640):  100-200 ms\n');
fprintf('  Total per frame:             150-300 ms\n');
fprintf('  → Achievable FPS:            3-6 FPS\n\n');

fprintf('GPU recommended for real-time autonomous driving.\n');

%% END OF QUICK REFERENCE
