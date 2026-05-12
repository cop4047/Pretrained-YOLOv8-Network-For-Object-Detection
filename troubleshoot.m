%% =====================================================================
%% AUTONOMOUS VEHICLE VISION SYSTEM - TROUBLESHOOTING GUIDE
%% =====================================================================
%% This file explains common issues and how to fix them
%%

clear; clc;

fprintf('\n%s\n', repmat('=', 1, 80));
fprintf('AUTONOMOUS VEHICLE VISION SYSTEM - TROUBLESHOOTING GUIDE\n');
fprintf('%s\n\n', repmat('=', 1, 80));

%% =====================================================================
%% ISSUE 1: ONNX MODEL NOT LOADING
%% =====================================================================

fprintf('\n[ISSUE 1] ONNX Model Loading Error\n');
fprintf('%s\n', repmat('-', 1, 80));

fprintf('\nPROBLEM:\n');
fprintf('  "importNetworkFromONNX: Invalid input arguments"\n');
fprintf('  OR "Undefined function importONNXNetwork"\n\n');

fprintf('SOLUTION:\n');
fprintf('\n1. Check MATLAB Version\n');
fprintf('   - MATLAB R2023b+ uses: importNetworkFromONNX()\n');
fprintf('   - MATLAB R2023a- uses: importONNXNetwork()\n');
fprintf('   - Current Code: Uses both (fallback implemented)\n\n');

fprintf('2. Verify Deep Learning Toolbox Installed\n');
fprintf('   >> ver  %% Check for ''Deep Learning Toolbox''\n\n');

fprintf('3. Check ONNX File Path\n');
fprintf('   >> isfile(''lane_segmentation.onnx'')  %% Should return 1\n\n');

fprintf('4. Update ONNX Opset Version\n');
fprintf('   - Your ONNX file should use opset 11 or 12\n');
fprintf('   - If using Python: model.export(format=''onnx'', opset=12)\n\n');

fprintf('5. Workaround: Use HSV Fallback\n');
fprintf('   - Code automatically switches to HSV if ONNX fails\n');
fprintf('   - No ONNX model needed - uses white lane detection\n\n');

%% =====================================================================
%% ISSUE 2: LANE SEGMENTATION NOT WORKING
%% =====================================================================

fprintf('\n[ISSUE 2] Lane Segmentation Producing Blank Masks\n');
fprintf('%s\n', repmat('-', 1, 80));

fprintf('\nPROBLEM:\n');
fprintf('  - Lane segmentation returns all black or all white\n');
fprintf('  - Lane centerline shows [NaN, NaN]\n');
fprintf('  - No lane detected in images\n\n');

fprintf('SOLUTION:\n');
fprintf('\n1. Check Image Input Format\n');
fprintf('   - Input MUST be uint8 RGB (0-255) or single (0-1)\n');
fprintf('   - Test: class(rgbFrame)  %% Should be uint8 or single\n\n');

fprintf('2. Verify Image Dimensions\n');
fprintf('   - Input must be height × width × 3\n');
fprintf('   - Test: size(rgbFrame)  %% Should show [H, W, 3]\n\n');

fprintf('3. Debug Lane Segmentation\n');
fprintf('   >> visionCtrl = AutonomousVisionController(...);\n');
fprintf('   >> [seg, centerline] = visionCtrl.segmentDrivableLane(testImage);\n');
fprintf('   >> imshow(seg)\n');
fprintf('   >> title(''Lane Mask'')\n\n');

fprintf('4. If Using ONNX: Check Normalization\n');
fprintf('   - Networks expect ImageNet normalization\n');
fprintf('   - Mean: [0.485, 0.456, 0.406]\n');
fprintf('   - Std:  [0.229, 0.224, 0.225]\n');
fprintf('   - Code applies this automatically\n\n');

fprintf('5. If Using HSV Fallback: Adjust Thresholds\n');
fprintf('   - Edit segmentWithHSV() method\n');
fprintf('   - Modify S < 0.1 and V > 0.7 thresholds\n');
fprintf('   - Test with: image with clear white lane lines\n\n');

%% =====================================================================
%% ISSUE 3: TRAFFIC LIGHT DETECTION WRONG
%% =====================================================================

fprintf('\n[ISSUE 3] Traffic Light Color Detection Incorrect\n');
fprintf('%s\n', repmat('-', 1, 80));

fprintf('\nPROBLEM:\n');
fprintf('  - Always returns ''unknown''\n');
fprintf('  - Detects wrong colors (red → green)\n');
fprintf('  - Sensitivity to lighting conditions\n\n');

fprintf('SOLUTION:\n');
fprintf('\n1. Verify Input Image Has Sufficient Pixels\n');
fprintf('   - classifyTrafficLightColor() requires ~50+ colored pixels\n');
fprintf('   - Test on images with distinct traffic lights\n\n');

fprintf('2. Adjust HSV Color Ranges\n');
fprintf('   - Current ranges (in AutonomousVisionController.m):\n');
fprintf('     * Red:    H ∈ [0-0.05, 0.95-1.0],  S > 0.3,  V > 0.3\n');
fprintf('     * Yellow: H ∈ [0.10-0.20],          S > 0.3,  V > 0.3\n');
fprintf('     * Green:  H ∈ [0.25-0.45],          S > 0.3,  V > 0.3\n\n');

fprintf('3. Debug Specific Image\n');
fprintf('   >> img = imread(''traffic_light.jpg'');\n');
fprintf('   >> hsvImg = rgb2hsv(im2single(img));\n');
fprintf('   >> imshow(hsvImg(:,:,1))\n');
fprintf('   >> title(''Hue Channel'')  %% Visually check colors\n\n');

fprintf('4. Adjust Thresholds for Your Conditions\n');
fprintf('   - Modify classifyTrafficLightColor() method\n');
fprintf('   - Test with: red light, yellow light, green light samples\n');
fprintf('   - Increase saturation threshold if environment is washed out\n\n');

%% =====================================================================
%% ISSUE 4: SLOW PROCESSING / LOW FPS
%% =====================================================================

fprintf('\n[ISSUE 4] Slow Processing Speed (<10 FPS)\n');
fprintf('%s\n', repmat('-', 1, 80));

fprintf('\nPROBLEM:\n');
fprintf('  - Processing 10+ ms per frame\n');
fprintf('  - Can''t process in real-time\n');
fprintf('  - GPU not being used even when available\n\n');

fprintf('SOLUTION:\n');
fprintf('\n1. Enable GPU Acceleration\n');
fprintf('   >> gpuDevice  %% Check GPU availability\n');
fprintf('   >> visionCtrl.UseGPU = true;\n');
fprintf('   Expected speedup: 2-5x faster\n\n');

fprintf('2. Profile Code to Find Bottleneck\n');
fprintf('   >> profile on\n');
fprintf('   >> AutonomousVehicleMPC_MainLoop\n');
fprintf('   >> profile viewer  %% Find slowest functions\n\n');

fprintf('3. Reduce Input Resolution\n');
fprintf('   >> visionCtrl.segmentationHeight = 256;\n');
fprintf('   >> visionCtrl.segmentationWidth = 256;\n');
fprintf('   Note: May reduce accuracy\n\n');

fprintf('4. Skip Visualization During Real Processing\n');
fprintf('   >> set VISUALIZE = false in MainLoop\n');
fprintf('   >> imshow() is ~50%% slower than processing\n\n');

%% =====================================================================
%% ISSUE 5: YOLOV8 DETECTION NOT INTEGRATED
%% =====================================================================

fprintf('\n[ISSUE 5] YOLOv8 Car/Pedestrian Detection\n');
fprintf('%s\n', repmat('-', 1, 80));

fprintf('\nPROBLEM:\n');
fprintf('  - detectObjects() returns empty cells\n');
fprintf('  - No car or pedestrian detection\n\n');

fprintf('SOLUTION:\n');
fprintf('\nOption A: Use Python Integration (Recommended)\n');
fprintf('  1. Install ultralytics: pip install ultralytics\n');
fprintf('  2. Create Python wrapper function:\n\n');

fprintf('     function [cars, peds] = detectWithYOLO(frame)\n');
fprintf('         py.system(cmd)  %% Call Python detection\n');
fprintf('     end\n\n');

fprintf('  3. Or use MATLAB''s pyrun() command:\n');
fprintf('     >> results = pyrun("from ultralytics import YOLO; ...");\n\n');

fprintf('Option B: Convert YOLOv8 to ONNX\n');
fprintf('  1. In Python: model.export(format=''onnx'', imgsz=640)\n');
fprintf('  2. Load ONNX in MATLAB:\n');
fprintf('     >> yoloNet = importNetworkFromONNX(''yolov8.onnx'');\n');
fprintf('  3. Modify detectObjects() to call yoloNet\n\n');

fprintf('Option C: Use MATLAB Pre-trained Detectors\n');
fprintf('  1. Use MATLAB''s built-in detectors:\n');
fprintf('     >> detector = yolov8(''yolov8m'');  %% MATLAB R2024+\n');
fprintf('  2. Or use older detectors:\n');
fprintf('     >> detector = fasterRCNNObjectDetector(...);\n\n');

%% =====================================================================
%% ISSUE 6: STRANGE RESULTS / UNEXPECTED BEHAVIOR
%% =====================================================================

fprintf('\n[ISSUE 6] Unexpected Behavior / Random Results\n');
fprintf('%s\n', repmat('-', 1, 80));

fprintf('\nPROBLEM:\n');
fprintf('  - Results vary between runs\n');
fprintf('  - Traffic light detection too sensitive\n');
fprintf('  - Lane detection jumps around frame-to-frame\n\n');

fprintf('SOLUTION:\n');
fprintf('\n1. Check Lane Smoothing\n');
fprintf('   - Code uses rolling history buffer (5 frames)\n');
fprintf('   - Should smooth out noise automatically\n');
fprintf('   - If still jumpy: increase laneHistorySize\n\n');

fprintf('2. Verify Traffic Light Threshold\n');
fprintf('   - Current: requires 50+ pixels of same color\n');
fprintf('   - Increase if getting false positives\n');
fprintf('   - Decrease if missing detections\n\n');

fprintf('3. Check For Competing Rules\n');
fprintf('   - Traffic light RED prevents driving\n');
fprintf('   - Pedestrian detection also prevents driving\n');
fprintf('   - Multiple rules can conflict\n\n');

%% =====================================================================
%% STEP-BY-STEP DEBUGGING
%% =====================================================================

fprintf('\n[DEBUGGING WORKFLOW] How to Systematically Find Issues\n');
fprintf('%s\n', repmat('-', 1, 80));

fprintf('\nSTEP 1: Test with Simple Image\n');
fprintf('  >> img = uint8(ones(480, 640, 3) * 128);  %% Gray image\n');
fprintf('  >> visionCtrl = AutonomousVisionController(...);\n');
fprintf('  >> out = visionCtrl.processFrame(img);\n\n');

fprintf('STEP 2: Test Each Component Separately\n');
fprintf('  >> [seg, centerline] = visionCtrl.segmentDrivableLane(img);\n');
fprintf('  >> fprintf(''Lane: %d pixels\\n'', sum(seg(:)));\n\n');

fprintf('  >> color = visionCtrl.classifyTrafficLightColor(img);\n');
fprintf('  >> fprintf(''Traffic Light: %s\\n'', color);\n\n');

fprintf('STEP 3: Check Intermediate Values\n');
fprintf('  >> diag = visionCtrl.getPerformanceDiagnostics();\n');
fprintf('  >> disp(diag)  %% Print all diagnostics\n\n');

fprintf('STEP 4: Compare With Reference\n');
fprintf('  >> out1 = visionCtrl.processFrame(testImg);\n');
fprintf('  >> expected_speed = 30;  %% What you expect\n');
fprintf('  >> assert(out1.speedLimit == expected_speed);\n\n');

%% =====================================================================
%% ADVANCED: MODIFYING FOR YOUR USE CASE
%% =====================================================================

fprintf('\n[ADVANCED] Customization & Integration\n');
fprintf('%s\n', repmat('-', 1, 80));

fprintf('\n1. Different Camera Parameters\n');
fprintf('   - Pass calibration file to AutonomousVisionController\n');
fprintf('   - Use: cameraParams.FocalLength, .PrincipalPoint, etc.\n\n');

fprintf('2. Different Speed Limits\n');
fprintf('   - Modify evaluateTrafficLawCompliance() method\n');
fprintf('   - Change ''speedLimit = 30'' to your values\n');
fprintf('   - Can tie to road sign detection (when integrated)\n\n');

fprintf('3. Different Lane Model\n');
fprintf('   - Replace ONNX model with your own\n');
fprintf('   - Code automatically falls back to HSV if needed\n\n');

fprintf('4. Integration with MPC Controller\n');
fprintf('   - Use output.laneCenterline as reference\n');
fprintf('   - Use output.speedLimit for speed command\n');
fprintf('   - Use output.isAllowedToDrive for safety gate\n\n');

fprintf('5. Real-time Monitoring\n');
fprintf('   >> visionCtrl.visualizeOutput(frame, output);\n');
fprintf('   >> diag = visionCtrl.getPerformanceDiagnostics();\n');
fprintf('   >> fprintf(''FPS: %.1f\\n'', diag.estimatedFPS);\n\n');

%% =====================================================================
%% QUICK START: 3-MINUTE TEST
%% =====================================================================

fprintf('\n%s\n', repmat('=', 1, 80));
fprintf('QUICK START: 3-MINUTE VERIFICATION TEST\n');
fprintf('%s\n\n', repmat('=', 1, 80));

fprintf('Run this to verify everything works:\n\n');

fprintf('  %% 1. Create controller\n');
fprintf('  >> visionCtrl = AutonomousVisionController(..., '', '', '', '');\n\n');

fprintf('  %% 2. Create test image\n');
fprintf('  >> testImg = uint8(ones(480, 640, 3) * 100);\n');
fprintf('  >> testImg(200:250, 250:400, :) = 255;  %% White lane\n');
fprintf('  >> testImg(50:100, 50:100, 1:2) = 255;  %% Red traffic light\n\n');

fprintf('  %% 3. Process\n');
fprintf('  >> out = visionCtrl.processFrame(testImg);\n\n');

fprintf('  %% 4. Check results\n');
fprintf('  >> fprintf(''Lane pixels: %%d\\n'', sum(out.laneSegmentation(:)));\n');
fprintf('  >> fprintf(''Traffic: %%s (should be red)\\n'', out.trafficLightColor);\n');
fprintf('  >> fprintf(''Speed: %%d mph\\n'', out.speedLimit);\n');
fprintf('  >> fprintf(''Drive allowed: %%d\\n'', out.isAllowedToDrive);\n\n');

fprintf('Expected output:\n');
fprintf('  Lane pixels: 12750 (approx)\n');
fprintf('  Traffic: red (should be red)\n');
fprintf('  Speed: 0 (red light = stop)\n');
fprintf('  Drive allowed: 0 (cannot drive on red)\n\n');

fprintf('%s\n', repmat('=', 1, 80));
fprintf('For more help, see AutonomousVehicle_Setup.m\n');
fprintf('%s\n\n', repmat('=', 1, 80));