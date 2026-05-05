%% SIMULINK VISION PROCESSOR FUNCTION
% Drop this into a MATLAB Function block in your Simulink model
% Handles real-time frame processing with proper handshake

function [lane_mask, lane_center_x, lane_center_y, traffic_light, speed_limit, ...
          is_allowed_drive, num_cars, num_pedestrians, proc_time] = ...
    VisionProcessor(frame_input)

% Vision processor for Simulink real-time integration
%
% INPUT:
%   frame_input: uint8 array, size [height, width, 3] (RGB frame)
%
% OUTPUTS:
%   lane_mask: uint8, [height, width] - binary lane segmentation
%   lane_center_x: double - lane centerline x coordinate
%   lane_center_y: double - lane centerline y coordinate
%   traffic_light: double - encoded traffic light state (0=unknown, 1=red, 2=yellow, 3=green)
%   speed_limit: double - speed limit in mph
%   is_allowed_drive: logical - safety flag
%   num_cars: double - number of cars detected
%   num_pedestrians: double - number of pedestrians detected
%   proc_time: double - processing time in milliseconds

persistent visionCtrl initialized;

%% Initialize vision controller (once, on first call)
if isempty(initialized)
    visionCtrl = AutonomousVisionController(...
        'models/onnx/lane_segmentation.onnx', ...
        'models/yolov8_custom.pt', ...
        'models/yolov8_traffic.pt', ...
        []);
    initialized = true;
end

%% Process frame
tic;
output = visionCtrl.processFrame(frame_input);
proc_time = toc * 1000;  % Convert to ms

%% Package outputs
% Lane mask
lane_mask = uint8(output.laneSegmentation) * 255;

% Lane centerline
if isnan(output.laneCenterline(1))
    lane_center_x = 0;
    lane_center_y = 0;
else
    lane_center_x = output.laneCenterline(1);
    lane_center_y = output.laneCenterline(2);
end

% Traffic light (encoded as: 0=unknown, 1=red, 2=yellow, 3=green)
switch output.trafficLightColor
    case 'red'
        traffic_light = 1;
    case 'yellow'
        traffic_light = 2;
    case 'green'
        traffic_light = 3;
    otherwise
        traffic_light = 0;
end

% Speed and safety
speed_limit = output.speedLimit;
is_allowed_drive = output.isAllowedToDrive;

% Detection counts
num_cars = length(output.cars);
num_pedestrians = length(output.pedestrians);

end
