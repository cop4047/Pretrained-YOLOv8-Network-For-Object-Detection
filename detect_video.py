import cv2
import yt_dlp
from ultralytics import YOLO

# Load your trained model
model = YOLO(r'C:\Users\alros\Documents\GitHub\Pretrained-YOLOv8-Network-For-Object-Detection\runs\detect\train6\weights\best.pt')

# Get the YouTube stream URL
youtube_url = 'https://www.youtube.com/watch?v=b4Rr_7y068U'  # replace with your URL

ydl_opts = {'format': 'best[ext=mp4]', 'quiet': True}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(youtube_url, download=False)
    stream_url = info['url']

print("Stream loaded, starting detection...")

# Open video stream
cap = cv2.VideoCapture(stream_url)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame, conf=0.25, verbose=False)

    # Draw boxes on frame
    annotated = results[0].plot()

    # Show frame
    cv2.imshow('YOLOv8 Detection', annotated)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
