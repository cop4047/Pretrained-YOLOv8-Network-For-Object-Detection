import cv2
import os

images_dir = 'D:/Downloads/mini/yolo/images/val'
labels_dir = 'D:/Downloads/mini/yolo/labels/val'

classes = ['vehicle.car', 'human.pedestrian.adult', 'vehicle.truck', 
           'vehicle.bus.rigid', 'vehicle.motorcycle', 'vehicle.bicycle']

imgs = [f for f in os.listdir(images_dir) if f.endswith('.jpg')][:5]

for img_name in imgs:
    img = cv2.imread(os.path.join(images_dir, img_name))
    h, w = img.shape[:2]
    
    label_file = os.path.join(labels_dir, img_name.replace('.jpg', '.txt'))
    if not os.path.exists(label_file):
        continue
        
    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            cls, xc, yc, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((xc - bw/2) * w)
            y1 = int((yc - bh/2) * h)
            x2 = int((xc + bw/2) * w)
            y2 = int((yc + bh/2) * h)
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img, classes[cls], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    
    cv2.imshow('Labels', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
