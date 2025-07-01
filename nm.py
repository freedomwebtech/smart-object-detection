import cv2
from ultralytics import YOLO
import cvzone



# Load YOLOv8 model
model = YOLO('best.pt')
names = model.names


# Open video
cap = cv2.VideoCapture(0)  # Or use 0 for webcam
cv2.namedWindow("RGB")

# Track frames
frame_count = 0



while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 2 != 0:
        continue  # Skip alternate frames for performance

    frame = cv2.resize(frame, (1020, 600))
    frame= cv2.flip(frame,1)
    results = model.track(frame, persist=True)  # car, bus, truck

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for box, track_id, class_id in zip(boxes, ids, class_ids):
            x1, y1, x2, y2 = box
            name = names[class_id]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Draw annotations
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cvzone.putTextRect(frame, f'{name}', (x1, y1), scale=1, thickness=1)

               

   
    # Show video
    cv2.imshow("RGB", frame)

    # Break on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()