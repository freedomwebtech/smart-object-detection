import cv2
from ultralytics import YOLO
import cvzone
import base64
import os
import threading
import time
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# === Gemini Setup ===
GOOGLE_API_KEY = ""
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

# === Load YOLO model ===
model = YOLO("best.pt")
names = model.names

# === Shared Globals ===
genai_results = {}
latest_frame = None
latest_object_names = set()

# === Video Capture ===
cap = cv2.VideoCapture(0)
cv2.namedWindow("RGB")
last_gemini_time = time.time()

# === Helper: Convert frame to base64 ===
def encode_image_to_base64(image):
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")

# === Gemini Worker Thread ===
def gemini_worker():
    global genai_results
    while True:
        if latest_frame is not None and latest_object_names:
            base64_image = encode_image_to_base64(latest_frame)
            prompt_text = (
                f"From this image, identify the horizontal position (Left or Right) of the following detected objects:\n\n"
                f"{', '.join(latest_object_names)}\n\n"
                "Only respond in this format:\n"
                "| Object | Position (Left/Right/Center) |\n"
                "If only one object is present, mark its position as 'Center'.\n"
            )
            try:
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                )
                response = gemini_model.invoke([message])
                response_text = response.content.strip()

                print("[Gemini Result]\n", response_text)
                # Parse Gemini response into dict
                parsed = {}
                for line in response_text.splitlines():
                    if "|" in line and not line.lower().startswith("| object"):
                        parts = [p.strip() for p in line.strip("|").split("|")]
                        if len(parts) == 2:
                            obj, pos = parts
                            parsed[obj.lower()] = pos
                genai_results = parsed
            except Exception as e:
                print("Gemini error:", e)
        time.sleep(5)

# === Start Gemini Thread ===
threading.Thread(target=gemini_worker, daemon=True).start()

# === Main Loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 600))
    frame = cv2.flip(frame, 1)

    # YOLO Detection
    results = model.track(frame, persist=True)

    # YOLO to Gemini transfer every 5 sec
    if time.time() - last_gemini_time >= 5:
        latest_frame = frame.copy()
        latest_object_names = set()

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        current_visible_names = set()

        for box, class_id in zip(boxes, class_ids):
            x1, y1, x2, y2 = box
            name = names[class_id].lower()
            current_visible_names.add(name)

            # Add to latest object list for Gemini every 5s
            if time.time() - last_gemini_time >= 5:
                latest_object_names.add(name)

            label = name
            if name in genai_results:
                pos = genai_results[name].lower()
                label += f" | {pos}"

                # Fancy color for position
                if pos == "left":
                    box_color = (255, 165, 0)   # orange
                elif pos == "right":
                    box_color = (0, 140, 255)   # light blue
                elif pos == "center":
                    box_color = (0, 255, 255)   # cyan
                else:
                    box_color = (0, 255, 0)
            else:
                box_color = (0, 255, 0)  # default green

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cvzone.putTextRect(
                frame, label, (x1, y1),
                scale=0.9,
                thickness=2,
                colorR=box_color,
                colorT=(0, 0, 0),
                font=cv2.FONT_HERSHEY_SIMPLEX
            )

        # Detect missing objects
        missing_objects = latest_object_names - current_visible_names
        if missing_objects:
            missing_text = "Missing: " + ", ".join(missing_objects)
            cvzone.putTextRect(
                frame,
                missing_text,
                (30, 60),
                scale=1.3,
                thickness=2,
                colorR=(0, 0, 255),  # Red box
                colorT=(255, 255, 255),  # White text
                font=cv2.FONT_HERSHEY_COMPLEX,
                offset=10
            )

        # Reset Gemini timer
        if time.time() - last_gemini_time >= 5:
            last_gemini_time = time.time()

    # Show Frame
    cv2.imshow("RGB", frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
