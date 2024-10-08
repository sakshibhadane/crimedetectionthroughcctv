import cv2
import numpy as np
#Pushbullet for notifs
from pushbullet import Pushbullet  

#Loading YOLO models
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

#Loadig class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

#Defining suspicious classes(weapons, etc.)
suspicious_classes = ["scissors", "knife", "gun", "backpack", "hammer", "explosives", "chainsaw", "helmet", "mask"]

#Pushbullet API setup for notifs to acc.
pb = Pushbullet("o.FhvdC6eYiiLJbPZeP8cYXPVOa0uAtY0N") 

#Sending notifs using Pushbullet
def send_alert(alert_message):
    try:
        pb.push_note("Crime Detection Alert", alert_message)
        print(f"Alert sent: {alert_message}")
    except Exception as e:
        print(f"Failed to send alert: {e}")


cap = cv2.VideoCapture(0) 

#Motion Detection
background = None
motion_detected = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape
    
    #Pre-processing frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    #Processing YOLO outputs
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                #Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    detected_suspicious = False
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)  #Green by default
            if label in suspicious_classes:
                color = (255, 0, 0)  #Red for sus objects
                detected_suspicious = True

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    #Alerting if sus object detected
    if detected_suspicious:
        send_alert("Suspicious activity detected in CCTV footage!")

    #For motion Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if background is None:
        background = gray
        continue

    #Calculating diff betw current frame & background
    frame_diff = cv2.absdiff(background, gray)
    thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #Significant motion detection
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue

        motion_detected = True
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    if motion_detected:
        send_alert("Motion detected in CCTV footage!")

    #Live feed with detections will be shown
    cv2.imshow('Crime Detection CCTV', frame)

    #'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Cam stops working
cap.release()
#All windows closed
cv2.destroyAllWindows()
