# CodeClause
object detection 
import cv2
# Load class names for recognition
classNames = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus',
              7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
              14: 'motorbike', 15: 'person', 16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}

# Load pre-trained model
model = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")

# Load an image
img = cv2.imread("bmwm4.jpg")
(h, w) = img.shape[:2]
img_resized = cv2.resize(img, (300, 300))
img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
# Convert image to blob input
blob = cv2.dnn.blobFromImage(img_resized, 0.007843, (300, 300), 127.5)
blob = cv2.dnn.blobFromImage(img_resized, 0.007843, (300, 300), 127.5, swapRB=True)
model.setInput(blob)

# Predict
detections = model.forward()

for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > 0.5:  # minimum accuracy threshold
        idx = int(detections[0, 0, i, 1])
        label = classNames[idx]
        box = detections[0, 0, i, 3:7] * [w, h, w, h]
        x1, y1, x2, y2 = box.astype("int")

        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

# Show output
cv2.imshow("Object Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
