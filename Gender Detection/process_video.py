import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from scipy.spatial import distance

# Load the saved model
try:
    model = load_model(r'C:\Users\Prashanth\Desktop\Gender detection\path_to_your_saved_model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load YOLO model
try:
    net = cv2.dnn.readNet(r'C:\Users\Prashanth\Desktop\Gender detection\yolov3.weights', r'C:\Users\Prashanth\Desktop\Gender detection\yolov3.cfg')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# Load YOLO class names
try:
    with open(r'C:\Users\Prashanth\Desktop\Gender detection\coco.names', 'r') as f:
        classes = f.read().strip().split("\n")
    print("Class names loaded successfully.")
except Exception as e:
    print(f"Error loading class names: {e}")
    exit()

# Get the output layers
try:
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    print("Output layers obtained successfully.")
except IndexError:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
except Exception as e:
    print(f"Error getting output layers: {e}")
    exit()

# Define colors for bounding boxes
colors = {
    'female': (0, 255, 0),
    'male': (0, 255, 255),
    'close_male': (0, 0, 255),
    'line_normal': (0, 255, 255),
    'line_close': (0, 0, 255)
}

def classify_gender(image):
    try:
        image = cv2.resize(image, (224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0
        prediction = model.predict(image, verbose=0)
        return 'male' if prediction[0][0] > 0.5 else 'female'
    except Exception as e:
        print(f"Error classifying gender: {e}")
        return 'unknown'

def apply_nms(boxes, confidences, score_threshold, nms_threshold):
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, nms_threshold)
    if len(indices) > 0:
        indices = indices.flatten()
    return indices

def process_frame(frame, input_size, distance_threshold):
    blob = cv2.dnn.blobFromImage(frame, 0.00392, input_size, swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    height, width = frame.shape[:2]

    boxes = []
    confidences = []
    class_ids = []

    for detection in outs:
        for object_detection in detection:
            scores = object_detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] == "person":
                center_x = int(object_detection[0] * width)
                center_y = int(object_detection[1] * height)
                w = int(object_detection[2] * width)
                h = int(object_detection[3] * height)

                startX = int(center_x - w / 2)
                startY = int(center_y - h / 2)
                endX = startX + w
                endY = startY + h

                boxes.append([startX, startY, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = apply_nms(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    genders = []
    centers = []

    for i in indices:
        box = boxes[i]
        startX, startY, w, h = box
        endX = startX + w
        endY = startY + h

        person_img = frame[startY:endY, startX:endX]
        gender = classify_gender(person_img)
        center = (startX + w // 2, startY + h // 2)

        genders.append(gender)
        centers.append(center)

        color = colors[gender] if gender in colors else (255, 255, 255)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.putText(frame, gender, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    for i in range(len(genders)):
        for j in range(i + 1, len(genders)):
            if (genders[i] == 'male' and genders[j] == 'female') or (genders[i] == 'female' and genders[j] == 'male'):
                dist = distance.euclidean(centers[i], centers[j])
                if dist < distance_threshold:
                    color_line = colors['line_close']
                    cv2.rectangle(frame, (boxes[i][0], boxes[i][1]), (boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]), colors['close_male'], 2)
                else:
                    color_line = colors['line_normal']

                cv2.line(frame, centers[i], centers[j], color_line, 2)

    return frame

def main():
    video_path = r'C:\Users\Prashanth\Desktop\Gender detection\cctv footage2.webm'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    print("Starting video processing...")

    distance_threshold = 100
    input_size = (320, 320)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        processed_frame = process_frame(frame, input_size, distance_threshold)
        
        # Display processed frame
        if processed_frame is not None:
            cv2.imshow('Video', processed_frame)
            # Limit frame rate to 30 FPS
            if cv2.waitKey(33) & 0xFF == ord('q'):
                print("Exiting...")
                break

    cap.release()
    cv2.destroyAllWindows()

    print("Video processing complete.")

if __name__ == "__main__":
    main()
