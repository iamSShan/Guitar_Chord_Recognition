import cv2
import numpy as np
import mediapipe as mp
import time
from tensorflow.keras.models import load_model
from gpt import get_chord_info


# Initialize MediaPipe hand detector and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load trained CNN model
model = load_model("guitar_model.h5")

# Resize settings for input image
IMG_WIDTH, IMG_HEIGHT = 200, 200

# Mapping model output index to corresponding chord label
chord_labels = {0: "C major", 1: "D major", 2: "E minor", 3: "F major", 4: "G major"}


# Resize frame to reduce display size (optional)
def resize_frame(frame, scale_percent=75):
    new_width = int(frame.shape[1] * scale_percent / 100)
    new_height = int(frame.shape[0] * scale_percent / 100)
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


# Prepare image for prediction
def preprocess_image(img):
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, IMG_WIDTH, IMG_HEIGHT, 1))
    return img


# Predict chord using CNN
def predict_chord(model, img):
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)[0]
    print(f"Prediction probs: {prediction}")  # Debug line
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]
    return confidence, predicted_class


def run_chord_detection():
    cap = cv2.VideoCapture(0)
    hand_detector = mp_hands.Hands(
        min_detection_confidence=0.7, min_tracking_confidence=0.7
    )
    last_gpt_fetch_time = 0
    chord_info = ""
    last_label = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hand_detector.process(rgb_image)

        frame_height, frame_width, _ = frame.shape

        if results.multi_hand_landmarks:
            # This detects both hands:
            # for hand_landmarks in results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # Only take the first hand

            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(thickness=5, circle_radius=5),
                mp_drawing.DrawingSpec(thickness=10, circle_radius=10),
            )

            # Get hand bounding box
            x_coords = [int(lm.x * frame_width) for lm in hand_landmarks.landmark]
            y_coords = [int(lm.y * frame_height) for lm in hand_landmarks.landmark]

            x_min = max(min(x_coords) - 20, 0)
            y_min = max(min(y_coords) - 20, 0)
            x_max = min(max(x_coords) + 20, frame_width)
            y_max = min(max(y_coords) + 20, frame_height)

            hand_region = frame[y_min:y_max, x_min:x_max]

            if hand_region.size == 0:
                continue  # Skip if region is empty

            # Preprocess for prediction
            gray_hand = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
            hand_resized = cv2.resize(gray_hand, (IMG_WIDTH, IMG_HEIGHT))

            # Debug view of the cropped hand image
            # cv2.imshow("Hand Region", hand_resized)

            # Predict
            confidence, predicted_class = predict_chord(model, hand_resized)
            predicted_label = chord_labels.get(predicted_class, "Unknown")

            print(
                f"[DEBUG] Predicted: {predicted_label} | Confidence: {confidence:.2f}"
            )

            # Optional: filter out weak predictions
            if confidence < 0.6:
                continue

            # Annotate frame
            cv2.putText(
                frame,
                # f"{predicted_label} ({confidence*100:.1f}%)",
                f"{predicted_label}",
                (x_min + 10, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 0, 0),
                3,
            )
            # Show GPT chord info only every 3 seconds
            current_time = time.time()
            # Only fetch GPT info every 4s or when chord changes
            if (current_time - last_gpt_fetch_time > 4) or (
                predicted_label != last_label
            ):
                chord_info = get_chord_info(predicted_label)
                last_gpt_fetch_time = current_time
                last_label = predicted_label

            if current_time - last_gpt_fetch_time <= 2:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                # Display multiline chord info below the bounding box
                y_offset = y_max + 30
                for i, line in enumerate(chord_info.split("\n")):
                    cv2.putText(
                        frame,
                        line,
                        (x_min, y_offset + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )

        # Display
        display_frame = resize_frame(frame)
        cv2.imshow("Chord Detection", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hand_detector.close()


if __name__ == "__main__":
    run_chord_detection()
