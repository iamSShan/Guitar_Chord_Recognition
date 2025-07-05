import os
import cv2
import mediapipe as mp  # MediaPipe for hand tracking and gesture recognition


class ChordsCapture:
    def __init__(self, chord_name, total_images=1000, width=200, height=200):
        # Initialize parameters for hand gesture capture
        self.chord_name = chord_name
        self.total_images = total_images
        self.width = width
        self.height = height

        # Initialize MediaPipe hands and drawing utilities
        self.drawing_utils = mp.solutions.drawing_utils
        self.hands_model = mp.solutions.hands
        self.hands_detector = self.hands_model.Hands(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

        # Image specification for drawing
        self.hand_landmarks_specs = self.drawing_utils.DrawingSpec(
            thickness=5, circle_radius=5
        )
        self.hand_connections_specs = self.drawing_utils.DrawingSpec(
            thickness=10, circle_radius=10
        )

        # Ensure the directory for saving images exists
        self.dataset_path = f"chords_dataset/{self.chord_name}"
        self.ensure_directory_exists(self.dataset_path)

        # Initialize video capture (use camera 1, or change to 0 if using another camera)
        self.capture = cv2.VideoCapture(0)

    def ensure_directory_exists(self, directory_name):
        """
        Ensure that the folder for saving images exists
        """
        os.makedirs(directory_name, exist_ok=True)

    def get_bounding_box_from_landmarks(
        self, landmarks, frame_width, frame_height, padding=20
    ):
        x_coords = [lm.x * frame_width for lm in landmarks.landmark]
        y_coords = [lm.y * frame_height for lm in landmarks.landmark]

        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))

        # Optional: Add padding
        x_min = max(x_min - padding, 0)
        y_min = max(y_min - padding, 0)
        x_max = min(x_max + padding, frame_width)
        y_max = min(y_max + padding, frame_height)

        return x_min, y_min, x_max - x_min, y_max - y_min

    def capture_hand_images(self):
        """
        Capture hand gesture images using MediaPipe landmarks
        """
        image_count = 0
        capturing = True

        while self.capture.isOpened():
            ret, frame = self.capture.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands_detector.process(frame_rgb)

            frame_height, frame_width, _ = frame.shape

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    self.drawing_utils.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.hands_model.HAND_CONNECTIONS,
                        self.hand_landmarks_specs,
                        self.hand_connections_specs,
                    )

                    # Get bounding box from landmarks
                    x, y, w, h = self.get_bounding_box_from_landmarks(
                        hand_landmarks, frame_width, frame_height, padding=20
                    )

                    # Draw the bounding box on screen
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    if capturing:
                        # Extract hand ROI
                        hand_roi = frame[y : y + h, x : x + w]
                        # if hand_roi.size == 0:
                        #     print("[WARNING] Empty ROI - skipping frame.")
                        #     continue
                        gray_hand = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
                        resized_hand = cv2.resize(gray_hand, (self.width, self.height))

                        # if x < 0 or y < 0 or (x + w) > frame_width or (y + h) > frame_height:
                        #     print(f"[WARNING] Skipping frame: invalid bounding box ({x}, {y}, {w}, {h})")
                        #     continue

                        # Save image
                        image_count += 1
                        save_path = f"{self.dataset_path}/{image_count}.jpg"
                        cv2.imwrite(save_path, resized_hand)

                        # Feedback
                        cv2.putText(
                            frame,
                            "Capturing...",
                            (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (127, 255, 255),
                            3,
                        )
                        cv2.putText(
                            frame,
                            f"Image {image_count}/{self.total_images}",
                            (30, 400),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            (127, 127, 255),
                            2,
                        )

                    if image_count >= self.total_images:
                        print("Captured all images.")
                        self.capture.release()
                        cv2.destroyAllWindows()
                        return

            # Display frames
            cv2.imshow("Gesture Capture", frame)

            # Key handling
            key = cv2.waitKey(1)

            # If you want to pause capturing at any moment
            if key == ord("c"):
                capturing = not capturing
                print("Capturing started." if capturing else "Capturing paused.")
            # To exit the application
            elif key == ord("q"):
                print("Exiting.")
                break

        self.capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Ask for the chord name and start the image capture process
    chord_name = input("Enter chord name: ")
    chords_capture = ChordsCapture(chord_name)
    chords_capture.capture_hand_images()
