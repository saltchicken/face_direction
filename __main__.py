import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load your image
image = cv2.imread('/home/saltchicken/Desktop/delete.jpg')

# Convert the image to RGB (MediaPipe works with RGB)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image and get the face landmarks
results = face_mesh.process(image_rgb)

# Check if any faces are detected
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        # Draw landmarks on the face
        # mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

        # Get the coordinates of key landmarks (e.g., nose, eyes, etc.)
        nose = face_landmarks.landmark[1]  # Nose tip (index 1)
        left_eye = face_landmarks.landmark[33]  # Left eye (index 33)
        right_eye = face_landmarks.landmark[263]  # Right eye (index 263)

        # Project 3D face orientation
        nose_point = np.array([nose.x, nose.y, nose.z])
        left_eye_point = np.array([left_eye.x, left_eye.y, left_eye.z])
        right_eye_point = np.array([right_eye.x, right_eye.y, right_eye.z])

        # Calculate vectors
        eye_vector = right_eye_point - left_eye_point
        nose_vector = nose_point - left_eye_point

        # Calculate the angle between nose and eyes
        dot_product = np.dot(eye_vector, nose_vector)
        magnitude_eye = np.linalg.norm(eye_vector)
        magnitude_nose = np.linalg.norm(nose_vector)
        angle = np.arccos(dot_product / (magnitude_eye * magnitude_nose))

        # Convert angle from radians to degrees
        angle_deg = np.degrees(angle)

        # Draw an arrow to indicate the face direction
        # We will use the direction of the nose vector
        nose_x = int(nose.x * image.shape[1])
        nose_y = int(nose.y * image.shape[0])

        # The arrow will point in the direction of the nose
        arrow_length = 100  # Length of the arrow
        arrow_dx = int(nose.x * image.shape[1] + arrow_length * nose_vector[0])
        arrow_dy = int(nose.y * image.shape[0] + arrow_length * nose_vector[1])

        # Draw the arrow
        cv2.arrowedLine(image, (nose_x, nose_y), (arrow_dx, arrow_dy), (0, 0, 255), 5)

        # Display the angle on the image
        cv2.putText(image, f'Face Angle: {angle_deg:.2f} degrees', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# Show the image with face landmarks, angle, and arrow
cv2.imshow('Face Orientation', image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

