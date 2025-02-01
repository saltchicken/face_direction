import cv2
import mediapipe as mp
import imutils
import numpy as np


class FaceOrienter:
    def __init__(self):
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
        self.image = cv2.imread('/home/saltchicken/Desktop/delete.jpg')

# Convert the image to RGB (MediaPipe works with RGB)
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

# Process the image and get the face landmarks
        self.results = self.face_mesh.process(self.image_rgb)

    @staticmethod
    def draw_line(frame, a, b, color=(255, 255, 0)):
        print(a, b)
        cv2.line(frame, a, b, color, 10)

    def orient(self):
        frame = self.image

        size = self.image.shape
        print(size)

        face_landmarks = self.results.multi_face_landmarks[0]

        nose = face_landmarks.landmark[1]  # Nose tip (index 1)
        left_eye = face_landmarks.landmark[33]  # Left eye (index 33)
        right_eye = face_landmarks.landmark[263]  # Right eye (index 263)
        chin = face_landmarks.landmark[152]
        left_eye_left_corner = face_landmarks.landmark[133]  # Left eye left corner (index 133)
        right_eye_right_corner = face_landmarks.landmark[362]  # Right eye right corner (index 362)
        left_mouth_corner = face_landmarks.landmark[61]  # Left mouth corner (index 61)
        right_mouth_corner = face_landmarks.landmark[185]  # Right mouth corner (index 185)

        # Project 3D face orientation
        # nose_point = np.array([nose.x, nose.y, nose.z])
        # left_eye_point = np.array([left_eye.x, left_eye.y, left_eye.z])
        # right_eye_point = np.array([right_eye.x, right_eye.y, right_eye.z])

        # 2D image points. If you change the image, you need to change vector
        image_points = np.array([
            (nose.x, nose.y),  # Nose tip
            (chin.x, chin.y),  # Chin
            (left_eye_left_corner.x, left_eye_left_corner.y),  # Left eye left corner
            (right_eye_right_corner.x, right_eye_right_corner.y),  # Right eye right corne
            (left_mouth_corner.x, left_mouth_corner.y),  # Left Mouth corner
            (right_mouth_corner.x, right_mouth_corner.y)  # Right mouth corner
        ], dtype="double")

        # 3D model points.
        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corne
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner

        ])
        # Camera internals
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                        dist_coeffs)

        (b1, jacobian) = cv2.projectPoints(np.array([(350.0, 270.0, 0.0)]), rotation_vector, translation_vector,
                                            camera_matrix, dist_coeffs)
        (b2, jacobian) = cv2.projectPoints(np.array([(-350.0, -270.0, 0.0)]), rotation_vector,
                                            translation_vector, camera_matrix, dist_coeffs)
        (b3, jacobian) = cv2.projectPoints(np.array([(-350.0, 270, 0.0)]), rotation_vector, translation_vector,
                                            camera_matrix, dist_coeffs)
        (b4, jacobian) = cv2.projectPoints(np.array([(350.0, -270.0, 0.0)]), rotation_vector,
                                            translation_vector, camera_matrix, dist_coeffs)

        (b11, jacobian) = cv2.projectPoints(np.array([(450.0, 350.0, 400.0)]), rotation_vector,
                                            translation_vector, camera_matrix, dist_coeffs)
        (b12, jacobian) = cv2.projectPoints(np.array([(-450.0, -350.0, 400.0)]), rotation_vector,
                                            translation_vector, camera_matrix, dist_coeffs)
        (b13, jacobian) = cv2.projectPoints(np.array([(-450.0, 350, 400.0)]), rotation_vector,
                                            translation_vector, camera_matrix, dist_coeffs)
        (b14, jacobian) = cv2.projectPoints(np.array([(450.0, -350.0, 400.0)]), rotation_vector,
                                            translation_vector, camera_matrix, dist_coeffs)

        b1 = (int(b1[0][0][0]), int(b1[0][0][1]))
        b2 = (int(b2[0][0][0]), int(b2[0][0][1]))
        b3 = (int(b3[0][0][0]), int(b3[0][0][1]))
        b4 = (int(b4[0][0][0]), int(b4[0][0][1]))

        b11 = (int(b11[0][0][0]), int(b11[0][0][1]))
        b12 = (int(b12[0][0][0]), int(b12[0][0][1]))
        b13 = (int(b13[0][0][0]), int(b13[0][0][1]))
        b14 = (int(b14[0][0][0]), int(b14[0][0][1]))

        self.draw_line(frame, b1, b3)
        self.draw_line(frame, b3, b2)
        self.draw_line(frame, b2, b4)
        self.draw_line(frame, b4, b1)

        self.draw_line(frame, b11, b13)
        self.draw_line(frame, b13, b12)
        self.draw_line(frame, b12, b14)
        self.draw_line(frame, b14, b11)

        self.draw_line(frame, b11, b1, color=(0, 255, 0))
        self.draw_line(frame, b13, b3, color=(0, 255, 0))
        self.draw_line(frame, b12, b2, color=(0, 255, 0))
        self.draw_line(frame, b14, b4, color=(0, 255, 0))

        cv2.imshow("Frame", frame)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    face = FaceOrienter()
    face.orient()
