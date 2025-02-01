import cv2
import mediapipe as mp
import numpy as np

class FaceOrienter:
    def __init__(self):
        # Initialize MediaPipe face mesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.2, min_tracking_confidence=0.2)
        self.mp_drawing = mp.solutions.drawing_utils  # For drawing landmarks

    @staticmethod
    def draw_line(frame, a, b, color=(255, 255, 0)):
        cv2.line(frame, a, b, color, 10)

    def plot_selected_landmarks(self, frame, landmarks):
        # Plot the specific landmarks as small circles
        selected_points = [
            4,  # Nose tip
            152,   # Chin
            263,  # Left eye left corner
            33,  # Right eye right corner
            291,  # Left mouth corner
            61   # Right mouth corner
        ]
        
        for index in selected_points:
            x = int(landmarks.landmark[index].x * frame.shape[1])
            y = int(landmarks.landmark[index].y * frame.shape[0])
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Draw a small green circle for each selected landmark

    def get_yaw_pitch_roll(self, rotation_vector):
        # Convert the rotation vector to a rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Extract the Euler angles from the rotation matrix
        pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])  # Rotation around X-axis (pitch)
        yaw = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1] ** 2 + rotation_matrix[2, 2] ** 2))  # Rotation around Y-axis (yaw)
        roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])  # Rotation around Z-axis (roll)

        # Convert from radians to degrees
        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)
        roll = np.degrees(roll)

        return yaw, pitch, roll

    def visualize_rotation_vector(self, frame, rotation_vector, translation_vector, camera_matrix, dist_coeffs, image_points):
        # Define the 3D axis (length of 100)
        axis_length = 100.0
        axis_points_3D = np.array([
            (axis_length, 0, 0),  # X-axis (Red)
            (0, axis_length, 0),  # Y-axis (Green)
            (0, 0, axis_length)   # Z-axis (Blue)
        ], dtype="double")

        # Project 3D points to the image plane
        projected_points, _ = cv2.projectPoints(axis_points_3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        # Get the nose tip as the origin
        nose_tip = tuple(map(int, image_points[0]))  # The first point in image_points is the nose tip

        # Convert projected points to tuples
        x_axis = tuple(map(int, projected_points[0].ravel()))
        y_axis = tuple(map(int, projected_points[1].ravel()))
        z_axis = tuple(map(int, projected_points[2].ravel()))

        # Draw the 3D coordinate axes
        cv2.line(frame, nose_tip, x_axis, (0, 0, 255), 3)  # X-axis in Red
        cv2.line(frame, nose_tip, y_axis, (0, 255, 0), 3)  # Y-axis in Green
        cv2.line(frame, nose_tip, z_axis, (255, 0, 0), 3)  # Z-axis in Blue

    def orient(self, image_path, show=False):
        # Read the image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Unable to load image {image_path}")
            return

        # Convert the image to RGB as MediaPipe expects RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)  # Process the image using MediaPipe Face Mesh

        # Check if landmarks are found
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                self.plot_selected_landmarks(frame, landmarks)  # Plot selected landmarks as circles

                size = frame.shape
                image_points = np.array([
                    (landmarks.landmark[4].x * size[1], landmarks.landmark[4].y * size[0]),  # Nose tip
                    (landmarks.landmark[152].x * size[1], landmarks.landmark[152].y * size[0]),  # Chin
                    (landmarks.landmark[263].x * size[1], landmarks.landmark[263].y * size[0]),  # Left eye left corner
                    (landmarks.landmark[33].x * size[1], landmarks.landmark[33].y * size[0]),  # Right eye right corner
                    (landmarks.landmark[291].x * size[1], landmarks.landmark[291].y * size[0]),  # Left mouth corner
                    (landmarks.landmark[61].x * size[1], landmarks.landmark[61].y * size[0])  # Right mouth corner
                ], dtype="double")

                # 3D model points
                model_points = np.array([
                    (0.0, 0.0, 0.0),  # Nose tip
                    (0.0, -330.0, -65.0),  # Chin
                    (-225.0, 170.0, -135.0),  # Left eye left corner
                    (225.0, 170.0, -135.0),  # Right eye right corner
                    (-150.0, -150.0, -125.0),  # Left mouth corner
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


                # print(f"Rotation Vector: {rotation_vector}")
                yaw, pitch, roll = self.get_yaw_pitch_roll(rotation_vector)
                # Projection of points to 2D
                if show:
                    self.visualize_rotation_vector(frame, rotation_vector, translation_vector, camera_matrix, dist_coeffs,image_points)
                    projection_points = []
                    for point in [
                        (350.0, 270.0, 0.0), (-350.0, -270.0, 0.0),
                        (-350.0, 270.0, 0.0), (350.0, -270.0, 0.0),
                        (450.0, 350.0, 400.0), (-450.0, -350.0, 400.0),
                        (-450.0, 350.0, 400.0), (450.0, -350.0, 400.0)
                    ]:
                        (proj_point, _) = cv2.projectPoints(np.array([point]), rotation_vector, translation_vector,
                                                        camera_matrix, dist_coeffs)
                        projection_points.append((int(proj_point[0][0][0]), int(proj_point[0][0][1])))

                    # Draw the projected lines
                    self.draw_line(frame, projection_points[0], projection_points[2])
                    self.draw_line(frame, projection_points[2], projection_points[1])
                    self.draw_line(frame, projection_points[1], projection_points[3])
                    self.draw_line(frame, projection_points[3], projection_points[0])

                    self.draw_line(frame, projection_points[4], projection_points[6])
                    self.draw_line(frame, projection_points[6], projection_points[5])
                    self.draw_line(frame, projection_points[5], projection_points[7])
                    self.draw_line(frame, projection_points[7], projection_points[4])

                    # Draw the connecting lines between 3D model points and projections
                    self.draw_line(frame, projection_points[4], projection_points[0], color=(0, 255, 0))
                    self.draw_line(frame, projection_points[6], projection_points[2], color=(0, 255, 0))
                    self.draw_line(frame, projection_points[5], projection_points[1], color=(0, 255, 0))
                    self.draw_line(frame, projection_points[7], projection_points[3], color=(0, 255, 0))


        # Display the image with the drawn lines and landmarks
                    cv2.imshow("Image with Selected Landmarks", frame)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                return yaw, pitch, roll


