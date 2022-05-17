# Face Mesh with MediaPipe
# projects.root@gmail.com
# Studymaterial: https://www.youtube.com/watch?v=6lNn5_-RPAA&list=PLBg7GSvtrU2OaYp2F-FqqZk0RUB4IUvvb&index=4
# Pictures are from
# https://pixabay.com/es/photos/gente-tres-retrato-negro-3104635/

import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)

with mp_face_mesh.FaceMesh(
    static_image_mode = False, #since source is online streaming
    max_num_faces = 2,
    min_detection_confidence = 0.5) as face_mesh:
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.flip(frame,1) #for mirror effect
        frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=face_mesh.process(frame_rgb)

        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame,face_landmarks, 
                    mp_face_mesh.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(color=(0,122,122), thickness=1, circle_radius = 1),
                    mp_drawing.DrawingSpec(color=(255,0,255), thickness=1))
        cv2.imshow("Frame", frame)
        k= cv2.waitKey(1) & 0xFF
        if k==27:
            break

cap.release()
cv2.destroyAllWindows()