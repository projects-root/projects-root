# Face Mesh with MediaPipe
# projects.root@gmail.com
# Studymaterial: https://www.youtube.com/watch?v=6lNn5_-RPAA&list=PLBg7GSvtrU2OaYp2F-FqqZk0RUB4IUvvb&index=4
# Pictures are from
# https://pixabay.com/es/photos/gente-tres-retrato-negro-3104635/

import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

index_list=[70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
            122, 196, 3, 51, 281, 248, 419, 351, 
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]

with mp_face_mesh.FaceMesh(
    static_image_mode = True,
    max_num_faces = 2,
    min_detection_confidence = 0.5) as face_mesh:
    #static_image_mode = false -> when source is streamingvideo
    image = cv2.imread("photo5.jpg")
    #Obtaining image's size
    height, width, _ = image.shape
    #For detection we need to change the image's channels from BGR to RGB
    image_rgb=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    #print("Face landmarks: ", results.multi_face_landmarks)

    #Drawing desired points according canonical face
    #Nose_Center
    if results.multi_face_landmarks is not None:
        for face_landmarks in results.multi_face_landmarks:
            x_nc =int(face_landmarks.landmark[4].x*width)
            y_nc =int(face_landmarks.landmark[4].y*height)
            cv2.circle(image, (x_nc,y_nc), 2, (255,0,255), 1)
    #using an index list
    if results.multi_face_landmarks is not None:
        for face_landmarks in results.multi_face_landmarks:
            for index in index_list:
                x_index=int(face_landmarks.landmark[index].x*width)
                y_index=int(face_landmarks.landmark[index].y*height)
                cv2.circle(image, (x_index,y_index), 2, (0,255,255), 1)

    #Drawing all points of canonical face
    #if results.multi_face_landmarks is not None:
    #    for face_landmarks in results.multi_face_landmarks:
    #        mp_drawing.draw_landmarks(image,face_landmarks, 
    #            mp_face_mesh.FACEMESH_CONTOURS,
    #            mp_drawing.DrawingSpec(color=(0,122,122), thickness=1, circle_radius = 1),
    #            mp_drawing.DrawingSpec(color=(255,0,255), thickness=1))
    cv2.imshow("Image",image)
    cv2.waitKey(0)

cv2.destroyAllWindows