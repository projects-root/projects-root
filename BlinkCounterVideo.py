# Blinking Eye Counter with Face Mesh - MediaPipe
# projects.root@gmail.com
# Studymaterial: https://youtu.be/WdWF4z1XgOE
from turtle import width
import cv2
import mediapipe as mp
import numpy as np

def drawing_output(frame,coordinates_left_eye,coordinates_right_eye, blink_counter):
    aux_image = np.zeros(frame.shape, np.uint8)
    contours1 = np.array([coordinates_left_eye])
    contours2 = np.array([coordinates_right_eye])
    cv2.fillPoly(aux_image, pts=[contours1], color=(50, 0, 0))
    cv2.fillPoly(aux_image, pts=[contours2], color=(50, 0, 0))
    output = cv2.addWeighted(frame,1,aux_image, 0.7, 1)
    #cv2.imshow("Aux_image", aux_image)
    
    cv2.rectangle(output, (0,0), (180,50), (125,125,125), -1 )
    cv2.rectangle(output, (182,0), (235,50), (125,0,0), 2 )
    cv2.putText(output, "Blink Counter:", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(output, "{}".format(blink_counter), (190,35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (128,9,250), 2)
    #cv2.imshow("Output", output)
    return output

def eye_aspect_ratio(coordinates):
    d_A=np.linalg.norm(np.array( np.array(coordinates[1]) - np.array(coordinates[5]) ) ) #P2-P6
    d_B=np.linalg.norm(np.array( np.array(coordinates[2]) - np.array(coordinates[4]) ) ) #P3-P5
    d_C=np.linalg.norm(np.array( np.array(coordinates[0]) - np.array(coordinates[3]) ) ) #P1-P4
    return (d_A+d_B)/(2*d_C)

cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
#cap=cv2.VideoCapture('video1.mp4')
mp_face_mesh = mp.solutions.face_mesh
index_left_eye = [33, 160, 158, 133, 153, 144]
index_right_eye = [362, 385, 387, 263, 373, 380]
EAR_THRESH = 0.22
NUM_FRAMES = 2
aux_counter = 0
blink_counter = 0

with mp_face_mesh.FaceMesh(
    static_image_mode = False,
    max_num_faces = 1) as face_mesh:

    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.flip(frame,1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        
        coordinates_left_eye=[]
        coordinates_right_eye=[]

        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                for index in index_left_eye:
                    x=int(face_landmarks.landmark[index].x*width)
                    y=int(face_landmarks.landmark[index].y*height)
                    coordinates_left_eye.append([x, y])
                    cv2.circle(frame,(x, y),2,(0, 255, 255), 1)
                    cv2.circle(frame,(x, y),1,(128, 0, 250), 1)
                
                for index in index_right_eye:
                    x=int(face_landmarks.landmark[index].x*width)
                    y=int(face_landmarks.landmark[index].y*height)
                    coordinates_right_eye.append([x, y])
                    cv2.circle(frame,(x, y),2,(128, 0, 250), 1)
                    cv2.circle(frame,(x, y),1,(0, 255, 255), 1)

        ear_left_eye = eye_aspect_ratio(coordinates_left_eye)
        ear_right_eye = eye_aspect_ratio(coordinates_right_eye)
        ear_average=0.5*(ear_left_eye+ear_right_eye)
        #print("ear_average: ",ear_average,"\tear_left_eye: ",ear_left_eye, "\tear_right_eye: ",ear_right_eye)
       
        #Blinking count
        # The aux counter counts the frames
        if ear_average < EAR_THRESH:
            aux_counter+=1
        else:
            if aux_counter>= NUM_FRAMES:
                aux_counter=0
                blink_counter +=1
        #print("Ear average:",ear_average,"\tEyes closed by ",blink_counter," times")
        frame = drawing_output(frame, coordinates_left_eye, coordinates_right_eye, blink_counter)
        cv2.imshow ("Frame",frame)
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()


