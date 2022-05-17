# Face Detection with MediaPipe
# projects.root@gmail.com
# Pictures are from
# https://pixabay.com/es/photos/diente-de-le%c3%b3n-semillas-flor-seco-7170280/
# https://pixabay.com/es/photos/foto-ni%c3%b1a-retrato-sonrisa-pura-2016619/
# https://pixabay.com/es/photos/modelo-colegio-vocacional-de-shenzhen-1753032/
# Studymaterial: https://www.youtube.com/watch?v=6lNn5_-RPAA&list=PLBg7GSvtrU2OaYp2F-FqqZk0RUB4IUvvb&index=4


import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

with mp_face_detection.FaceDetection(min_detection_confidence=0.8) as face_detection:
    #min_detection_confidence  minimum confidence detection value in % (0.5 = 50%) so the detection can be considered a sucess
    image = cv2.imread("photo.jpg") #Reading image from JPG file
    height, width, _ =image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)
    
    print("Detections: ",results.detections)

    #Printing Relative Keypoints
    #0 -> Right_Eye
    #1 -> Left_Eye
    #2 -> Nose_Tip
    #3 -> Mouth_Center
    #4 -> Right_Ear_Tragion
    #5 -> Left_Ear_Tragion

    if results.detections is not None:
        for detection in results.detections:
            #Bounding Box
            xmin=int(detection.location_data.relative_bounding_box.xmin*width)
            ymin=int(detection.location_data.relative_bounding_box.ymin*height)
            w=int(detection.location_data.relative_bounding_box.width*width)
            h=int(detection.location_data.relative_bounding_box.height*height)
            cv2.rectangle(image, (xmin,ymin), (xmin+w,ymin+h),(0,255,0),2)

            #Drawing Right Eye
            x_RE=int(detection.location_data.relative_keypoints[0].x*width)
            y_RE=int(detection.location_data.relative_keypoints[0].y*height)
            cv2.circle(image,(x_RE,y_RE),3,(0,0,255),2)

            #Drawing Left Eye
            x_LE=int(detection.location_data.relative_keypoints[1].x*width)
            y_LE=int(detection.location_data.relative_keypoints[1].y*height)
            cv2.circle(image,(x_LE,y_LE),3,(255,0,0),2)

            #Drawing Nose Tip
            x_NT=int(detection.location_data.relative_keypoints[2].x*width)
            y_NT=int(detection.location_data.relative_keypoints[2].y*height)
            cv2.circle(image,(x_NT,y_NT),3,(255,0,255),2)

            #Mouth Center         
            x_MC=int(mp_face_detection.get_key_point(detection,mp_face_detection.FaceKeyPoint.MOUTH_CENTER).x*width)
            y_MC=int(mp_face_detection.get_key_point(detection,mp_face_detection.FaceKeyPoint.MOUTH_CENTER).y*height)
            cv2.circle(image,(x_MC,y_MC),3,(0,255,0),2)
            
            #Right_Ear_Tragion
            x_RET=int(mp_face_detection.get_key_point(detection,mp_face_detection.FaceKeyPoint.RIGHT_EAR_TRAGION).x*width)
            y_RET=int(mp_face_detection.get_key_point(detection,mp_face_detection.FaceKeyPoint.RIGHT_EAR_TRAGION).y*height)
            cv2.circle(image,(x_RET,y_RET),3,(0,255,255),2)
            
            #Left_Ear_Tragion
            x_LET=int(mp_face_detection.get_key_point(detection,mp_face_detection.FaceKeyPoint.LEFT_EAR_TRAGION).x*width)
            y_LET=int(mp_face_detection.get_key_point(detection,mp_face_detection.FaceKeyPoint.LEFT_EAR_TRAGION).y*height)
            cv2.circle(image,(x_LET,y_LET),3,(0,255,255),2)
  

    cv2.imshow("Image",image)
    cv2.waitKey(0)

cv2.destroyAllWindows()

