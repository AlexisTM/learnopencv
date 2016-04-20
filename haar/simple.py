import numpy as np
import cv2
import time
from VideoCapture import Device

cam = Device()
#cam.setResolution(1280,720)
#cam.setResolution(640,480)


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_mcs_eyepair_big.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

while True:
    img = cam.getImage()
    im1 = np.array(img)
    imColor = cv2.cvtColor(im1,cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(im1,cv2.COLOR_RGB2GRAY)
    im_swapped = cv2.cvtColor(im1,cv2.COLOR_RGB2BGR)

    faces = face_cascade.detectMultiScale(gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE 
    )

    i = 1
    dataFaces = list()
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = imColor[y:y+h, x:x+w]

        # Take the face, with a circle arround the face
        circle_mask = np.zeros((h, w), np.uint8)
        cv2.ellipse(circle_mask, (w/2, h/2),(w/3,h/2),0,0,360,(255,255,255),-1) 

        center = (int(x+w/2), int(y+h/2-0.1*w))
        radius = int((w+h)/(4/0.7))
        print center, radius
        #cv2.circle(circle_mask, center, radius, (255,255,255), thickness=-1)
        # cv2.circle(circle_mask, center, int((w+h)/(4/0.7)), (255,255,255), thickness=-1)
        

        # Show all faces
        i=i+1

        #cv2.rectangle(imColor,(x,y),(x+w,y+h),(255,0,0),2)
        data = [x,y,w,h,np.copy(roi_color), circle_mask]
        if(len(dataFaces) < i):
            dataFaces.append(data)

        center = (int(x+w/2), int(y+h/2+0.15*w))
        # Show the faces detected on the image in color
        cv2.circle(imColor, (int(x+w/2), int(y+h/2-0.1*w)), int((w+h)/(4/0.7)), (255,0,0), thickness=3)
        cv2.circle(imColor, center, int((w+h)/(4/0.7)), (255,0,0), thickness=3)
        cv2.ellipse(imColor, (x+w/2, y+h/2),(w/3,h/2),0,0,360,(0,255,0),-1) 
        
        #cv2.ellipse(roi_color, ((x+w/2, y+h/2), (100,100),0), (255,0,0))


        # Eyes
        eyes = eye_cascade.detectMultiScale(roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(10, 10),
            flags=cv2.CASCADE_SCALE_IMAGE)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2) 

        # Mouths
        # mouths = mouth_cascade.detectMultiScale(roi_gray,
        #     scaleFactor=1.1,
        #     minNeighbors=5,
        #     minSize=(30, 50),
        #     flags=cv2.CASCADE_SCALE_IMAGE)
        # for (ex,ey,ew,eh) in mouths:
        #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2) 

        # Noses
        noses = nose_cascade.detectMultiScale(roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(10,10),
            flags=cv2.CASCADE_SCALE_IMAGE)
        for (ex,ey,ew,eh) in noses:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,255),2) 

        # Smiles
        # smiles = smile_cascade.detectMultiScale(roi_gray,
        #     scaleFactor=1.1,
        #     minNeighbors=5,
        #     minSize=(30, 50),
        #     flags=cv2.CASCADE_SCALE_IMAGE)
        # for (ex,ey,ew,eh) in smiles:
        #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,255,0),2) 

    cv2.imshow('img',imColor)

    # Now, change faces
    number_faces = len(dataFaces)
    for i in range(number_faces):
        x,y,w,h,dst_face,_ = dataFaces[i-1%number_faces]
        src_face = dataFaces[i%number_faces][4]
        src_mask = dataFaces[i%number_faces][5]
        resized_face = cv2.resize(src_face,(w, h), interpolation = cv2.INTER_NEAREST )
        resized_mask = cv2.resize(src_mask,(w, h), interpolation = cv2.INTER_NEAREST )

        im_swapped = cv2.seamlessClone(resized_face, im_swapped, resized_mask, (x+h/2, y+w/2), cv2.NORMAL_CLONE)
        
        # height, width = src_face.shape[:2]
        # print "image resized", height, width
        # height, width = resized_mask.shape[:2]
        # print "mask resized", height, width
        # height, width = im_swapped.shape[:2]
        # print "desination image", height, width
    

    cv2.imshow('img swapped', im_swapped)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#cv2.waitKey(0)
cv2.destroyAllWindows()
