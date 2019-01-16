import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)

# TODO : 
#cam.setResolution(1280,720)
#cam.setResolution(640,480)


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_mcs_eyepair_big.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

def generatePoints():
    pass

while True:
    ret, img = cap.read()
    im1 = np.array(img)
    # imColor = cv2.cvtColor(im1,cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(im1,cv2.COLOR_RGB2GRAY)
    im_swapped = cv2.cvtColor(im1,cv2.COLOR_RGB2BGR)

    faces = face_cascade.detectMultiScale(gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_DO_CANNY_PRUNING and cv2.CASCADE_SCALE_IMAGE and cv2.CASCADE_DO_ROUGH_SEARCH 
    )

    i = 1
    dataFaces = list()
    for face in faces:
        x,y,w,h = face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = imColor[y:y+h, x:x+w]

        # Take the face, with a circle arround the face
        center = (h/2, h/2)
        #circle_mask2 = np.zeros((h, w), np.uint8)
        #cv2.circle(circle_mask, (w/2, h/2), int((w+h)/(4/0.7)), (255,255,255), thickness=-1)
        #cv2.circle(circle_mask2, (int(w/2), int(h/2-0.1*w)), int((w+h)/(4/0.7)), (255,255,255), thickness=-1)

        #cv2.ellipse(circle_mask, center,(w/4,h/2),0,0,360,255,-1) 

        #cv2.circle(circle_mask, center, int((w+h)/(4/0.7)), (255,255,255), thickness=-1)


        #cv2.circle(circle_mask, center, radius, (255,255,255), thickness=-1)
        # cv2.circle(circle_mask, center, int((w+h)/(4/0.7)), (255,255,255), thickness=-1)
        

        # Show all faces
        i=i+1

        #cv2.rectangle(imColor,(x,y),(x+w,y+h),(255,0,0),2)
        #data = [x,y,w,h,np.copy(roi_color), circle_mask]
        data = [face,np.copy(roi_color)]
        #if(len(dataFaces) < i):

        center = (int(x+w/2), int(y+h/2+0.15*w))
        # Show the faces detected on the image in color
        cv2.circle(imColor, (int(x+w/2), int(y+h/2-0.1*w)), int((w+h)/(4/0.7)), (255,0,0), thickness=3)
        cv2.circle(imColor, center, int((w+h)/(4/0.7)), (255,0,0), thickness=3)
        #cv2.ellipse(imColor, center,(w/4,h/2),0,0,360,(255),3) 
        
        #cv2.ellipse(roi_color, ((x+w/2, y+h/2), (100,100),0), (255,0,0))


        # Eyes
        eyes = eye_cascade.detectMultiScale(roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(10, 10),
            flags=cv2.CASCADE_SCALE_IMAGE and cv2.CASCADE_FIND_BIGGEST_OBJECT and cv2.CASCADE_DO_ROUGH_SEARCH )
        if len(eyes) >= 1:
            data.append(eyes[0])
            cv2.rectangle(roi_color,(eyes[0][0],eyes[0][1]),(eyes[0][0]+eyes[0][2],eyes[0][1]+eyes[0][3]),(0,255,0),2) 
        else :
            data.append(None)
        # for (ex,ey,ew,eh) in eyes:
        #     

        # Mouths
        # mouths = mouth_cascade.detectMultiScale(roi_gray,
        #     scaleFactor=1.1,
        #     minNeighbors=10,
        #     minSize=(30, 50),
        #     flags=cv2.CASCADE_SCALE_IMAGE)
        # for (ex,ey,ew,eh) in mouths:
        #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2) 

        # Noses
        noses = nose_cascade.detectMultiScale(roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(10,10),
            flags=cv2.CASCADE_SCALE_IMAGE and cv2.CASCADE_FIND_BIGGEST_OBJECT)
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

        dataFaces.append(data)
    cv2.imshow('img',imColor)

    # Now, change faces
    number_faces = len(dataFaces)
    for i in range(number_faces):
        face_last, dst_face, eyes_last = dataFaces[i-1%number_faces]
        face_new, src_face, eyes_new = dataFaces[i%number_faces]

        x, y, w, h = face_last
        xn, yn, wn, hn = face_new
        print(face_last)
        print(face_new)

        resized_face = cv2.resize(src_face,(w, h), interpolation = cv2.INTER_NEAREST )
        mask = np.zeros((wn, hn), dtype = np.uint8)  
       

        #cv2.fillConvexPoly(mask, np.int32(cv2.hull8U), (255, 255, 255))
        # cv2.fillPoly(mask, points, (255,255,255))
        #hullIndex = cv2.convexHull(np.array(points), returnPoints = False)
        #print(hullIndex)
        # haut gauche (xe,ye-he)
        # milieu gauche (int(xe*0.7),h/2) % 20% en X
        # bottom gauche (int(xe*1.2), h-ye)
        # bottom droit ((xe+we)/1.2), h-ye)
        # millieu droit (int((xe+he)*1.3),h/2)
        # haut droit (xe+we, ye-he)

        resized_mask = cv2.resize(mask,(w, h), interpolation = cv2.INTER_NEAREST)

        print("-----")
        print(im_swapped.shape)
        print(dst_face.shape)
        print(x+w/2, y+h/2)
        cv2.imshow('1 img mask', resized_mask)
        cv2.imshow('2 img mask', mask)
        
        #im_swapped = cv2.seamlessClone(resized_face, im_swapped, mask, (200,200), cv2.NORMAL_CLONE)
    
        #im_swapped = cv2.seamlessClone(resized_face, im_swapped, mask, (x+w/2, y+h/2), cv2.NORMAL_CLONE)
        #im_swapped = cv2.seamlessClone(resized_face, im_swapped, mask, (int(y+h/2), int(x+w/2)), cv2.NORMAL_CLONE)
    

    #cv2.imshow('img swapped', im_swapped)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#cv2.waitKey(0)
cv2.destroyAllWindows()
