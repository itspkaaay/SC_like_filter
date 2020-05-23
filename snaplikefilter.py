import numpy as np
import cv2
#Loading Cascade Files
eyes_detect= cv2.CascadeClassifier('/Users/pk/Desktop/snapchat/Train/third-party/frontalEyes35x16.xml')
nose_detect= cv2.CascadeClassifier('/Users/pk/Desktop/snapchat/Train/third-party/Nose18x15.xml')


#Loading Filter Images
im_mush= cv2.imread('/Users/pk/Desktop/snapchat/Train/mustache.png',cv2.IMREAD_UNCHANGED)
im_glasses= cv2.imread('/Users/pk/Desktop/snapchat/Train/glasses.png',cv2.IMREAD_UNCHANGED)

cap= cv2.VideoCapture(0)
while (True):
    ret, frame= cap.read()
    if(ret==False):
        continue

    # Detect features
    eye = eyes_detect.detectMultiScale(frame, 1.3, 5)
    nose = nose_detect.detectMultiScale(frame, 1.3, 5)


    # Overlay Filters
    for e in eye:
        x_eyes, y_eyes, w_eyes, h_eyes = e
        im_glasses = cv2.resize(im_glasses, (w_eyes, h_eyes))
        eyes_roi = frame[y_eyes:y_eyes + h_eyes, x_eyes:x_eyes + w_eyes, :]
        mask = im_glasses[:, :, 3:] > 0
        new_img = np.where(mask, im_glasses[:, :, :3], eyes_roi)
        frame[y_eyes:y_eyes + h_eyes, x_eyes:x_eyes + w_eyes, :] = new_img




    # Overlaying filter on nose

    for n in nose:
        offset = 30
        x_nose, y_nose, w_nose, h_nose = n
        # cv2.rectangle(im_load, (x_nose, y_nose), (x_nose + w_nose, y_nose + h_nose), (0, 0, 255), 2)
        im_mush = cv2.resize(im_mush, (w_nose, h_nose))
        nose_roi = frame[y_nose + offset:y_nose + h_nose + offset, x_nose:x_nose + w_nose, :]
        mask = im_mush[:, :, 3:] > 0
        new_img = np.where(mask, im_mush[:, :, :3], nose_roi)
        frame[y_nose + offset:y_nose + h_nose + offset, x_nose:x_nose + w_nose, :] = new_img

    cv2.imshow('Cool Boi Filter', frame)
    keypressed = cv2.waitKey(1) & 0xff
    if keypressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()