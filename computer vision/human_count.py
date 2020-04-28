from imutils import object_detection 
# import non_max_suppression 
import numpy as np 
import imutils #ry tool for DIP and will let us perform different transformations from our results
import cv2 
import requests 
import time 
import argparse #read commands from our command terminal inside our script.

# Opencv pre-trained SVM with HOG people features 
HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def detector(image):
    '''
    @image is a numpy array
    '''

    image = imutils.resize(image, width=min(400, image.shape[1]))
    clone = image.copy()

    (rects, weights) = HOGCV.detectMultiScale(image, winStride=(8, 8),
                                              padding=(32, 32), scale=1.05)

    # Applies non-max supression from imutils package to kick-off overlapped causing false positives or detection errors
    # boxes
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    result = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    # 
    return result

def localDetect(image_path):
    result = []
    image = cv2.imread(image_path)
    if len(image) <= 0:
        print("[ERROR] could not read your local image")
        return result
    print("[INFO] Detecting people")
    result = detector(image)

    # shows the result
    for (xA, yA, xB, yB) in result:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    cv2.imshow("result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return (result, image)

def cameraDetect(token, device, variable, sample_time=5):
    
    cap = cv2.VideoCapture(0)
    init = time.time()

    # Allowed sample time for Ubidots is 1 dot/second
    if sample_time < 1:
        sample_time = 1

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=min(400, frame.shape[1]))
        result = detector(frame.copy())

        # shows the result
        for (xA, yA, xB, yB) in result:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
        cv2.imshow('frame', frame)

        '''
        # Sends results
        if time.time() - init >= sample_time:
            print("[INFO] Sending actual frame results")
            # Converts the image to base 64 and adds it to the context
            b64 = convert_to_base64(frame)
            context = {"image": b64}
            #sendToUbidots(token, device, variable,len(result), context=context)
            init = time.time()
        '''

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def convert_to_base64(image):
    image = imutils.resize(image, width=400)
    img_str = cv2.imencode('.png', image)[1].tostring()
    b64 = base64.b64encode(img_str)
    return b64.decode('utf-8')

def detectPeople(camera):
    #     image_path = args["image"]
    camera = True
    # if str(args["camera"]) == 'true' else False

    '''
    # Routine to read local image
    if image_path != None and not camera:
        print("[INFO] Image path provided, attempting to read image")
        (result, image) = localDetect(image_path)
        print("[INFO] sending results")
        # Converts the image to base 64 and adds it to the context
        b64 = convert_to_base64(image)
        context = {"image": b64}

        
        # Sends the result
        req = sendToUbidots(TOKEN, DEVICE, VARIABLE,
                            len(result), context=context)
        if req.status_code >= 400:
            print("[ERROR] Could not send data to Ubidots")
            return req
        '''

    # Routine to read images from webcam
    if camera:
        print("[INFO] reading camera images")
        cameraDetect(TOKEN, DEVICE, VARIABLE)

def main():
    # args = argsParser()
    detectPeople("camera")
