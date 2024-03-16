import io
import os
import pandas as pd
from google.cloud import vision
from google_vision_ai import VisionAI
from google_vision_ai import *
from ultralytics import YOLO
import numpy as np
from imutils.perspective import four_point_transform
import cv2
import torch
import imutils

global detected_text

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'key.json'

client = vision.ImageAnnotatorClient()

def detect_properties(path):
    """Detects the dominant color in the image."""
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.image_properties(image=image)
    props = response.image_properties_annotation

    for color in props.dominant_colors.colors:
        # Extract RGB values
        red = color.color.red
        green = color.color.green
        blue = color.color.blue

        # Determine the dominant color based on RGB values
        if red > green and red > blue:
            return "Red(Non-Refundable)"
        elif green > red and green > blue:
            return "Green(Refundable)"
        else:
            return "White(Non-Refundable)"

    # If no dominant color is found, return "Unknown"
    return "Unknown"


def VisionAI_based(uploaded_image_path):
 


 def orientation(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

  # Find contours and sort for largest contour
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    displayCnt = None

    for c in cnts:
    # Fit a rotated rectangle to the contour
      rect = cv2.minAreaRect(c)
      box = cv2.boxPoints(rect)
      box = np.int0(box)
      if len(box) == 4:
        displayCnt = box
        break

 
    if displayCnt is not None:
    # Obtain birds' eye view of the image
      warped = four_point_transform(image, displayCnt)

      cv2.imwrite("FinalResult.jpg", warped)

    return warped
 

 model = YOLO("best.pt")
 im1 = cv2.imread(uploaded_image_path)
 height, width, _ = im1.shape
 if width > height:
     im1 = cv2.rotate(im1, cv2.ROTATE_90_CLOCKWISE)
     
 im1=cv2.resize(im1, (480, 1024))
 results = model.predict(source=im1,imgsz=1024,save=True, save_crop=True)
 
 im2=cv2.imread(uploaded_image_path)
 height, width, _ = im2.shape
 if width > height:
       im2 = cv2.rotate(im2, cv2.ROTATE_90_CLOCKWISE)  
 im2=cv2.resize(im2, (480, 1024))
 for result in results:
    for mask in result.masks:
        m = torch.squeeze(mask.data)
        composite = torch.stack((m, m, m), 2)
        tmp = im2 * composite.cpu().numpy().astype(np.uint8)
        
        cv2.imwrite("result.jpg",tmp)
        FinalResult=orientation('result.jpg')
        image_file_path= './FinalResult.jpg'
        image = prepare_image_local(image_file_path)
        va= VisionAI(client,image)
        texts= va.text_detection()
        detected_text=texts[0].description
        with open('dt.txt', 'w') as file:
          file.write(detected_text)
        #print(texts[0].description)
        detect_properties(image_file_path)
        #cv2.imshow("result", FinalResult)
        #cv2.waitKey(0)