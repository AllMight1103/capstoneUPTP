import cv2
import numpy as np
import os
from ultralytics import YOLO


root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Use the root path to construct the model paths
lp_model_path = os.path.join(root_path, "models", "License_plate_detection", "LP_model", "weights", "best.pt")
ocr_model_path = os.path.join(root_path, "models", "OCR_detection", "OCR_model", "weights", "best.pt")

LP_model = YOLO(lp_model_path)
OCR_model = YOLO(ocr_model_path)


def addtext(image,text1,text2,x1,y1):

    (text_width1, text_height1), _ = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
    cv2.rectangle(image, (int(x1), int(y1) - text_height1 - 10), (int(x1) + text_width1 + 5, int(y1)), (0, 0, 255), -1)  # Background rectangle
    cv2.putText(image, text1, (int(x1) +5, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)
    
    #text2
    (text_width2, text_height2), _ = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
    cv2.rectangle(image, (int(x1), int(y1) - (3*text_height2) - 5), (int(x1) + text_width2 + 5,  int(y1) - text_height1 - 10), (0, 0, 255), -1)  # Background rectangle
    cv2.putText(image, text2, (int(x1) +5, int(y1) - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)
    
    return image

def get_license_text(image):
    characters=[]
    image = cv2.resize(image, (640,640))
    yolo_output = OCR_model(image)[0]
    box = yolo_output.boxes
    names=yolo_output.names
    total_confidence = 0
    for j in range(len(box)):
        labels = names[box.cls[j].item()]
        coordinates = box.xyxy[j].tolist()
        confidence = np.round(box.conf[j].item(), 2)
        total_confidence += confidence
        characters.append({'label': labels, 'coords': coordinates, 'confidence': confidence})
        
    
    # Sort characters based on X-axis coordinates
    characters = sorted(characters, key=lambda x: x['coords'][0])  # Sorting based on the X-axis
    
    
    license_text = ""
    for idx, char in enumerate(characters):
        license_text += char['label']
    if license_text is not None:
        average_confidence = total_confidence / len(characters) if characters else 0   
        return license_text,average_confidence
    else:
        return None