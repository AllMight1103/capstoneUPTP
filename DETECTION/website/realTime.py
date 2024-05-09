import cv2
import sys
import os
from datetime import datetime

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
IMAGE_SAVE = os.path.join(BASE_PATH,'static/image.jpg')

from database import ALPRDatabase
from utilities.utils import LP_model,addtext,get_license_text


def RTdetection(image_np):           
    license_plates = LP_model(image_np)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # Crop license plate
        license_plate_crop = image_np[int(y1):int(y2), int(x1): int(x2), :]

        license_plate_text,text_score = get_license_text(license_plate_crop)
        print(license_plate_text,text_score)
        cv2.rectangle(image_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
        
        image_np=addtext(image_np,f'license: {round(score,2)}',f"text:{license_plate_text}",x1,y1)
        # if text_score >=0.81:
        #     db.insert_license_plate(license_plate_text,datetime.now().strftime("%Y-%m-%d %H:%M:%S"))   
        cv2.imwrite(IMAGE_SAVE,image_np)
    return image_np

