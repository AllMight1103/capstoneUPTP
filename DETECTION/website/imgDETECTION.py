
import cv2
import sys
import os
from datetime import datetime

current_dir = os.path.abspath(os.getcwd())
parent_dir = os.path.dirname(current_dir)
utilities_path = os.path.join(parent_dir)
sys.path.append(utilities_path)

#from database import ALPRDatabase
from utilities.utils import LP_model,addtext,get_license_text


def main(image_np):
    
    # image_np = cv2.imread(image_path)
    license_plates = LP_model(image_np)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        license_plate_crop = image_np[int(y1):int(y2), int(x1): int(x2), :]
        license_plate_text,text_score = get_license_text(license_plate_crop)
                       
        cv2.rectangle(image_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
        if license_plate_text is not None:
        # Draw bounding box around the license plat
            image_np=addtext(image_np,f'license: {round(score,2)}',f"text:{license_plate_text}",x1,y1)
            # if text_score >=0.81:
            #     db.insert_license_plate(license_plate_text,datetime.now().strftime("%Y-%m-%d %H:%M:%S"))  
    
    return image_np
    
    # cv2.imwrite('saved_image.jpg', image_np)     # save image


# main(r'C:\Users\fabri\Escritorio\ALPR_FINAL\website_for_capstone_myself\img_000013.jpg','img.jpg')

