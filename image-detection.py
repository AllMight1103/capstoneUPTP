
import cv2
import numpy as np

from ultralytics import YOLO

LP_model = YOLO("/Users/fabriwasmosy/Desktop/capstone/LicensePlate_text_extraction/runs/detect/Yolo_v8n_Model/weights/best.pt")
OCR_model = YOLO("/Users/fabriwasmosy/Desktop/capstone/OCR/runs/detect/OCR_YoloModel/weights/best.pt")



def get_license_text(image):
    characters=[]
    image = cv2.resize(image, (640,640))
    yolo_output = OCR_model(image)[0]
    box = yolo_output.boxes
    names=yolo_output.names
    for j in range(len(box)):
        labels = names[box.cls[j].item()]
        coordinates = box.xyxy[j].tolist()
        confidence = np.round(box.conf[j].item(), 2)
        characters.append({'label': labels, 'coords': coordinates, 'confidence': confidence})
    # Sort characters based on X-axis coordinates
    characters = sorted(characters, key=lambda x: x['coords'][0])  # Sorting based on the X-axis
    
    license_text = ""
    for idx, char in enumerate(characters):
        license_text += char['label']
        
    return license_text

def addtext(image,text1,text2,x1,y1):
    (text_width1, text_height1), _ = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
    cv2.rectangle(image, (int(x1), int(y1) - text_height1 - 5), (int(x1) + text_width1 + 5, int(y1)), (0, 0, 255), -1)  # Background rectangle
    cv2.putText(image, text1, (int(x1) +5, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    
    #text2
    (text_width2, text_height2), _ = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
    cv2.rectangle(image, (int(x1), int(y1) - (3*text_height2) - 5), (int(x1) + text_width2 + 5,  int(y1) - text_height1 - 5), (0, 0, 255), -1)  # Background rectangle
    cv2.putText(image, text2, (int(x1) +5, int(y1) - 37), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def main(image_path):
    image_np = cv2.imread(image_path)
    license_plates = LP_model(image_np)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        license_plate_crop = image_np[int(y1):int(y2), int(x1): int(x2), :]
        license_plate_text = get_license_text(license_plate_crop)
                        
        cv2.rectangle(image_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
        if license_plate_text is not None:
        # Draw bounding box around the license plat
            image_np=addtext(image_np,f'license: {round(score,2)}',f"text:{license_plate_text}",x1,y1)


    # Display the processed image
    cv2.imshow('License Plate Detection', image_np)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()  

    # cv2.imwrite('saved_image.jpg', image_np)     # save image


if __name__ == "__main__":
    image_path = "/Users/fabriwasmosy/Downloads/image1.jpeg"
    main(image_path)