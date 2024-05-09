import xml.etree.ElementTree as ET
import shutil
import os
import cv2

class xml_to_yolo:

    def convert_annotation(self,xml_path, model, output_label_path, image_dir, cropped_image_path,cropped_license):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

             # Derive image file path from xml_path
            img_file_name = os.path.splitext(os.path.basename(xml_path))[0]
            img_path = os.path.join(image_dir, img_file_name + '.jpg')  # Adjust if your images have a different format

            # Ensure the cropped image directory exists if cropping is requested
            

            # Extract image dimensions
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)

            # Open output file
            with open(output_label_path, 'w') as out_file:
                for obj in root.findall('object'):
                    # Get class name (assuming 'license_plate' is the class)
                    cls_name = obj.find('name').text
                    if model == 'LP':
                        if cls_name == 'license_plate':
                            cls_id = 0  # Assign a class ID (if needed)
                    elif model == 'OCR':
                        if cls_name == '0':
                            cls_id = 0  # Assign a class ID (if needed)
                        elif cls_name == '1':
                            cls_id = 1
                        elif cls_name == '2':
                            cls_id = 2
                        elif cls_name == '3':
                            cls_id = 3
                        elif cls_name == '4':
                            cls_id = 4
                        elif cls_name == '5':
                            cls_id = 5
                        elif cls_name == '6':
                            cls_id = 6
                        elif cls_name == '7':
                            cls_id = 7
                        elif cls_name == '8':
                            cls_id = 8
                        elif cls_name == '9':
                            cls_id = 9
                        elif cls_name == 'A':
                            cls_id = 10
                        elif cls_name == 'B':
                            cls_id = 11
                        elif cls_name == 'C':
                            cls_id = 12
                        elif cls_name == 'D':
                            cls_id = 13
                        elif cls_name == 'E':
                            cls_id = 14
                        elif cls_name == 'F':
                            cls_id = 15
                        elif cls_name == 'G':
                            cls_id = 16
                        elif cls_name == 'H':
                            cls_id = 17
                        elif cls_name == 'I':
                            cls_id = 18
                        elif cls_name == 'J':
                            cls_id = 19
                        elif cls_name == 'K':
                            cls_id = 20
                        elif cls_name == 'L':
                            cls_id = 21
                        elif cls_name == 'M':
                            cls_id = 22
                        elif cls_name == 'N':
                            cls_id = 23
                        elif cls_name == 'O':
                            cls_id = 24
                        elif cls_name == 'P':
                            cls_id = 25
                        elif cls_name == 'Q':
                            cls_id = 26
                        elif cls_name == 'R':
                            cls_id = 27
                        elif cls_name == 'S':
                            cls_id = 28
                        elif cls_name == 'T':
                            cls_id = 29
                        elif cls_name == 'U':
                            cls_id = 30
                        elif cls_name == 'V':
                            cls_id = 31
                        elif cls_name == 'W':
                            cls_id = 32
                        elif cls_name == 'X':
                            cls_id = 33
                        elif cls_name == 'Y':
                            cls_id = 34
                        elif cls_name == 'Z':
                            cls_id = 35
                    else:
                        continue  # Skip other objects if any

                    # Get bounding box coordinates
                    bbox = obj.find('bndbox')
                    x_min = float(bbox.find('xmin').text)
                    y_min = float(bbox.find('ymin').text)
                    x_max = float(bbox.find('xmax').text)
                    y_max = float(bbox.find('ymax').text)

                    # Convert coordinates to YOLO format
                    x_center = (x_min + x_max) / (2.0 * w)
                    y_center = (y_min + y_max) / (2.0 * h)
                    box_width = (x_max - x_min) / w
                    box_height = (y_max - y_min) / h

                    # Write to output file in YOLO format
                    out_file.write(f"{cls_id} {x_center} {y_center} {box_width} {box_height}\n")

                    if cropped_license == 'YES':
                        image = cv2.imread(img_path)
                        if image is not None:
                            xmin, ymin, xmax, ymax = map(int, [x_min, y_min, x_max, y_max])
                            license_plate_region = image[ymin:ymax, xmin:xmax]
                            # Optional: Resize the cropped image
                            resized_license_plate = cv2.resize(license_plate_region, (640, 640))
                            cropped_img_name = f"{img_file_name}_license_plate.jpg"
                            cropped_img_path = os.path.join(cropped_image_path, cropped_img_name)
                            cv2.imwrite(cropped_img_path, resized_license_plate)
                        else:
                            print(f"Image file not found for XML: {xml_path}")
                # Save license plate region
                    # Generate output path for the license plate image
                    # img_filename = os.path.basename(img_path)
                    # license_plate_output_path = os.path.join(output_license_plate_dir, img_filename.replace(os.path.splitext(img_filename)[1], f"_license_plate_{cls_id}.jpg"))

                # # Save license plate regionÍÍ
                    # save_license_plate(img_path, license_plate_output_path, [x_min, y_min, x_max, y_max])

        except Exception as e:
            print(f'Error processing {xml_path}: {e}')
    
    def main(self,base_dir,model,cropp_image="NO"):
        # Directories
        #base_dir = r"C:\Users\fabri\Escritorio\ALPR\working\data"
        img_dir = os.path.join(base_dir, 'Images')
        xml_dir = os.path.join(base_dir, 'annotations')
        
        train_img_output_dir = os.path.join(base_dir, 'train/images')
        train_label_output_dir = os.path.join(base_dir, 'train/labels')
        val_img_output_dir = os.path.join(base_dir, 'val/images')
        val_label_output_dir = os.path.join(base_dir, 'val/labels')

        cropped_license_output_dir =  os.path.join(base_dir, 'LicenseCropped')
        # Create output directories if they don't exist
        for dir_path in [train_img_output_dir, train_label_output_dir, val_img_output_dir, val_label_output_dir]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        if cropp_image == 'YES':
            if os.path.exists(cropped_license_output_dir):
                shutil.rmtree(cropped_license_output_dir)
            os.makedirs(cropped_license_output_dir)
        # Process each XML file
        xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
        total_files = len(xml_files)

        train_split = int(0.8 * total_files)  # 80% for training

        for idx, xml_file in enumerate(xml_files): 
            img_file = xml_file.replace('.xml', '')  # File name without extension

            img_extensions = ['.png', '.jpg', '.jpeg']  # Define the image extensions

            # Find the existing image file with one of the specified extensions
            img_found = False
            for ext in img_extensions:
                img_with_ext = img_file + ext
                if os.path.exists(os.path.join(img_dir, img_with_ext)):
                    img_file = img_with_ext
                    img_found = True
                    break

            if img_found:
                if idx < train_split:
                    img_output_path = os.path.join(train_img_output_dir, img_file)
                    label_output_path = os.path.join(train_label_output_dir, img_file.replace(os.path.splitext(img_file)[1], '.txt'))
                else:
                    img_output_path = os.path.join(val_img_output_dir, img_file)
                    label_output_path = os.path.join(val_label_output_dir, img_file.replace(os.path.splitext(img_file)[1], '.txt'))

                # Copy image to respective directory
                shutil.copy(os.path.join(img_dir, img_file), img_output_path)

                # Convert annotation
                self.convert_annotation(os.path.join(xml_dir, xml_file), model, label_output_path,img_dir,cropped_license_output_dir,cropp_image)
            else:
                print(f"No matching image found for {xml_file}.")