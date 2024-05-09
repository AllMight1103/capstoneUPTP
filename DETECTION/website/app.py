from flask import Flask, render_template, request, send_file,make_response,jsonify
import os
import cv2
import base64
from io import BytesIO
import numpy as np
from imgDETECTION import main
app = Flask(__name__)





@app.route('/', methods=['GET', 'POST'])
def home():
    context = {
        'upload': False,
        'original_image': '',
        'processed_image': ''
    }
    if request.method == "POST":
        upload_file = request.files.get('image_name')
        
        if upload_file:
            # Read the image file in a way that allows for memory handling
            in_memory_file = BytesIO(upload_file.read())
            data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
            original_image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            original_image = cv2.resize(original_image, (1280,720))
            # Process image (modify this function to your processing)
            processed_image = main(original_image.copy())

            # Encode both images to base64
            context['original_image'] = encode_image_to_base64(original_image)
            context['processed_image'] = encode_image_to_base64(processed_image)
            context['upload'] = True

    return render_template('index.html', **context)


def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return base64_image



if __name__ == '__main__':
    app.run(debug=True)
