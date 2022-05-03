
from flask import Flask, request, render_template
from flask_uploads import UploadSet, IMAGES
from flask_restful import Api, Resource, reqparse
import os
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
api = Api(app)
datatest = os.path.join('static', 'datatest')

model = tf.keras.models.load_model('../face_detection_model.h5') #get model
photos = UploadSet('photos', IMAGES) #get file extension
app.config['UPLOAD_FOLDER'] = datatest #set upload path

get_image_args = reqparse.RequestParser()
get_image_args.add_argument("name", type=str)

images = {}

class Images(Resource):
    def get(self, name):
        data = get_image_args.parse_args()
        images[name] = data
        return

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        if 'image' not in request.files:
            return 'Image not found'
        up_image = request.files['image']
        up_image_path = "static/datatest/" + up_image.filename
        up_image.save(up_image_path)

        image = cv2.imread(up_image_path)

        image = cv2.resize(image,(64,64))
        image_arr = np.array(image).reshape(-1, 64, 64, 3)

        classes = model.predict(image_arr)
        score = tf.nn.softmax(classes[0])

        confidence = round(100 * np.max(score),2)
        result = np.argmax(classes)

        if result == 0:
            result = 'not a face'
        else:
            result = 'face'

        uploaded_image = "static/datatest/" + up_image.filename        

    return render_template("index.html",up_image_path = uploaded_image, confidence=confidence, result=result)


if __name__ == '__main__':
    app.run(debug=True)