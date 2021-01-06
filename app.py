import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

model = keras.models.load_model("MNIST_classifier_nn_model.h5")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('predict.html')

@app.route('/api',methods=['POST'])
def get_delay():

    img = request.form
    x = image.img_to_array(img)
    x = np.true_divide(x, 255)
    pred = model.predict(x)
    pred = np.argmax(pred[0].round(2))
    print(pred)

    output = round(pred[0], 2)
    if output == 0:
        return render_template('predict.html', prediction_text="It's a T-shirt/top!")
    elif output == 1:
        return render_template('predict.html', prediction_text="It's a Trouser")
    elif output == 2:
        return render_template('predict.html', prediction_text="It's a Pullover")
    elif output == 3:
        return render_template('predicthtml', prediction_text="It's a Dress")
    elif output == 4:
        return render_template('predict.html', prediction_text="It's a Coat")
    elif output == 5:
        return render_template('predict.html', prediction_text="It's a Sandal")
    elif output == 6:
        return render_template('predict.html', prediction_text="It's a Shirt")
    elif output == 7:
        return render_template('predict.html', prediction_text="It's a Sneaker")
    elif output == 8:
        return render_template('predict.html', prediction_text="It's a Bag")
    else:
        return render_template('predict.html', prediction_text="It's a Ankle boot")



if __name__ == '__main__':
    app.run(port=8080, debug=True)


'''@app.route('/', methods=['GET'])
def home():
    return render_template('predict.html')

@app.route('/predict',methods=['POST'])
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.true_divide(x, 255)
    # x = np.expand_dims(x, axis=0)

    # x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)


# Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(pred_class[0][0][1])               # Convert to string
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)'''



'''import os
import numpy as np
import tensorflow as tf
from PIL import Image
# from resizeimage import resizeimage
from flask import Flask, request, render_template, send_from_directory
from skimage import transform
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

import sys
import glob
import re

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
model = load_model("MNIST_classifier_nn_model.h5")

@app.route('/', methods=['GET'])
def home():
    return render_template('FashionClassifier.html')

@app.route('/predict',methods=['POST'])
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.true_divide(x, 255)
    # x = np.expand_dims(x, axis=0)

    # x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)


# Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(pred_class[0][0][1])               # Convert to string
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)'''
