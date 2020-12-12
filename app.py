import os
import numpy as np
import tensorflow as tf
from PIL import Image
from resizeimage import resizeimage
from flask import Flask, request, render_template, send_from_directory
from skimage import transform

app = Flask(__name__)
model = tf.keras.models.load_model("MNIST_classifier_nn_model.h5")

@app.route('/')
def home():
    return render_template('FashionClassifier.html')

@app.route('/predict',methods=['POST'])
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('FashionClassifier.html')

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
    app.run(debug=True)




'''import os
import numpy as np
import tensorflow as tf
from PIL import Image
from resizeimage import resizeimage
from flask import Flask, request, render_template, send_from_directory
from skimage import transform

# Your directory name where images should be uploaded
UPLOAD_FOLDER = 'uploads'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
new_model = tf.keras.models.load_model("MNIST_classifier_nn_model.h5")

@app.route('/')
def home():
    return render_template('FashionClassifier.html')


# Your Route
@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        name = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], name))

        image_file = Image.open("uploads/{}".format(name))  # open colour image
        image_file = image_file.convert('1')  # convert image to black and white
        image_file.save('uploads/{}'.format(name))

        with open('uploads/{}'.format(name), 'r+b') as f:
            with Image.open(f) as image:
                cover = resizeimage.resize_cover(image, [28, 28])
                cover.save('uploads/{}'.format(name), image.format)

        img = Image.open('uploads/{}'.format(name))
        array = np.array(img)

        new_model = tf.keras.models.load_model("MNIST_classifier_nn_model.h5")

        pred = new_model.predict(np.array([array]))

        pred = np.argmax(pred)

        # labels of images
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                       'Ankle boot']
        val = class_names[pred]
        # return "Machine predicted value is: {}".format(val)
        return render_template('predict.html', image_file_name=file.filename, val=val)
    else:
        return render_template("FashionClassifier.html")


@app.route('/upload/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


def create_app():
    load__model()
    return app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
    '''
    

'''
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from skimage import transform

app = Flask(__name__)
loaded_model = load_model("MNIST_classifier_nn_model.h5")
loaded_model._make_predict_function()
graph = tf.get_default_graph()


def valuepredictor(np_arr):
    global graph
    with graph.as_default():
        resultss = loaded_model.predict(np_arr)
    return resultss[0]


# noinspection PyUnresolvedReferences
def image_preprocess(img):
    new_shape = (50, 50, 3)
    img = image.load_img(img)
    image_array = image.img_to_array(img)
    image_array = transform.resize(image_array, new_shape, anti_aliasing=True)
    image_array /= 255
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


@app.route('/')
def home():
    return render_template("FashionClassifier2.html")


# noinspection PyUnresolvedReferences
@app.route('/result', methods=['POST'])
def result():
    prediction = ''
    if request.method == 'POST':
        img = request.files['pic']
        img_arr = image_preprocess(img)
        result = ValuePredictor(img_arr)
        print("result from model", result)
        result = int(np.argmax(result))
        print("result actual", result)
        if result == 0:
            prediction = 'This cell is most likely to be Not Infected with Malarial Parasite.'
        else:
            prediction = 'This cell is most likely to be Infected with Malarial Parasite.'
        print(prediction)
        return render_template("result.html", prediction=prediction)


if __name__ == "__main__":
    app.run()
'''

'''
from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app2
app = Flask(__name__)

# Model saved with Keras model.save()
#MODEL_PATH = 'models/model_resnet.h5'

# Load your trained model
model = load_model('MNIST_classifier_nn_model.h5')
model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')

@app.route('/predict',methods=['POST'])
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('FashionClassifier.html')


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
    app.run(debug=True)
'''
