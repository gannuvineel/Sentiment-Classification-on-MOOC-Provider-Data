# Import packages
import numpy as np 
from flask import Flask, request, render_template 
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import text
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import sys
import h5py

from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import matplotlib
matplotlib.use('Agg')

import io
from PIL import Image
import base64

from google.cloud import storage

from datetime import datetime

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# Instance of Flask class
app = Flask(__name__)

# Defaults Cache control
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Loads the model from the model.h5 file
model = load_model('model.h5')

@app.route('/')
# Renders maincontent.html file content
def home():
    return render_template('maincontent.html')

@app.route('/predict',methods=['POST'])
# Get the input and predict the output using model
def predict():
    client = storage.Client(project='eng-voice-297302')
    bucket = client.bucket('eng-voice-297302.appspot.com')

    imagename = 'graph' + str(datetime.now()) + '.png'

    print('ImageName', imagename, file=sys.stderr)

    blob = bucket.blob('graph/' + imagename)

    # Gets the values from the request form
    review = [x for x in request.form.values()]

    # Opens the tokenizer.pickle and loads the tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Tokenizes the input review into sequences
    input_transformed = tokenizer.texts_to_sequences(review)

    # Pads the sequences with max length of 500
    input_transformed_pad = pad_sequences(input_transformed, maxlen=500)

    # Model prediction
    prediction = model.predict(input_transformed_pad)

    # Build a bar graph and save it as an image
    fig = create_figure(prediction[0])
    output = io.BytesIO()

    # Calls the create_figure function
    FigureCanvas(fig).print_png(output)             
    plt.savefig(output, format='png')

    # upload buffer contents to gcs
    print(blob.upload_from_string(output.getvalue(), content_type='image/png'), file=sys.stderr)
    print('URL: ', blob.public_url,file=sys.stderr)
    output.close()

    filename = blob.public_url
    return render_template('maincontent.html', prediction_text='The predicted sentiment values for the given review are: <br/> <br/> {}'.format(prediction[0]), graph = filename)    



# create_figure function creates the figure
def create_figure(data):
    labels = ['1', '2', '3', '4', '5']
    accuracy = data
    fig = plt.figure()
    plt.ylim(0, 1)
    plt.xlabel('Class Label', fontsize=16)
    plt.ylabel('Probability', fontsize=16)
    fig.patch.set_facecolor('#F0F8FF')
    plt.bar(labels, accuracy)
    return fig  

@app.after_request
# add_header function is used to control the cache after request is processed
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response      

if __name__ == "__main__":
    prediction = []
    app.run(debug=True) 