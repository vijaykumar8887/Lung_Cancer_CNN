#!/usr/bin/env python
import base64
import urllib
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps
# Flask utils
from flask import Flask, request, render_template
from keras.src.saving import load_model

# from skimage import io

app = Flask(__name__)


# Load your trained model


@app.route("/")
@app.route("/first")
def first():
    return render_template('first.html')


@app.route("/login")
def login():
    return render_template('login.html')


@app.route("/chart")
def chart():
    return render_template('chart.html')


@app.route("/performance")
def performance():
    return render_template('performance.html')


@app.route("/index", methods=['GET'])
def index():
    return render_template('index.html')


@app.route("/upload", methods=['POST'])
def upload_file():
    print("Hello")
    try:
        img = Image.open(BytesIO(request.files['imagefile'].read())).convert('RGB')
        img = ImageOps.fit(img, (224, 224), Image.ANTIALIAS)
    except:
        error_msg = "Please choose an image file!"

    # Call Function to predict
    args = {'input': img}
    out_pred, out_prob = predict(args)
    out_prob = out_prob * 100

    print(out_pred, out_prob)
    danger = "danger"
    if out_pred == "You Are Safe, But Do keep precaution":
        danger = "success"
    print(danger)
    img_io = BytesIO()
    img.save(img_io, 'PNG')

    png_output = base64.b64encode(img_io.getvalue())
    processed_file = urllib.parse.quote(png_output)

    return render_template('result.html', **locals())


def predict(args):
    img = np.array(args['input']) / 255.0
    img = np.expand_dims(img, axis=0)

    model = 'cancer.h5'
    # Load weights into the new model
    model = load_model(model)

    pred = model.predict(img)

    if np.argmax(pred, axis=1)[0] == 0:
        out_pred = "Cancerous"
    elif np.argmax(pred, axis=1)[0] == 1:
        out_pred = "non-Cancerous"

    return out_pred, float(np.max(pred))


if __name__ == '__main__':
    app.run(debug=True)

