from tensorflow import keras
from keras.applications.vgg16 import VGG16, decode_predictions
import numpy as np
from PIL import Image

vgg = VGG16()

for fname in ['bird1', 'bird2', 'cat2', 'dog2']:
    filename = "static/images/uploads/" + fname + ".jpg"
    img = np.array(Image.open(filename).resize((224, 224)))

    yhat = vgg.predict(img.reshape(-1, 224, 224, 3))
    label_key = np.argmax(yhat)
    label = decode_predictions(yhat)
    label = label[0][0]
    pct = '%.2f' % (label[2]*100)
    print(label[1], pct)