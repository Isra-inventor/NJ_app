from keras.models import model_from_json
import tensorflow as tf
def init():
  json_file = open('/NJ_app/model.json','r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  #load weights into new model
  loaded_model.load_weights("/NJ_app/model (1).h5")
  print("Loaded Model from disk")
  #compile and evaluate loaded model
  loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
  graph = tf.compat.v1.get_default_graph()
  return loaded_model,graph
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation , GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import matplotlib.image as mpimg
from keras.preprocessing.image import img_to_array
import cv2
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.preprocessing import image
from keras.models import load_model
from flask import *
from flask_ngrok import run_with_ngrok
import os
from werkzeug.utils import secure_filename
from IPython.display import Image
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
 
global model, graph
# initialize these variables
model, graph = init()
 
UPLOAD_FOLDER = '/content/static'
app = Flask(__name__)  
run_with_ngrok(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/')  
def upload():  
    return render_template("file_upload.html")  
 
@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename)  
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        imr=image.load_img('/content/'+ filename,target_size=(200,200,3))
        plt.imshow(imr)
        plt.show()
        X = image.img_to_array(imr)
        X=X/255
        X= np.expand_dims(X,axis=0)
        images=np.vstack([X])
        val= model.predict(images)
        return render_template("success.html", name = f.filename, pred = np.argmax(val), im="/static/"+f.filename)  
 
if __name__ == '__main__':  
    app.run()
