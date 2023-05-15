from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import cv2

app = Flask(__name__)

dic = {0 :"Diabetic Retinopathy Not Detected", 1 : "Diabetic Retinopathy Detected"}
      

@app.route('/')
def home():
    return render_template('index.html')


img_size=224
model = load_model('model.h5')

model.make_predict_function()

def predict_label(img_path):
    img=cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    resized=cv2.resize(gray,(img_size,img_size)) 
    i = img_to_array(resized)/255.0
    i = i.reshape(1,img_size,img_size,1)
    predict_x=model.predict(i) 
    p=np.argmax(predict_x,axis=1)
    return dic[p[0]]


@app.route('/predict',methods=['POST'])
def predict():
#     '''
#     For rendering results on HTML GUI
    
#     '''

    if request.method == 'POST':
       img = request.files['file']
       img_path = "uploads/" + img.filename    
       img.save(img_path)
       p = predict_label(img_path)
       return str(p).lower()
       
    # if p == 0:
    #     return render_template("two.html", result="true")
    # elif p == 1:
    #     return render_template("one.html", result="true")
 

if __name__ == "__main__":
    app.run(debug=True)
