from flask import Flask,render_template,request
from scipy.misc import imread,imresize
import numpy as np
import re
import pybase24
from load import init

global graph,model

model,graph = init()

app = Flask(__name__)

@app.route('/')
def index_view():
    return render_template('index.html')

def convert_image(img_data):
    img_str = re.search(b'base64,(.*)',img_data).group(1)
    with open('output.png','wb') as output:
        output.write(pybase24.b64decode(img_str))

@app.route('/predict/',methods=['GET','POST'])
def predict():
    image_data = request.get_data()
    convert_image(image_data)
    x = imread('output.png',mode='L')
    x = np.invert(x)
    x = imresize(x,(28,28))
    x = x.reshape(1,28,28,1)

    with graph.as_default():
        out = model.predict(x)
        print(out)
        print(np.argmax(out,axis=1))

        response = np.array_str(np.argmax(out,axis=1))
        return response  

if __name__=='__main__':
    app.run(debug=True,port=8000,host='0.0.0.0')