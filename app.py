import sys
import io
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import os
import time
import flask
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

from imageai.Detection import ObjectDetection


app = Flask(__name__)


@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'

    return response
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/indexcatdog', methods=['GET'])
def project():

    return render_template('images.html')

@app.route('/predictcatdog', methods=['GET', 'POST'])
def upload_catdog():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
   
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
       
       
    
        name,file= predict_objects(file_path)

    
        #pytesseract.pytesseract.tesseract_cmd = './bin/tesseract'
        pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
        textpresent=True
        objectpresent=True
        im = Image.open(file_path)
        im = im.filter(ImageFilter.MedianFilter())
        enhancer = ImageEnhance.Contrast(im)
        im = enhancer.enhance(2)
        im = im.convert('1')
        im.save('temp2.jpg')
        text = pytesseract.image_to_string(Image.open('temp2.jpg'))

        if(len(file)==0):
            objectpresent=False
        if(len(text)==0):
            textpresent=False

        print(text)

        

        return render_template('result.html',text=text,textpresent=textpresent,objectpresent=objectpresent,file=file, nameobj=name,len=len(name))
    return None   



def predict_objects(imagepath):
    

    new_graph_name = "graph" + str(time.time()) + ".jpg"
    fullfilepath="static/"+new_graph_name

    for filename in os.listdir('static/'):
        if filename.startswith('graph'):  # not to remove other images
            os.remove('static/' + filename)
    
    objectx=[]
    probability=[]
    detector = ObjectDetection()
    detector.setModelTypeAsTinyYOLOv3()
    detector.setModelPath("yolo-tiny.h5")
    detector.loadModel()
    detection = detector.detectObjectsFromImage(input_image=imagepath, output_image_path=fullfilepath)
    for eachItem in detection:
        print(eachItem["name"] , " : ", eachItem["percentage_probability"])
        final= str(eachItem["name"])+ " : " + str(eachItem["percentage_probability"]) 
        objectx.append(final)

    return objectx,new_graph_name


if __name__ == '__main__':
    app.run(debug=True)     