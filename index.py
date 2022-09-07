from flask import Flask , render_template , Response
from PIL import Image
from picamera2 import Picamera2, Preview
from flask import jsonify
from flask import request
import os

from camera import Camera

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import datasets, models, transforms

import numpy as np
import time
import shlex, subprocess

comand = 'ls -l | grep ^- | wc -l'
def cam2():
    picam2 = Picamera2()
    camera_config = picam2.create_preview_configuration()
    picam2.configure(camera_config)
    # picam2.start_preview(Preview.DRM)
    # t1 = subprocess.check_output("cd imgs/; ls -l | grep ^- | wc -l; cd ../",shell=True, encoding='utf-8')
    # t2 = subprocess.call("cd imgs/; ls -l | grep ^- | wc -l; cd ../",shell=True)
    # t1 = str(t1)
    # t1 = t1[:len(t1)-3]
    # print(f"t1 : {t1} t2 : {t2}")
    picam2.start()
    # p = subprocess.call("cd ./static/imgs/; ls -l | grep ^- | wc -l; cd ../",shell=True)
    p = subprocess.check_output("cd static/imgs/; ls -l | grep ^- | wc -l; cd ../..",shell=True, encoding='utf-8')
    p =str(p)
    p = p[:len(p)-3]
    print(p)
    picam2.capture_file(f"./static/imgs/pic{p}.jpg")
    pic_path = f"./static/imgs/pic{p}.jpg"

    return p , pic_path
    
img_width = 224
img_height = 224

# variable
device = torch.device("cpu")
transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class_names = ['담배피는사람','담배안피는사람']

model = models.resnet34(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2) # 2개 class 
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model.load_state_dict(torch.load('./smoke_model.pt',map_location='cpu'))
model.eval()
app = Flask(__name__)

@app.route("/flasktest")
def helloworld():
    return "hello flask!!!!!!!"

@app.route("/rq", methods=['GET','POST'])
def smoke_rq():
    return render_template('index.html')

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

@app.route("/hello/<_name>")
def hello(_name):
   return render_template('page.html', name=_name)


@app.route("/recive", methods=['GET','POST'])
def smoke_rp():

    result1 = request.files['chooseFile']
    result1.save('./static/imgs/'+'sss.jpg')

    img_path = os.path.dirname(os.path.abspath('__file__'))
    img_path += r'/static/imgs/sss.jpg'

    image = Image.open(img_path).convert('RGB')
    image = image.resize( (img_width , img_height ) )
    image = transforms_test(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        running_loss = 0
        running_corrects = 0
        # labels = labels.to(device)
        outputs = model(image)
            
        _, preds = torch.max(outputs,1)
        result1= class_names[preds[0]]
        test_acc = accuracy(image,preds[0])
        # print(_)
        # print(preds)
        # print(torch.max(outputs,1))
        print(test_acc)
    return render_template('index.html',result1=result1,test_acc=test_acc)

@app.route("/test")
def test_pic():
    p , pic_path= cam2()
    print(p)
    print(pic_path)
    return render_template('pic.html',p=p,pic_path=pic_path)

def main():
    app.debug = True
    app.run(host="10.150.150.2", port="8080")


if __name__ == '__main__':
    main()