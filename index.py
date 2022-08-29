from flask import Flask , render_template
from PIL import Image
from flask import jsonify

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import datasets, models, transforms

import numpy as np
import time

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
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model.load_state_dict(torch.load('./smoke_model.pt',map_location='cpu'))
model.eval()

app = Flask(__name__)

@app.route("/flasktest")
def helloworld():
    return "hello flask!!!!!!!"

@app.route("/smoketest")
def smoke_rq():
    return render_template('index.html')

@app.route("/recive")
def smoke_rp():
    running_corrects = 0
    image = Image.open('test_image2.jpg')
    image = transforms_test(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs,1)
        print(class_names[preds[0]])

        # labels = labels.to(device)
        # running_corrects += torch.sum(preds == labels.data)
        # print(running_corrects)
        # epoch_acc = running_corrects / 20 * 100
        # print(epoch_acc)
    return 'testing'

def main():
    app.debug = True
    app.run(host="10.150.150.2", port="8080")


if __name__ == '__main__':
    main()