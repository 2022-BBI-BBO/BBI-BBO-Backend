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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device 객체
    model = models.resnet34(pretrained=True)
    um_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2) # 2개 class 
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    model.load_state_dict(torch.load('./smoke_model.pt'))
    model.eval()
    
    image = Image.open('test_image2.jpg')
    image = transforms_test(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs,1)
        print(class_names[preds[0]])
        labels = labels.to(device)
        running_corrects += torch.sum(preds == labels.data)
        print(running_corrects)

        epoch_acc = running_corrects / 20 * 100
        print(epoch_acc)
    #     # imshow(image.cpu().data[0], title='예측 결과: ' + class_names[preds[0]])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device 객체
    print(device)
    return 'test'#class_names[preds[0]]
# _*_ coding: utf-8 _*_

# import logging
# from upbit import Upbit
# from flask import Flask, request, render_template

# app = Flask(__name__)
# upbit = Upbit()
# upbit.get_hour_candles('KRW-BTC')

# @app.route('/')
# def root():
#     market = request.args.get('market')
#     if market is None or market == '':
#         return 'No market parameter'

#     candles = upbit.get_hour_candles(market)
#     if candles is None:
#         return 'invalid market: {}'.format(market)

#     label = market
#     xlabels = []
#     dataset = []
#     i = 0
#     for candle in candles:
#         xlabels.append('')
#         dataset.append(candle['trade_price'])
#         i += 1
#     return render_template('chart.html', **locals())

def main():
    app.debug = True
    app.run(host="10.150.150.2", port="8080")


if __name__ == '__main__':
    main()
# if __name__ == "__main__":
#     app.run(host="10.150.150.2", port="8080")