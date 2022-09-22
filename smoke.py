import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import datasets, models, transforms

img_width = 224
img_height = 224

def model_predict( p ):# model 예측 결과 넘겨주는 함수 인자로는 사진 위치를 받음
    class_names = ['담배피는사람','담배안피는사람']
    # model configuration
    device = torch.device("cpu")
    transforms_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    model = models.resnet34(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2) # 2개 class 
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.load_state_dict(torch.load('./smoke_model.pt',map_location='cpu'))
    model.eval()
    img_path = os.path.dirname(os.path.abspath('__file__'))
    img_path = p
    # 사진 받아서 저장하는 부분 근데 이제 쓸모 X 카메라 모듈로 저장함
    # result1 = request.files['chooseFile']
    # result1.save('./static/imgs/model/'+'model.jpg')
    # img_path += r'./static/imgs/model/model.jpg'
    image = Image.open(img_path).convert('RGB')
    image = image.resize( (img_width , img_height ) )
    image = transforms_test(image).unsqueeze(0).to(device)
    with torch.no_grad():
        running_loss = 0
        running_corrects = 0
        # labels = labels.to(device)
        outputs = model(image)
            
        _, preds = torch.max(outputs,1)
        result= class_names[preds[0]]

        if ( result == '담배피는사람' ):
            result = 1
        else :
            result = 0
    return result