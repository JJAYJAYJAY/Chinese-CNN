import os.path
import pickle

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


class ResNetModel(nn.Module):
    def __init__(self, num_class, kernel=(7,7), stride=(2,2)):
        super(ResNetModel, self).__init__()

        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=kernel, stride=stride, padding=(3, 3), bias=False)

        self.resnet.fc = nn.Sequential(
            nn.Linear(512, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_class)
        )

    def forward(self, x):
        x = self.resnet(x)
        return x


def get_chinese_dict(path):
    with open(path, 'rb') as file:
        char_dict = pickle.load(file)
        # 将键和值的顺序互调
        reversed_dict = {str(value): key for key, value in char_dict.items()}
        return reversed_dict


class ChineseOcrModel:
    def __init__(self, pth_path, num_class, image_size=224,kernel=(7,7),stride=(2,2)):
        self.model = ResNetModel(num_class,kernel,stride=stride)
        self.model.eval()
        checkpoint = torch.load(pth_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.chinese_dict = get_chinese_dict(os.path.abspath('./model/char_dict'))
        self.image_size = image_size

    def inference_single_char(self, char_input):
        # 图片预处理
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

        # 加载图片
        char_input = transform(char_input)
        char_input = char_input.unsqueeze(0)

        # 进行预测
        output = self.model(char_input)
        _, pred = torch.max(output.data, 1)
        return ('%4d' % pred).strip()

    def inference(self, divideCharImg):
        predictions = []

        for item in divideCharImg:
            line = []
            for char in item['characters']:
                prediction = self.inference_single_char(char)
                chinese_label = self.chinese_dict.get(prediction, "未知标签")
                line.append(chinese_label)
            predictions.append(line)
        return predictions
