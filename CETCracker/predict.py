import torch
import torch.nn as nn
from PIL import Image
from models import Models
from datasets import CaptchaData
from torchvision.transforms import Compose, ToTensor
import matplotlib.pyplot as plot

alphabet = 'abcdefghijklmnopqrstuvwxyz'


def predict(img_dir='./data/test', model_path='./checkpoints/model_resnet18.pth', arch='resnet18'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transforms = Compose([ToTensor()])
    dataset = CaptchaData(img_dir, transform=transforms)
    cnn = Models(arch=arch)
    if torch.cuda.is_available():
        cnn = cnn.cuda()
    cnn.eval()
    cnn.load_state_dict(torch.load(model_path))

    char_cnt = 0
    char_total = 0
    captcha_cnt = 0
    captcha_total = 0
    for k, (img, target) in enumerate(dataset):
        img = img.view(1, 1, 25, 80).to(device)
        target = target.view(1, 4 * 26).to(device)
        output = cnn(img)

        output = output.view(-1, 26)
        target = target.view(-1, 26)
        output = nn.functional.softmax(output, dim=1)
        output = torch.argmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        output = output.view(-1, 4)[0]
        target = target.view(-1, 4)[0]

        pred = ''.join([alphabet[i] for i in output.cpu().numpy()])
        true = ''.join([alphabet[i] for i in target.cpu().numpy()])

        if pred == true:
            print('pred: ' + pred, end='')
            print('\ttrue: ' + true)
            captcha_cnt += 1
            char_cnt += 4
        else:
            for i in range(len(pred)):
                char_cnt += int(pred[i] == true[i])
        captcha_total += 1
        char_total += 4


        # plot.imshow(img.permute((0, 2, 3, 1))[0].cpu().numpy())
        # plot.show()
    print('char accuracy: ' + str(char_cnt * 1.0 / char_total))
    print('captcha accuracy: ' + str(captcha_cnt * 1.0/captcha_total))


def predict_img(img_path='captcha.jpg', model_path='./checkpoints/model_resnet18.pth', arch='resnet18'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn = Models(arch=arch)
    if torch.cuda.is_available():
        cnn = cnn.cuda()
    cnn.eval()
    cnn.load_state_dict(torch.load(model_path, map_location='cpu' if torch.cuda.is_available() else None))

    transform = Compose([ToTensor()])
    img = Image.open(img_path).convert('1')
    img = transform(img)
    img = img.view(1, 1, 25, 80).to(device)
    output = cnn(img)
    output = nn.functional.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)
    output = output.view(-1, 4)[0]

    text = ''.join([alphabet[i] for i in output.cpu().numpy()])
    return text


if __name__ == "__main__":
    predict()