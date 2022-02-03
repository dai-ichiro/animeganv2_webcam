import torch
from  torchvision import transforms
import cv2
import numpy as np
from autogluon.core.utils import download, mkdir

device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="paprika")
net.eval().to(device)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img_tensor = transforms.ToTensor()(frame).unsqueeze(0)
    img_tensor = -1 + 2 * img_tensor

    with torch.no_grad():
        out = net(img_tensor.to(device))

    out= out[0].to('cpu').numpy()
    out = ((out + 1) * 127.5).clip(0, 255).astype('uint8')
    out = np.transpose(out, (1, 2, 0))

    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    cv2.imshow('animegan2', out)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
