import torch
from  torchvision import transforms
from torchvision.datasets.utils import download_url
import cv2
import numpy as np

model_url = 'https://github.com/bryandlee/animegan2-pytorch/raw/main/model.py'
model_fname = model_url.split('/')[-1]
download_url(model_url, root = '.', filename = model_fname)
from model import Generator

device = 'cuda' if torch.cuda.is_available() else 'cpu'

checkpoint = './pytorch_generator_Paprika.pt'

net = Generator()
net.load_state_dict(torch.load(checkpoint))
net.eval().to(device)

print(f"model loaded: {checkpoint}")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

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