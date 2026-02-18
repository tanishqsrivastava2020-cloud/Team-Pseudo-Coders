import torch
import cv2
import numpy as np
from model import get_model

device = "cuda" if torch.cuda.is_available() else "cpu"

model = get_model()
model.load_state_dict(torch.load("model.pth"))
model.to(device)
model.eval()

image = cv2.imread("test_images/test1.jpg")
image = cv2.resize(image, (256,256))
input_img = image.transpose(2,0,1)
input_img = torch.tensor(input_img, dtype=torch.float32).unsqueeze(0)/255.0
input_img = input_img.to(device)

with torch.no_grad():
    output = model(input_img)
    pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

cv2.imwrite("prediction.png", pred)
