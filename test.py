import torch
import yaml
import os
import cv2
from model import UNet

with open("config.yaml") as f:
    config = yaml.safe_load(f)

device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

model = UNet(config["training"]["num_classes"]).to(device)
model.load_state_dict(torch.load(config["model"]["save_path"]))
model.eval()

test_dir = config["dataset"]["test_images"]

for img_name in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_name)
    image = cv2.imread(img_path)
    image = cv2.resize(image, (config["training"]["image_size"], config["training"]["image_size"]))
    image = image.transpose(2, 0, 1) / 255.0
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).cpu().numpy()[0]

    cv2.imwrite(f"prediction_{img_name}", prediction)

print("Testing complete.")
