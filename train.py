import torch
from torch.utils.data import DataLoader
from dataset import SegmentationDataset
from model import get_model
import torch.nn as nn
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = SegmentationDataset("train_images", "train_masks")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = get_model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    total_loss = 0

    for images, masks in tqdm(loader):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader)}")

torch.save(model.state_dict(), "model.pth")
