import torch
import numpy as np

def calculate_iou(pred, mask, num_classes):
    pred = torch.argmax(pred, dim=1).cpu().numpy()
    mask = mask.cpu().numpy()

    ious = []

    for cls in range(num_classes):
        intersection = np.logical_and(pred == cls, mask == cls).sum()
        union = np.logical_or(pred == cls, mask == cls).sum()

        if union == 0:
            continue
        ious.append(intersection / union)

    return np.mean(ious)
