# Team-Pseudo-Coders
# Duality AI Offroad Semantic Segmentation

## Setup

conda create -n EDU python=3.9
conda activate EDU
pip install -r requirements.txt

## Train

python train.py

## Test

python test.py

## Output

- Model weights saved as model.pth
- Predictions saved as prediction_*.png

## Notes

- Train only on train/val folders
- Do NOT use testImages for training
- IoU metric included in utils.py
