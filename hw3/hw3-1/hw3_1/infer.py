from baseline import save_imgs
from model import Generator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument("-m","--model", type=str, default='models/test/G4.pkl', help="path of the model for inference")
parser.add_argument("-o","--output", type=str, default='output.png', help="file of output")
opt = parser.parse_args()
print(opt)

if __name__ == "__main__":
    model = Generator().to(device)
    model.load_state_dict(torch.load(opt.model, map_location=device))
    save_imgs(model,opt)
