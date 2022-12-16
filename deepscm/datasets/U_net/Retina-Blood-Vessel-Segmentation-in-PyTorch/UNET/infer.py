import os, time
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import imageio
import torch
import torch
import torchvision as tv


from model import build_unet
from utils import create_dir, seeding

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask

""" Seeding """
seeding(42)


""" Load dataset """
test_x = sorted(glob("/mnt/alpha/diabetes/MS/data/images_512/*"))
# test_x  = test_x[:100]
""" Hyperparameters """
H = 512
W = 512
size = (W, H)
checkpoint_path = "files/checkpoint.pth"

""" Load the checkpoint """
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = build_unet()
model = model.to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

time_taken = []
transform_list = []
transform_list += [tv.transforms.ToPILImage()]
transform = tv.transforms.Compose(transform_list)


for i, (x) in tqdm(enumerate(test_x), total=len(test_x)):
    """ Extract the name """
    name = x.split("/")[-1].split(".")[0]

    """ Reading image """
    image = cv2.imread(x, cv2.IMREAD_COLOR) ## (512, 512, 3)
    ## image = cv2.resize(image, size)
    x = np.transpose(image, (2, 0, 1))      ## (3, 512, 512)
    x = x/255.0
    x = np.expand_dims(x, axis=0)           ## (1, 3, 512, 512)
    x = x.astype(np.float32)
    x = torch.from_numpy(x)
    x = x.to(device)

    with torch.no_grad():
        """ Prediction and Calculating FPS """
        start_time = time.time()
        pred_y = model(x)
        pred_y = torch.sigmoid(pred_y)

        pred_y = pred_y[0].cpu().numpy()        ## (1, 512, 512)
        pred_y = np.squeeze(pred_y, axis=0)     ## (512, 512)
        pred_y = pred_y > 0.5
        pred_y = np.array(pred_y, dtype=np.uint8)

    """ Saving masks """
    # pred_y = mask_parse(pred_y)
    img = transform(pred_y*255)
    img.save(f"/mnt/alpha/diabetes/MS/data/images_masks_512/{name}.tiff")

total_time = time.time() - start_time
print(total_time)

    # cv2.imwrite(f"/mnt/alpha/diabetes/MS/data/images_mask_512/{name}.png", pred_y)