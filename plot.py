from data import plot_slices, mosmed_dataloaders
import torch
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
fold = 1
model_file = f'model/seg_{fold}.pth'

from model import Unet3d
from seg_model import UNet3dLite
start_epoch = 0
metrics = {}

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

if Path(model_file).exists():
    # model = Unet3d()
    model = UNet3dLite().to(device)
    ckpt = torch.load(model_file)
    model.load_state_dict(ckpt['model'])

import pdb

def plot_slices(n_rows, n_cols, width, height, data, mask, filename = 'figs/file.png', mask_color = [138,43,226]):
    # data = np.rot90(data, k = 1, axes = (0, 1))
    fig_width = 12.0
    fig_height = 12.0 / (n_cols * width) * (n_rows * height)

    fig, ax = plt.subplots(n_rows, n_cols, figsize = (fig_width, fig_height), dpi = 250)

    for i in range(n_rows):
        for j in range(n_cols):
            img = (data[:, :, i, j].numpy() * 255.).astype(np.uint8)
            # pdb.set_trace()
            stacked_img = np.stack((img,)*3, axis=-1)
            for x in range(len(img)):
                for y in range(len(img[x])):
                    if mask[x,y, i, j]==1:
                        stacked_img[x,y] = mask_color

            ax[i,j].imshow(stacked_img)
            ax[i,j].axis('off')

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.savefig(filename)

import random
if __name__ == '__main__':
    train_loader, val_loader = mosmed_dataloaders(6, 8, val_split=.2, kfold = True, only_mask_samples = True)[4]
    
    # processed_scan, processed_mask, score, has_mask = next(iter(train_loader))

    for i, (processed_scan, processed_mask, score, has_mask) in enumerate(train_loader):
        scan = processed_scan.permute(0, -1, 1, 2).unsqueeze(1).type(torch.float32).to(device)
        pred = model(scan)
        pred = torch.where(pred[:, 0] < .5, 0., 1.)
        
        for index in range(len(processed_scan)):
            p_scan = processed_scan[index]
            p_mask = processed_mask[index]
            p_mask = torch.where(p_mask > .5, 1, 0)
            n_rows = 4
            n_cols = 16

            pred_mask = pred[index].permute(1,2,0)
            transformed_scan = p_scan.reshape(p_scan.shape[0], p_scan.shape[1], n_rows, n_cols)
            transformed_mask = p_mask.reshape(p_mask.shape[0], p_mask.shape[1], n_rows, n_cols)
            pred_mask = pred_mask.reshape(pred_mask.shape[0], pred_mask.shape[1], n_rows, n_cols)

            plot_slices(n_rows, n_cols, transformed_scan.shape[0], transformed_scan.shape[1], transformed_scan, transformed_mask, filename = f'figs/real{i}_{index}.png', mask_color = [138,43,226])
            plot_slices(n_rows, n_cols, transformed_scan.shape[0], transformed_scan.shape[1], transformed_scan, pred_mask, filename = f'figs/pred{i}_{index}.png', mask_color = [218,112,214])
        del pred, pred_mask
        torch.cuda.empty_cache()
        
    exit(0)


    index = random.randint(0, 6)

    scan = processed_scan.permute(0, -1, 1, 2).unsqueeze(1).type(torch.float32).to(device)
    pred = model(scan)
    pred = torch.where(pred[:, 0] < .5, 0., 1.)

    q = 0
    chosen= 0
    for index in range(6):
        p_scan = processed_scan[index]; p_mask = processed_mask[index]
        p_mask = torch.where(p_mask > .5, 1, 0)
        if p_mask.sum().item() > q:
            q = p_mask.sum().item()
            chosen = index
    
    print(chosen)
    processed_scan = processed_scan[chosen]
    processed_mask = processed_mask[chosen]
    processed_mask = torch.where(processed_mask > .5, 1, 0)

    pred_mask = pred[chosen].permute(1,2,0)

    n_rows = 4
    n_cols = 16
    
    # pdb.set_trace()
    transformed_scan = processed_scan.reshape(processed_scan.shape[0], processed_scan.shape[1], n_rows, n_cols)
    transformed_mask = processed_mask.reshape(processed_mask.shape[0], processed_mask.shape[1], n_rows, n_cols)
    pred_mask = pred_mask.reshape(pred_mask.shape[0], pred_mask.shape[1], n_rows, n_cols)


    plot_slices(n_rows, n_cols, transformed_scan.shape[0], transformed_scan.shape[1], transformed_scan, transformed_mask, filename = 'figs/real.png', mask_color = [138,43,226])
    plot_slices(n_rows, n_cols, transformed_scan.shape[0], transformed_scan.shape[1], transformed_scan, pred_mask, filename = 'figs/pred.png', mask_color = [218,112,214])
    # plot_slices(n_rows, n_cols, transformed_mask.shape[0], transformed_mask.shape[1], transformed_mask, cmap = 'binary')
