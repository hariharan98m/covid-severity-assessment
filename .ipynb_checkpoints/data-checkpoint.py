import nibabel as nib
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import pdb
import tqdm
from torch.utils.data.dataset import Subset

def read_nifti_file(filepath):
    # read file
    scan = nib.load(filepath)
    # get raw data volume.
    scan = scan.get_fdata()
    return scan

def normalize(volume, min=-1000, max=400):
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    return volume.astype(float)

def resize_volume(volume, desired_width, desired_height, desired_depth, curr_width, curr_height, curr_depth):
    # rotate
    volume = ndimage.rotate(volume, 90, reshape=False)
    # resize volume along z-axis
    volume = ndimage.zoom(volume, (desired_height/curr_height, desired_width/curr_width, desired_depth/curr_depth), order=1)
    return volume

def process_scan(path, maskpath = None):
    mask = None # init mask
    # read scan
    volume = read_nifti_file(path)

    if maskpath is not None:
        mask = read_nifti_file(maskpath)

    # norm
    volume = normalize(volume)
    # resize width, height, depth
    volume = resize_volume(volume, desired_width=128, desired_height=128, desired_depth= 64, curr_width= volume.shape[1], curr_height= volume.shape[0], curr_depth=volume.shape[2])

    if maskpath is not None:
        # resize the mask in the same fashion
        mask = resize_volume(mask, desired_width=128, desired_height=128, desired_depth= 64, curr_width= mask.shape[1], curr_height= mask.shape[0], curr_depth=mask.shape[2])
    return volume, mask

def plot_slices(n_rows, n_cols, width, height, data, cmap = 'gray', filename = 'figs/file.png'):
    # data = np.rot90(data, k = 1, axes = (0, 1))
    fig_width = 12.0
    fig_height = 12.0 / (n_cols * width) * (n_rows * height)

    fig, ax = plt.subplots(n_rows, n_cols, figsize = (fig_width, fig_height))

    for i in range(n_rows):
        for j in range(n_cols):
            ax[i,j].imshow(data[:, :, i, j], cmap = cmap)
            ax[i,j].axis('off')

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.savefig(filename)

from torch.utils.data import DataLoader, Dataset, sampler, Subset
import torch
from collections import defaultdict
from pathlib import Path

def weighted_sampler(scores):
    counts = defaultdict(int)
    for score in scores: counts[score]+=1
    weight_per_class = {cls: len(scores)/count for cls, count in counts.items()}
    sample_weights = [weight_per_class[score] for score in scores]
    # pdb.set_trace()
    return sampler.WeightedRandomSampler(sample_weights, len(sample_weights))

class CTDataset(Dataset):
    def __init__(self, studiespath = None, maskpath = None, transform = None):
        super().__init__()
        self.data_items = []
        for cls, score in zip(['CT-0', 'CT-1', 'CT-2', 'CT-3', 'CT-4'], [0., 2.5, 5., 7.5, 10.]):
            for path in (Path(studiespath) / cls).iterdir():
                # study label
                study_label = path.name.split('.', 1)[0]
                # item tuple.
                item = [str(path), None, score/10.]
                maskfile = Path(maskpath) / (study_label+ '_mask.nii.gz')
                if (maskfile).exists():
                    item[1] = str(maskfile)
                self.data_items.append(item)

    def __len__(self):
        return len(self.data_items)

    def process(self, scanpath, maskpath, score, file):
        scan, mask = process_scan(scanpath, maskpath)
        has_mask = 1
        if mask is None:
            mask = np.zeros_like(scan); has_mask = 0
        torch.save([torch.tensor(scan), torch.tensor(mask), torch.tensor([score]), torch.tensor(has_mask)], file)

    def __getitem__(self, index):
        item = self.data_items[index]

        study_label = Path(item[0]).name.split('.', 1)[0]
        file = f'processed_data/{study_label}_{item[2]}.pth'

        if not Path(file).exists():
            self.process(*item, file)
        else:
            out = torch.load(file)
            if out[0].size()[2]==128:
                self.process(*item, file)

        return torch.load(file)

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold

def mosmed_dataloaders(train_batch_size, val_batch_size, val_split = 0.2, kfold = False, only_mask_samples = False):
    dataset = CTDataset(studiespath= 'mosmed/studies',
                        maskpath= 'mosmed/masks/')

    if only_mask_samples:
        mask_indices = [i for i, data in enumerate(dataset) if data[3].item()==1]
        dataset = Subset(dataset, mask_indices)

    classes = np.array([int(score*10.) for _, _, score, _ in tqdm.tqdm(dataset)])

    if kfold:
        
        skf = StratifiedKFold(n_splits = 5, random_state = 42)
        
        # arrange into folds.
        folds = {}
        
        for fold, (train_indices, val_indices) in enumerate(skf.split(list(range(len(classes))), classes)):

            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)
            
            train_sampler = weighted_sampler(classes[train_indices])
            val_sampler = weighted_sampler(classes[val_indices])
            
            train_loader = DataLoader(train_dataset, batch_size= train_batch_size, num_workers = 4, shuffle = False) #, sampler = train_sampler
            val_loader = DataLoader(val_dataset, batch_size= val_batch_size, num_workers = 4, sampler = val_sampler)

            folds[fold] = (train_loader, val_loader)
        
        return folds

    else:

        split_strategy = StratifiedShuffleSplit(n_splits=1, random_state= 42, test_size= val_split)

        splits = split_strategy.split(list(range(len(classes))), classes)
        train_indices, val_indices = next(iter(splits))

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        
        train_sampler = weighted_sampler(classes[train_indices])
        val_sampler = weighted_sampler(classes[val_indices])
        
        train_loader = DataLoader(train_dataset, batch_size= train_batch_size, num_workers = 4, sampler = train_sampler)
        val_loader = DataLoader(val_dataset, batch_size= val_batch_size, num_workers = 4, sampler = val_sampler)

        return train_loader, val_loader

if __name__ == '__main__':
    train_loader, val_loader = mosmed_dataloaders(8, 8, val_split=.2, kfold = True, only_mask_samples = True)
    print(next(iter(train_loader)))
    exit(0)
    scanfile = 'mosmed/studies/CT-1/study_0255.nii.gz'
    maskfile = 'mosmed/masks/study_0255_mask.nii.gz'
    data = read_nifti_file(scanfile)
    # print(data.shape, np.min(data), np.max(data))

    # processed_scan, processed_mask =process_scan(scanfile, maskfile)
    # dataset = CTDataset(studiespath= '/Users/hariharan/hari_works/nonlocal_se/mosmed/studies/',
    #                     maskpath= '/Users/hariharan/hari_works/nonlocal_se/mosmed/masks/')

    processed_scan, processed_mask, score = dataset[0]
    for (scan, mask, score) in dataset:
        print(scan.shape, mask.shape if mask is not None else None, score)

    n_rows = 8
    n_cols = 8
    transformed_scan = processed_scan.reshape(processed_scan.shape[0], processed_scan.shape[1], n_rows, n_cols)
    # transformed_mask = processed_mask.reshape(processed_mask.shape[0], processed_mask.shape[1], n_rows, n_cols)

    plot_slices(n_rows, n_cols, transformed_scan.shape[0], transformed_scan.shape[1], transformed_scan)
    # plot_slices(n_rows, n_cols, transformed_mask.shape[0], transformed_mask.shape[1], transformed_mask, cmap = 'binary')
