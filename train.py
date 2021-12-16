import torch
import torch.nn as nn
from model import CNN3d, Unet3d
from data import CTDataset, mosmed_dataloaders
import pdb

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# train model.
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, classification_report
import tqdm

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
 
    def forward(self, inputs, targets, smooth=1):        
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        
        return 1 - dice

def get_class(prob):
    if prob < 0: return 0
    if prob > 1: return 1
    bands = [0, .25, .5, .75, 1.]
    for cls, b in enumerate(bands):
        if prob > b-.125 and prob <= b+.125:
            return cls

def val_step(data_loader, model, desc):
    model.eval()
    with torch.no_grad():
        labels= []; preds = []
        for i, batch in tqdm.tqdm(enumerate(data_loader), total = len(data_loader), desc = desc):
            scan, y = batch[0].to(device), batch[2].to(device)
            scan = scan.permute(0, -1, 1, 2).unsqueeze(1).type(torch.float32)
            pred = model(scan)
            preds.extend(pred[:, 0].cpu().numpy().tolist()); labels.extend(y[:, 0].cpu().numpy().tolist())
            scan.detach(); y.detach(); pred.detach()
            del scan, batch, pred, y
            torch.cuda.empty_cache()
        
        class_labels = [get_class(c) for c in labels] 
        class_preds = [get_class(p) for p in preds]
        # pdb.set_trace()
        return {
           'mse': mean_squared_error(labels, preds),
           'mae': mean_absolute_error(labels, preds), 
           'r2': r2_score(labels, preds),
           'acc': accuracy_score(class_labels, class_preds),
           'pre': precision_score(class_labels, class_preds, average = 'weighted'),
           'rec': recall_score(class_labels, class_preds, average = 'weighted')
        }


def val_step_seg(data_loader, model, desc):
    model.eval()
    with torch.no_grad():
        pre = 0.; rec = 0.; dice = 0.
        for i, batch in tqdm.tqdm(enumerate(data_loader), total = len(data_loader), desc = desc):
            scan, y = batch[0].to(device), batch[2].to(device)
            scan = scan.permute(0, -1, 1, 2).unsqueeze(1).type(torch.float32)
            pred = model(scan)
            # pred = torch.argmax(pred, dim = 1).type(torch.float32).view(-1)
            pred = torch.where(pred[:, 0] < .5, 0., 1.).view(-1)
            
            mask = batch[1].permute(0, -1, 1, 2).to(device)
            mask = torch.where(mask <.5, 0., 1.).view(-1)
            # pdb.set_trace()
            tp = (pred * mask).sum()
            fp = (pred * (1-mask)).sum()
            fn = ((1-pred) * mask).sum()

            precision = tp / (tp+fp)
            recall = tp/(tp+fn)
            
            smooth = 1             
            d = (2.*tp + smooth)/(pred.sum() + mask.sum() + smooth)

            pre+= precision.item(); rec+= recall.item(); dice+= d.item()
            del scan, batch, pred, y, precision, recall, d
            torch.cuda.empty_cache()

        pre/=len(data_loader)
        rec/=len(data_loader)
        dice/=len(data_loader)
        # pdb.set_trace()
        return {
           'pre': pre,
           'rec': rec,
           'dice': dice
        }

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_seg(train_loader, val_loader, fold):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss(reduction='mean')
    entropy_criterion = nn.CrossEntropyLoss()
    dice_criterion = DiceLoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    num_epochs = 50
    prev_val_dice = -2
    from pathlib import Path
    model_file = f'model/myseg_{fold}.pth'

    start_epoch = 0
    metrics = {}
    if Path(model_file).exists():
        ckpt = torch.load(model_file)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        metrics = ckpt['metrics']
        start_epoch = list(metrics)[-1]
        prev_val_dice = metrics[start_epoch]['val']['dice']

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        l_sum = 0
        for i, batch in tqdm.tqdm(enumerate(train_loader), total = len(train_loader)):
            
            scan, y = batch[0].to(device), batch[2].to(device)
            mask = batch[1].permute(0, -1, 1, 2).to(device)
            # only for seg.
            mask = torch.where(mask <.5, 0., 1.)

            scan = scan.permute(0, -1, 1, 2).unsqueeze(1).type(torch.float32)
            pred = model(scan)
            # only for seg.
            pred = pred[:, 0]

            optimizer.zero_grad()
            # loss = criterion(pred, y)
            loss = dice_criterion(pred, mask)
            # loss = entropy_criterion(pred, mask)

            loss.backward()
            
            print('epoch {}/{} iter {}/{} train loss - {:.2f}'.format(epoch+1, start_epoch + num_epochs, i+1, len(train_loader), loss))
            optimizer.step()
            l_sum+=loss.item()
            scan.detach(); y.detach(); pred.detach(); loss.detach()
            del scan, y, pred, loss, batch
            torch.cuda.empty_cache()
            # break
        scheduler.step()
        print('-'*60)
        curr_loss = l_sum/len(train_loader)
        train_report = val_step_seg(train_loader, model, 'train')
        val_report = val_step_seg(val_loader, model, 'val')
        metrics[epoch] = {
            'train': train_report,
            'val': val_report
        }
        if val_report['dice'] > prev_val_dice:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'start_epoch': epoch,
                'metrics': metrics,
            }, model_file)
            prev_val_acc = val_report['dice']
        else:
            ckpt = torch.load(model_file)
            ckpt['metrics'] = metrics; ckpt['start_epoch'] = epoch
            torch.save(ckpt, model_file)

        print('epoch stats: lr: {:.2f} avg train loss- {:.2f} avg val dice: {:.2f}'.format(get_lr(optimizer), curr_loss, val_report['dice']))
        print('-'*60)


def train(train_loader, val_loader, fold):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss(reduction='mean')
    dice_criterion = DiceLoss()
    num_epochs = 20
    prev_val_acc = -2
    from pathlib import Path
    model_file = f'model/cnn_{fold}.pth'

    start_epoch = 0
    metrics = {}
    if Path(model_file).exists():
        ckpt = torch.load(model_file)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        metrics = ckpt['metrics']
        start_epoch = list(metrics)[-1]
        prev_val_acc = metrics[start_epoch]['val']['acc']

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        l_sum = 0
        for i, batch in tqdm.tqdm(enumerate(train_loader), total = len(train_loader)):
            
            scan, y = batch[0].to(device), batch[2].to(device)
            mask = batch[1].to(device)

            scan = scan.permute(0, -1, 1, 2).unsqueeze(1).type(torch.float32)
            pred = model(scan)
                        
            optimizer.zero_grad()
            loss = criterion(pred, y)
            loss.backward()
            
            print('epoch {}/{} iter {}/{} train loss - {:.2f}'.format(epoch+1, start_epoch + num_epochs, i+1, len(train_loader), loss))
            optimizer.step()
            l_sum+=loss.item()
            scan.detach(); y.detach(); pred.detach(); loss.detach()
            del scan, y, pred, loss, batch
            torch.cuda.empty_cache()

        print('-'*60)
        curr_loss = l_sum/len(train_loader)
        train_report = val_step(train_loader, model, 'train')
        val_report = val_step(val_loader, model, 'val')
        metrics[epoch] = {
            'train': train_report,
            'val': val_report
        }
        if val_report['acc'] > prev_val_acc:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'start_epoch': epoch,
                'metrics': metrics,
            }, model_file)
            prev_val_acc = val_report['acc']
        else:
            ckpt = torch.load(model_file)
            ckpt['metrics'] = metrics; ckpt['start_epoch'] = epoch
            torch.save(ckpt, model_file)

        print('epoch stats: avg train loss- {:.2f} avg val acc: {:.2f}'.format(curr_loss, val_report['acc']))
        print('-'*60)

from seg_model import UNet3dLite
if __name__ == '__main__':
    
    folds = mosmed_dataloaders(15, 1, 0.2, kfold = True, only_mask_samples = True)
    for fold, (train_loader, val_loader) in folds.items():
        # if fold == 0:
        #     continue
        # model = CNN3d()
        model = NonLocalSE().to(device)
        model = Unet3d()
        # model = UNet3dLite()
        model = model.to(device)
        train_seg(train_loader, val_loader, fold)

    # train_loader, val_loader = mosmed_dataloaders(32, 1, 0.2)
    # train_shapes = [scan.shape for scan, _, _, _ in train_loader]
    # val_shapes = [scan.shape for scan, _, _, _ in val_loader]
    # print(train_shapes)
    # print(val_shapes)

    # create metrics
    