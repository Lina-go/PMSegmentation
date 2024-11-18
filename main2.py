import time
import copy
import torch
import monai
import wandb
import pickle
import argparse
import torchvision
import numpy as np
import torch.nn as nn
import os.path as osp
import torch.optim as optim
import utils.Graphics as gr
import matplotlib.pyplot as plt
import utils.scoring as scoring
import torch.nn.functional as F
import torch.optim.lr_scheduler as sch
import torch.utils.model_zoo as model_zoo
from config import model_config
from torch.autograd import Variable
from Dataloader import PMDataset
from utils.Evaluator import Evaluator
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from monai.networks.nets import UNETR

# Load the configuration
args = model_config()

# Check cuda availability
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Set random seed for GPU or CPU
if args.cuda:
    torch.cuda.manual_seed(args.seed)
else:
    torch.manual_seed(args.seed)

# Set arguments for Dataloaders
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# Set main directory of data
directory_t2w = "Data"

# Initialize WandB
wandb.login()
run_name = f"bs{args.batch_size}_lr{args.lr}"
wandb.init(project="paraspinal", name=run_name, config=args)
args = wandb.config

# Load the data
print("\033[1;35;40m Loading the folders...\033[0m")

train_loader = DataLoader(PMDataset(data_path=directory_t2w, dataset_type='train',
                transform=transforms.Compose([transforms.ToTensor(),
                transforms.RandomRotation(degrees=30)])), args.batch_size,
                shuffle=True, **kwargs)

test_loader = DataLoader(PMDataset(data_path=directory_t2w, dataset_type='test',
                transform=transforms.Compose([transforms.ToTensor()])), args.batch_size,
                shuffle=False, **kwargs)

# Load the model
print("\033[1;35;40m Loading the model...\033[0m")

model = UNETR(
    in_channels=3,
    out_channels=2,
    img_size=(512, 512),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed="conv",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
    spatial_dims=2
)

load_model = False

# Set number of GPUs to use
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs.")
        model = torch.nn.DataParallel(model)
    model.cuda()

# Set the optimizer, scheduler and loss function
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = sch.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
criterion = monai.losses.DiceLoss(softmax=True, to_onehot_y=True, include_background=False)

# Initialize the evaluator
metrics = Evaluator()

# Train the model
def train(epoch):
    wandb.watch(model, criterion, log="all", log_freq=1)
    model.train()
    loss_list = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.float()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target).long()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        loss_list.append(loss.item())
    print("Mean Training Loss: ", np.mean(loss_list))
    wandb.log({"train_loss": np.mean(loss_list), "epoch": epoch})

class_t = {
    0: "Background",
    1: "Paraspinal muscle"
}

def log_images(prefix, result_set):
    for i, (dsc, image, pred, target) in enumerate(result_set):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        ax1.imshow(image.squeeze(), cmap='gray')
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        ax2.imshow(image.squeeze(), cmap='gray')
        mask = pred > 0
        ax2.imshow(np.where(mask, pred, np.nan), cmap='plasma', alpha=0.5)
        ax2.set_title(f'Prediction (DSC: {dsc:.4f})')
        ax2.axis('off')
        
        ax3.imshow(image.squeeze(), cmap='gray')
        target_mask = target.squeeze() > 0
        ax3.imshow(np.where(target_mask, target.squeeze(), np.nan), cmap='inferno', alpha=0.5)
        ax3.set_title('Ground Truth')
        ax3.axis('off')
        
        wandb.log({f"{prefix}_{i+1}": wandb.Image(plt)})
        plt.close(fig)

# Test the model
def test(epoch, best_dice, model):
    dice_class = {1: []} 
    DSC = []
    pred_list = []
    model.eval()
    metrics.reset()
    test_loss = 0.
    mask_list = []
    results = []

    for data, target in test_loader:
        datis, targit = data, target

        data = data.float()
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target).long() 

        with torch.no_grad():
            output = model(data)

        test_loss += criterion(output, target).item()

        pred = output.cpu()
        pred = F.softmax(pred, dim=1).numpy()
        target = target.cpu().numpy()
        pred_list.append(pred)
        pred = np.argmax(pred, axis=1)

        if epoch in (args.epochs, args.epochs // 2):
            for i in range(target.shape[0]):
                pred_tensor = torch.tensor(pred[i], dtype=torch.float64)
                
                wandb.log(
                {"Prediction" : wandb.Image(data[i][0], masks={
                    "predictions" : {
                        "mask_data" : np.array(pred_tensor),
                        "class_labels" : class_t
                }
                    
                })
                })

                wandb.log(
                {"Ground Truth" : wandb.Image(data[i][0], masks={
                    
                    "ground_truth" : {
                        "mask_data" : np.array(targit[i].squeeze(0)),
                    "class_labels" : class_t
                }
                })})
                
        lista = []
        lista_1 = []
        for i in range(target.shape[0]):
            diccio_l = scoring.multiclass_dice_score(target[i], pred[i], 2)

            dsc = diccio_l[0]
            lista.append(dsc)

            if 1 in diccio_l.keys():
                lista_1.append(diccio_l[1])
            
            results.append((dsc, data[i][0].cpu().numpy(), pred[i], target[i]))
            
        DSC.extend(lista)
        dice_class[1].extend(lista_1)
  
    Dice = np.nanmean(DSC)
    channel_1 = np.nanmean(dice_class[1])
    
    wandb.log({"dice_score": Dice, "epoch": epoch})
    wandb.log({"test_loss": test_loss, "epoch": epoch})
    wandb.log({f"dice_score_{class_t[1]}": channel_1, "epoch": epoch})
    
    results.sort(key=lambda x: x[0])

    bottom_4 = results[:4]
    top_4 = results[-4:]
    mid_4 = results[len(results)//2-2:len(results)//2+2]

    log_images("bottom", bottom_4)
    log_images("top", top_4)
    log_images("mid", mid_4)

    if Dice > best_dice:
        best_dice = Dice
        torch.save(model.state_dict(), f"{args.model}_best.pth")

    with open(f"dice{args.model}.txt", "a") as f:
        f.write(f"Test\n")
        f.write(f"Epoch: {epoch}\n")
        f.write(f"Dice: {Dice}\n")
    print('Test:')
    print('[Epoch: %d, numImages: %5d]' % (epoch, len(test_loader.dataset)))
    print("Dice:{}".format(Dice))

    return test_loss, best_dice

# Run the model
if __name__ == '__main__':
    best_loss = None
    best_dice = 0
    if load_model:
        best_loss = test(0, best_dice, model)
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train(epoch)
            test_loss, dice = test(epoch, best_dice, model)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s '.format(
                epoch, time.time() - epoch_start_time))
            print('-' * 89)
            
            if best_loss is None or test_loss < best_loss:
                best_loss = test_loss
            if dice > best_dice:
                best_dice = dice
            
            scheduler.step() 

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')