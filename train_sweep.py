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
import os
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

def get_optimizer(opt_name, model_params, lr, weight_decay):
    if opt_name.lower() == 'adam':
        return optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
    elif opt_name.lower() == 'sgd':
        return optim.SGD(model_params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer {opt_name} not supported")

def setup_cuda(gpu_ids):
    """Setup CUDA devices"""
    if gpu_ids is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        print(f"Using GPU(s): {gpu_ids}")
    
    if torch.cuda.is_available():
        print(f"Found {torch.cuda.device_count()} CUDA devices")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        return True
    else:
        print("No CUDA devices available, using CPU")
        return False

def train_model(config=None):
    with wandb.init(config=config) as run:
        config = wandb.config
        
        # Setup CUDA - you can specify GPU IDs in the sweep config
        cuda_available = setup_cuda(config.get('gpu_ids', None))
        device = torch.device("cuda" if cuda_available else "cpu")
        print(f"Using device: {device}")
        
        # Set random seed for reproducibility
        torch.manual_seed(config.seed)
        if cuda_available:
            torch.cuda.manual_seed(config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # DataLoader kwargs
        kwargs = {'num_workers': config.num_workers, 'pin_memory': cuda_available}
        
        # Set main directory of data
        directory_t2w = "Data"
        
        # Load the data
        print("\033[1;35;40m Loading the folders...\033[0m")
        
        train_loader = DataLoader(
            PMDataset(
                data_path=directory_t2w, 
                dataset_type='train',
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.RandomRotation(degrees=30)
                ])
            ), 
            batch_size=config.batch_size,
            shuffle=True, 
            **kwargs
        )

        test_loader = DataLoader(
            PMDataset(
                data_path=directory_t2w, 
                dataset_type='test',
                transform=transforms.Compose([
                    transforms.ToTensor()
                ])
            ), 
            batch_size=config.batch_size,
            shuffle=False, 
            **kwargs
        )
        
        # Load model
        print("\033[1;35;40m Loading the model...\033[0m")
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                             in_channels=3, out_channels=1, init_features=32, pretrained=True)
        model.conv = nn.Conv2d(32, 2, kernel_size=(1, 1), stride=(1, 1))
        
        if cuda_available:
            if torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs.")
                model = torch.nn.DataParallel(model)
            model = model.to(device)
        
        # Initialize optimizer, scheduler and loss function
        optimizer = get_optimizer(config.optimizer, model.parameters(), 
                                config.learning_rate, config.weight_decay)
        scheduler = sch.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
        criterion = monai.losses.DiceLoss(softmax=False, to_onehot_y=True, 
                                        include_background=False, reduction="mean")
        
        if cuda_available:
            criterion = criterion.to(device)
        
        # Initialize evaluator
        metrics = Evaluator()
        best_dice = 0
        
        # Training loop
        for epoch in range(1, config.epochs + 1):
            model.train()
            epoch_loss = []
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.float().to(device)
                target = target.long().to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss.append(loss.item())
                
                if batch_idx % config.log_interval == 0:
                    print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                          f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
                    
                    # Log GPU memory usage if available
                    if cuda_available:
                        for i in range(torch.cuda.device_count()):
                            memory_allocated = torch.cuda.memory_allocated(i) / 1024**2
                            memory_cached = torch.cuda.memory_reserved(i) / 1024**2
                            wandb.log({
                                f'gpu_{i}_memory_allocated_MB': memory_allocated,
                                f'gpu_{i}_memory_cached_MB': memory_cached
                            })
            
            mean_loss = np.mean(epoch_loss)
            print(f"Mean Training Loss: {mean_loss}")
            
            # Test phase
            model.eval()
            test_loss = 0
            dice_scores = []
            
            with torch.no_grad():
                for data, target in test_loader:
                    data = data.float().to(device)
                    target = target.long().to(device)
                    
                    output = model(data)
                    test_loss += criterion(output, target).item()
                    
                    pred = F.softmax(output.cpu(), dim=1).numpy()
                    pred = np.argmax(pred, axis=1)
                    target = target.cpu().numpy()
                    
                    for i in range(target.shape[0]):
                        dice = scoring.multiclass_dice_score(target[i], pred[i], 2)[0]
                        dice_scores.append(dice)
            
            mean_dice = np.mean(dice_scores)
            test_loss /= len(test_loader)
            
            # Log metrics to wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": mean_loss,
                "test_loss": test_loss,
                "dice_score": mean_dice,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
            
            if mean_dice > best_dice:
                best_dice = mean_dice
                # Save model with GPU ID info
                model_path = f"best_model_{run.id}_gpu{os.environ.get('CUDA_VISIBLE_DEVICES', 'cpu')}.pth"
                torch.save(model.state_dict(), model_path)
            
            scheduler.step()
            
        return best_dice

if __name__ == "__main__":
    sweep_config = {
        "method": "random",
        "metric": {
            "name": "dice_score",
            "goal": "maximize"
        },
        "parameters": {
            "batch_size": {
                "values": [4, 8, 16, 32]
            },
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 1e-2
            },
            "optimizer": {
                "values": ["adam", "sgd"]
            },
            "weight_decay": {
                "values": [1e-5, 1e-4, 1e-3]
            },
            "gamma": {
                "values": [0.1, 0.3, 0.5]
            },
            "step_size": {
                "values": [2, 3, 4]
            },
            "epochs": {
                "values": [20, 40, 60]
            },
            "seed": {
                "value": 22
            },
            "log_interval": {
                "value": 10
            },
            "num_workers": {
                "value": 4
            },
            "gpu_ids": {
                "value": "0,1,2,3"  # Specify which GPUs to use
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="paraspinal")
    wandb.agent(sweep_id, train_model, count=10)  # Run 10 experiments