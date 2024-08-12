import csv
from datetime import datetime
from glob import glob
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from glob import glob
from sklearn.model_selection import train_test_split
import random
from custom_data import CustomDataset, load_all_data
from models import EEGNet_8_2
import matplotlib.pyplot as plt


# Set the random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Gather all data from all directories
all_dirs = glob('/netscratch/shimizu/yanagisawa-ape/processed_data/*.npz')

# Shuffle the dirs
random.shuffle(all_dirs)

train_dirs, val_dirs = train_test_split(all_dirs, test_size=0.2, random_state=42)
val_dirs, test_dirs = train_test_split(val_dirs, test_size=0.5, random_state=42)
train_data = load_all_data(train_dirs)
val_data = load_all_data(val_dirs)
test_data = load_all_data(test_dirs)

# Create datasets and data loaders
train_dataset = CustomDataset(data=train_data)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

val_dataset = CustomDataset(data=val_data)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)

test_dataset = CustomDataset(data=test_data)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

os.makedirs('/netscratch/shimizu/yanagisawa-ape',exist_ok=True)
current_path = '/netscratch/shimizu/yanagisawa-ape/'

# Parameters
drops = [0.5, 0.25, 0.1, 0]
lrs = [0.001, 0.0005, 0.0001]
wds = [0, 0.001, 0.0005]
models = {
    'eegnet_8_2': EEGNet_8_2,
    # 'eegnet_4_2': EEGNet_4_2,
    # 'eegnex_8_32': EEGNeX_8_32,
    # 'singlelstm': SingleLSTM,
    # 'singlegru': SingleGRU,
    # 'onedcnn': OneDCNN,
    # 'onedcnncausal': OneDCNNCausal,
    # 'onedcnndilated': OneDCNNDilated,
    # 'onedcnncd': OneDCNNCausalDilated,
    # 'twodcnn': TwoDCNN,
    # 'twodcnndepth': TwoDCNNDepthwise,
    # 'twodcnndilated': TwoDCNNDilated,
    # 'twodcnnseparable': TwoDCNNSeparable,
    # 'cnnlstm': CNNLSTM,
    # 'cnngru': CNNGRU,
    # 'flashlightnet': FlashlightNet,
    # 'ienet': IENet,
    # 'mieegnet': MI_EEGNet,
    # 'seegnet': SEEGNet,
    # 'mibminet': MI_BMInet,
}

def train_and_validate(drop, lr, wd, model_class, modelname):
    save_dir = 'ape_0.5_0.5_5models'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(n_timesteps=500, n_features=32, n_outputs=2, DR=drop)
    model.to(device).double() 
    
    criterion = nn.BCEWithLogitsLoss().double()  # Use double precision
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    
    best_val_acc = 0.0
    patience_counter = 0
    epochs = 300
    patience = 50
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    test_accuracies = [] 
    
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device).double(), labels.to(device).double()  # Use double precision
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() / len(train_loader)
            acc = (outputs.argmax(dim=1) == labels.argmax(dim=1)).float().mean()
            epoch_train_acc += acc / len(train_loader)
        
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_acc = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device).double(), labels.to(device).double()  # Use double precision
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item() / len(val_loader)
                acc = (outputs.argmax(dim=1) == labels.argmax(dim=1)).float().mean()
                epoch_val_acc += acc / len(val_loader)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accuracies.append(epoch_train_acc.item())
        val_accuracies.append(epoch_val_acc.item())
        
        print(f'Epoch {epoch + 1}: Train Acc: {epoch_train_acc:.2f}, Train Loss: {epoch_train_loss:.2f}')
        print(f'Epoch {epoch + 1}: Val Acc: {epoch_val_acc:.2f}, Val Loss: {epoch_val_loss:.2f}')

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            patience_counter = 0
            print("Validation accuracy improved! Saving model...")
            os.makedirs(current_path + f'/{save_dir}/{modelname}/{drop}_{lr}_{wd}', exist_ok=True)
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc
            }, current_path + f'/{save_dir}/{modelname}/{drop}_{lr}_{wd}/model_checkpoint_{drop}_{lr}_{wd}.pth')
            epoch_test_acc = 0.0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device).double(), labels.to(device).double()
                    outputs = model(inputs)
                    acc = (outputs.argmax(dim=1) == labels.argmax(dim=1)).float().mean().item()
                    epoch_test_acc += acc
            epoch_test_acc /= len(test_loader)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping: No improvement for {patience} epochs.')
                break
    
    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)  
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.subplot(1, 3, 2)  
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(val_accuracies, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    
    plt.subplot(1, 3, 3) 
    plt.plot(test_accuracies, label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy over Epochs')
    plt.legend()

    os.makedirs(current_path + f'/{save_dir}_png/{modelname}/{drop}_{lr}_{wd}', exist_ok=True)
    plt.savefig(current_path + f'/{save_dir}_png/{modelname}/{drop}_{lr}_{wd}/training_plot_{drop}_{lr}_{wd}_{epoch_test_acc}.png')
    plt.show()
    return epoch_test_acc

# Main function to run the processes
import itertools

param_combinations = list(itertools.product(drops, lrs, wds))

results = {modelname: [] for modelname in models.keys()}

for modelname, model_class in models.items():
    for drop, lr, wd in param_combinations:
        acc = train_and_validate(drop, lr, wd, model_class, modelname)
        results[modelname].append((drop, lr, wd, acc))

output_dir = current_path + '/result'
output_path = os.path.join(output_dir, "results_test_splited.txt")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(output_path, "w") as f:
    for modelname, values in results.items():
        f.write(f"Model: {modelname}\n")
        for drop, lr, wd, acc in values:
            f.write(f"Drop: {drop}, LR: {lr}, WD: {wd}, Accuracy: {acc}\n")
        f.write("\n")