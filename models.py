import torch
import torch.nn as nn
import torch.nn.functional as F

# FC Layers
def same_padding(kernel_size, stride, input_dim):
    if stride == 1:
        pad = (kernel_size - 1) // 2
        return pad, pad, pad, pad
    else:
        pad_total = max(kernel_size - stride, 0)
        pad_start = pad_total // 2
        pad_end = pad_total - pad_start
        return pad_start, pad_end, pad_start, pad_end

class SamePadAvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None):
        super(SamePadAvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, x):
        # Calculate padding for height and width
        pad_h = same_padding(self.kernel_size[0], self.stride, x.shape[-2])
        pad_w = same_padding(self.kernel_size[1], self.stride, x.shape[-1])
        
        # Apply padding
        x = F.pad(x, (pad_w[0], pad_w[1], pad_h[0], pad_h[1]))
        
        # Apply AvgPool2d
        x = F.avg_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        return x

# Basic Models
class SingleLSTM(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs, DR = 0.5):
        super(SingleLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=100, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=100, hidden_size=100, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=100, hidden_size=100, batch_first=True)
        self.dropout = nn.Dropout(DR)
        self.fc1 = nn.Linear(100, 100)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(100, n_outputs)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # x = x.permute(0,2,1)
        x, _ = self.lstm1(x)
        # print(x.shape, x.size())
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = self.dropout(x)
        x = self.fc1(x[:, -1, :])
        x = self.elu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

class SingleGRU(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs, DR = 0.5):
        super(SingleGRU, self).__init__()
        self.gru1 = nn.GRU(input_size=n_features, hidden_size=100, batch_first=True)
        self.gru2 = nn.GRU(input_size=100, hidden_size=100, batch_first=True)
        self.gru3 = nn.GRU(input_size=100, hidden_size=100, batch_first=True)
        self.dropout = nn.Dropout(DR)
        self.fc1 = nn.Linear(100, 100)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(100, n_outputs)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # x = x.permute(0,2,1)
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x, _ = self.gru3(x)
        x = self.dropout(x)
        x = self.fc1(x[:, -1, :])
        x = self.elu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

class OneDCNN(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs, DR = 0.5):
        super(OneDCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.elu1 = nn.ELU()
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.elu2 = nn.ELU()
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(64)
        self.elu3 = nn.ELU()
        self.dropout = nn.Dropout(DR)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1, 100)
        self.elu4 = nn.ELU()
        self.fc2 = nn.Linear(100, n_outputs)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        device = x.device
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.elu3(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = self.flatten(x).float()
        # Capture shape dynamically
        if not hasattr(self, '_flattened_size'):
            self._flattened_size = x.shape[1]
            self.fc1 = nn.Linear(self._flattened_size, self.fc1.out_features).to(device)
        x = self.fc1(x).double()
        x = self.elu4(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    
class OneDCNNCausalDilated(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs, DR=0.5):
        super(OneDCNNCausalDilated, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=3, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.elu1 = nn.ELU()
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=4, dilation=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.elu2 = nn.ELU()
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=8, dilation=4)
        self.bn3 = nn.BatchNorm1d(64)
        self.elu3 = nn.ELU()
        self.dropout = nn.Dropout(DR)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * (n_timesteps // 2), 100)
        self.elu4 = nn.ELU()
        self.fc2 = nn.Linear(100, n_outputs)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        device = x.device
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.elu3(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = self.flatten(x).float()
        if not hasattr(self, '_flattened_size'):
            self._flattened_size = x.shape[1]
            self.fc1 = nn.Linear(self._flattened_size, self.fc1.out_features).to(device)
        x = self.fc1(x).double()
        x = self.elu4(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

class OneDCNNCausal(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs, DR=0.5):
        super(OneDCNNCausal, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=3, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.elu1 = nn.ELU()
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.elu2 = nn.ELU()
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        self.elu3 = nn.ELU()
        self.dropout = nn.Dropout(DR)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * (n_timesteps // 2), 100)
        self.elu4 = nn.ELU()
        self.fc2 = nn.Linear(100, n_outputs)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        device = x.device
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.elu3(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = self.flatten(x).float()
        if not hasattr(self, '_flattened_size'):
            self._flattened_size = x.shape[1]
            self.fc1 = nn.Linear(self._flattened_size, self.fc1.out_features).to(device)
        x = self.fc1(x).double()
        x = self.elu4(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

class OneDCNNDilated(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs, DR=0.5):
        super(OneDCNNDilated, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.elu1 = nn.ELU()
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, dilation=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.elu2 = nn.ELU()
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, dilation=4)
        self.bn3 = nn.BatchNorm1d(64)
        self.elu3 = nn.ELU()
        self.dropout = nn.Dropout(DR)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * ((n_timesteps - 8) // 2), 100)
        self.elu4 = nn.ELU()
        self.fc2 = nn.Linear(100, n_outputs)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        device = x.device
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.elu3(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = self.flatten(x).float()
        if not hasattr(self, '_flattened_size'):
            self._flattened_size = x.shape[1]
            self.fc1 = nn.Linear(self._flattened_size, self.fc1.out_features).to(device)
        x = self.fc1(x).double()
        x = self.elu4(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

class TwoDCNN(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs, DR = 0.5):
        super(TwoDCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 3))
        self.bn1 = nn.BatchNorm2d(64)
        self.elu1 = nn.ELU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3))
        self.bn2 = nn.BatchNorm2d(64)
        self.elu2 = nn.ELU()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3))
        self.dropout = nn.Dropout(DR)
        self.pool = nn.AvgPool2d(kernel_size=(2, 1), padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1, 100)
        self.elu3 = nn.ELU()
        self.fc2 = nn.Linear(100, n_outputs)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = x.unsqueeze(0).permute(1, 0, 2, 3)
        device = x.device
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu2(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = self.flatten(x).float()
        # Capture shape dynamically
        if not hasattr(self, '_flattened_size'):
            self._flattened_size = x.shape[1]
            self.fc1 = nn.Linear(self._flattened_size, self.fc1.out_features).to(device)
        x = self.fc1(x).double()
        x = self.elu3(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    
class TwoDCNNDilated(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs, DR=0.5):
        super(TwoDCNNDilated, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.elu1 = nn.ELU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), dilation=2, padding=(0, 2))
        self.bn2 = nn.BatchNorm2d(64)
        self.elu2 = nn.ELU()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), dilation=4, padding=(0, 4))
        self.bn3 = nn.BatchNorm2d(64)
        self.elu3 = nn.ELU()
        self.dropout = nn.Dropout(DR)
        self.pool = nn.AvgPool2d(kernel_size=(2, 1), padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * ((n_timesteps - 1) // 2), 100)
        self.elu4 = nn.ELU()
        self.fc2 = nn.Linear(100, n_outputs)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = x.unsqueeze(1)  # Adding channel dimension
        device = x.device
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.elu3(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = self.flatten(x).float()
        if not hasattr(self, '_flattened_size'):
            self._flattened_size = x.shape[1]
            self.fc1 = nn.Linear(self._flattened_size, self.fc1.out_features).to(x.device)
        x = self.fc1(x).double()
        x = self.elu4(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

class TwoDCNNSeparable(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs, DR=0.5):
        super(TwoDCNNSeparable, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.elu1 = nn.ELU()
        
        # Separable Convolution: Depthwise followed by Pointwise
        self.depthwise_conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 16), padding=(0, 8), groups=16, bias=False)
        self.pointwise_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.elu2 = nn.ELU()
        
        self.depthwise_conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 16), padding=(0, 8), groups=32, bias=False)
        self.pointwise_conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.elu3 = nn.ELU()
        
        self.dropout = nn.Dropout(DR)
        self.pool = nn.AvgPool2d(kernel_size=(2, 1), padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * (n_timesteps // 2), 100)  # Adjusted for flattened size
        self.elu4 = nn.ELU()
        self.fc2 = nn.Linear(100, n_outputs)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = x.unsqueeze(1)  # Adding channel dimension
        device = x.device
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu1(x)
        
        x = self.depthwise_conv2(x)
        x = self.pointwise_conv2(x)
        x = self.bn2(x)
        x = self.elu2(x)
        
        x = self.depthwise_conv3(x)
        x = self.pointwise_conv3(x)
        x = self.bn3(x)
        x = self.elu3(x)
        
        x = self.dropout(x)
        x = self.pool(x)
        x = self.flatten(x).float()
        # Capture shape dynamically
        if not hasattr(self, '_flattened_size'):
            self._flattened_size = x.shape[1]
            self.fc1 = nn.Linear(self._flattened_size, self.fc1.out_features).to(device)
        x = self.fc1(x).double()
        x = self.elu4(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    
class TwoDCNNDepthwise(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs, DR=0.5):
        super(TwoDCNNDepthwise, self).__init__()
        self.depthwise_conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(22, 1), groups=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.elu1 = nn.ELU()
        
        # Depthwise Convolution: Each channel separately
        self.depthwise_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), groups=64, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.elu2 = nn.ELU()
        
        self.depthwise_conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), groups=64, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.elu3 = nn.ELU()
        
        self.dropout = nn.Dropout(DR)
        self.pool = nn.AvgPool2d(kernel_size=(2, 1), padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * (n_timesteps // 2), 100)  # Adjusted for flattened size
        self.elu4 = nn.ELU()
        self.fc2 = nn.Linear(100, n_outputs)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = x.unsqueeze(1)  # Adding channel dimension
        device = x.device
        x = self.depthwise_conv1(x)
        x = self.bn1(x)
        x = self.elu1(x)
        
        x = self.depthwise_conv2(x)
        x = self.bn2(x)
        x = self.elu2(x)
        
        x = self.depthwise_conv3(x)
        x = self.bn3(x)
        x = self.elu3(x)
        
        x = self.dropout(x)
        x = self.pool(x)
        x = self.flatten(x).float()
        # Capture shape dynamically
        if not hasattr(self, '_flattened_size'):
            self._flattened_size = x.shape[1]
            self.fc1 = nn.Linear(self._flattened_size, self.fc1.out_features).to(device)
        x = self.fc1(x).double()
        x = self.elu4(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

class CNNLSTM(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs, DR = 0.5):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.elu1 = nn.ELU()
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.elu2 = nn.ELU()
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(64)
        self.elu3 = nn.ELU()
        self.dropout1 = nn.Dropout(DR)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTM(input_size=1, hidden_size=480, batch_first=True)
        self.dropout2 = nn.Dropout(DR)
        self.fc1 = nn.Linear(480, 100)
        self.elu4 = nn.ELU()
        self.fc2 = nn.Linear(100, n_outputs)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        device = x.device
        # batch_size, timesteps, channels = x.size()
        c_in = x
        c_in = self.conv1(c_in)
        c_in = self.bn1(c_in)
        c_in = self.elu1(c_in)
        c_in = self.conv2(c_in)
        c_in = self.bn2(c_in)
        c_in = self.elu2(c_in)
        c_in = self.conv3(c_in)
        c_in = self.bn3(c_in)
        c_in = self.elu3(c_in)
        c_in = self.dropout1(c_in)
        c_in = self.pool(c_in)
        c_in = c_in.permute(0,2,1).float()
        if not hasattr(self, '_flattened_size'):
            self._flattened_size = c_in.shape[2]
            self.lstm = nn.LSTM(input_size=self._flattened_size, hidden_size=self.lstm.hidden_size, batch_first=self.lstm.batch_first).to(device)
        c_in, _ = self.lstm(c_in)
        c_in = c_in.double()
        c_in = self.dropout2(c_in[:, -1, :])
        c_in = self.fc1(c_in)
        c_in = self.elu4(c_in)
        c_in = self.fc2(c_in)
        c_in = self.softmax(c_in)
        return c_in

class CNNGRU(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs, DR = 0.5):
        super(CNNGRU, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.elu1 = nn.ELU()
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.elu2 = nn.ELU()
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(64)
        self.elu3 = nn.ELU()
        self.dropout1 = nn.Dropout(DR)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.gru = nn.GRU(input_size=1, hidden_size=480, batch_first=True)
        self.dropout2 = nn.Dropout(DR)
        self.fc1 = nn.Linear(480, 100)
        self.elu4 = nn.ELU()
        self.fc2 = nn.Linear(100, n_outputs)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        device = x.device
        # batch_size, timesteps, channels = x.size()
        c_in = x
        c_in = self.conv1(c_in)
        c_in = self.bn1(c_in)
        c_in = self.elu1(c_in)
        c_in = self.conv2(c_in)
        c_in = self.bn2(c_in)
        c_in = self.elu2(c_in)
        c_in = self.conv3(c_in)
        c_in = self.bn3(c_in)
        c_in = self.elu3(c_in)
        c_in = self.dropout1(c_in)
        c_in = self.pool(c_in)
        c_in = c_in.permute(0,2,1).float()
        if not hasattr(self, '_flattened_size'):
            self._flattened_size = c_in.shape[2]
            self.gru = nn.GRU(input_size=self._flattened_size, hidden_size=self.gru.hidden_size, batch_first=self.gru.batch_first).to(device)
        c_in, _ = self.gru(c_in)
        c_in = c_in.double()
        c_in = self.dropout2(c_in[:, -1, :])
        c_in = self.fc1(c_in)
        c_in = self.elu4(c_in)
        c_in = self.fc2(c_in)
        c_in = self.softmax(c_in)
        return c_in

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class SingleConvLSTM2D(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs, DR=0.5):
        super(SingleConvLSTM2D, self).__init__()
        self.convlstm = ConvLSTM(input_dim=1, hidden_dim=64, kernel_size=(1, 3), num_layers=1, batch_first=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.elu1 = nn.ELU()
        self.dropout = nn.Dropout(DR)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * n_features * n_timesteps, 100)
        self.elu2 = nn.ELU()
        self.fc2 = nn.Linear(100, n_outputs)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, _ = self.convlstm(x)
        x = x[0][:, -1, :, :, :]  # Get the last output of the LSTM
        x = self.bn1(x)
        x = self.elu1(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.elu2(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    
class EEGNet(nn.Module):
    def __init__(self, F1, D, F2, n_timesteps, n_features, n_outputs, DR = 0.5):
        super(EEGNet, self).__init__()
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise_conv = nn.Conv2d(F1, F1 * D, (n_features, 1), groups=F1, padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu1 = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, 4)) #SamePadAvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(DR)
        self.separable_conv = nn.Conv2d(F1 * D, F2, (1, 16), padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.avg_pool2 = nn.AvgPool2d((1, 8)) #SamePadAvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(DR)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1, n_outputs) 
        # self.fc = nn.Linear(F2 * (n_timesteps // 32) * n_features, n_outputs)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        device = x.device
        x = x.unsqueeze(0).permute(1, 0, 2, 3)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.avg_pool1(x)
        x = self.dropout1(x)
        x = self.separable_conv(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.avg_pool2(x)
        x = self.dropout2(x)
        x = self.flatten(x).float()
        # Capture shape dynamically
        if not hasattr(self, '_flattened_size'):
            self._flattened_size = x.shape[1]
            self.fc = nn.Linear(self._flattened_size, self.fc.out_features).to(device)
        x = self.fc(x)
        #print(x.shape)
        x = self.softmax(x)
        return x

class EEGNet_8_2(EEGNet):
    def __init__(self, n_timesteps, n_features, n_outputs, DR = 0.5):
        super(EEGNet_8_2, self).__init__(8, 2, 16, n_timesteps, n_features, n_outputs, DR = 0.5)

class EEGNet_4_2(EEGNet):
    def __init__(self, n_timesteps, n_features, n_outputs, DR = 0.5):
        super(EEGNet_4_2, self).__init__(4, 2, 8, n_timesteps, n_features, n_outputs, DR = 0.5)

class EEGNeX_8_32(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs, DR = 0.5):
        super(EEGNeX_8_32, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, (1, 64), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.elu1 = nn.ELU()
        self.conv2 = nn.Conv2d(8, 32, (1, 64), padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.depthwise_conv = nn.Conv2d(32, 32 * 2, (n_features, 1), groups=32, padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(32 * 2)
        self.elu2 = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, 4)) #SamePadAvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(DR)
        self.conv3 = nn.Conv2d(32 * 2, 32, (1, 16), padding='same', dilation=(1, 2), bias=False)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 8, (1, 16), padding='same', dilation=(1, 4), bias=False)
        self.bn5 = nn.BatchNorm2d(8)
        self.elu3 = nn.ELU()
        self.avg_pool2 = nn.AvgPool2d((1, 4)) #SamePadAvgPool2d((1, 4))
        self.dropout2 = nn.Dropout(DR)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1, n_outputs) 
        # self.fc = nn.Linear(8 * (n_timesteps // 32) * (n_features), n_outputs)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        device = x.device
        x = x.unsqueeze(0).permute(1, 0, 2, 3)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.depthwise_conv(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.avg_pool1(x)
        x = self.dropout1(x)
        x = self.conv3(x)
        x = self.bn4(x)
        x = self.conv4(x)
        x = self.bn5(x)
        x = self.elu3(x)
        x = self.avg_pool2(x)
        x = self.dropout2(x)
        x = self.flatten(x).float()
        # Capture shape dynamically
        if not hasattr(self, '_flattened_size'):
            self._flattened_size = x.shape[1]
            self.fc = nn.Linear(self._flattened_size, self.fc.out_features).to(device)
        x = self.fc(x)
        x = self.softmax(x)
        return x

class FlashlightNet(nn.Module):
    def __init__(self,  n_timesteps, n_features, n_outputs=4, DR=0.5):
        super(FlashlightNet, self).__init__()

        # Smoothing module
        self.smooth_conv = nn.Conv2d(1, 10, (1, 25), padding='same')
        self.smooth_bn = nn.BatchNorm2d(10)

        # Channel fusion module
        self.cf_conv = nn.Conv2d(10, 80, (22, 1), groups=10)
        self.cf_bn = nn.BatchNorm2d(80)
        
        # Time domain module
        self.td_conv1 = nn.Conv2d(80, 80, (1, 125), padding='same', dilation=1)
        self.td_conv2 = nn.Conv2d(80, 80, (1, 125), padding='same', dilation=2)
        self.td_conv3 = nn.Conv2d(80, 80, (1, 125), padding='same', dilation=3)
        self.pool = nn.AvgPool2d((1, 5))
        self.dropout = nn.Dropout(DR)

        # Feature pool module
        self.fp_conv1 = nn.Conv2d(240, 80, (1, 10), padding='same', dilation=1)
        self.fp_conv2 = nn.Conv2d(80, 80, (1, 10), padding='same', dilation=3)
        self.fp_conv3 = nn.Conv2d(240, 80, (1, 20), padding='same', dilation=1)
        self.fp_conv4 = nn.Conv2d(80, 80, (1, 20), padding='same', dilation=3)
        self.fp_conv5 = nn.Conv2d(240, 80, (1, 30), padding='same', dilation=1)
        self.fp_conv6 = nn.Conv2d(80, 80, (1, 30), padding='same', dilation=3)
        self.fp_bn = nn.BatchNorm2d(480)

        # Classification module
        self.fc_main = nn.Linear(1, n_outputs)  # assuming 1x90x720 after pooling and dropout
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        device = x.device
        x = x.unsqueeze(0).permute(1, 0, 2, 3)
        # x = x.unsqueeze(1)  # Add a channel dimension
        
        # Smoothing module
        x = self.smooth_conv(x)
        x = self.smooth_bn(x)
        x = torch.relu(x)
        
        # Channel fusion module
        x = self.cf_conv(x)
        x = self.cf_bn(x)
        x = torch.relu(x)
        
        # Time domain module
        x1 = self.td_conv1(x)
        x2 = self.td_conv2(x)
        x3 = self.td_conv3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.pool(x)
        x = self.dropout(x)

        # Feature pool module
        x1 = self.fp_conv1(x)
        x2 = self.fp_conv2(x1)
        x3 = self.fp_conv3(x)
        x4 = self.fp_conv4(x3)
        x5 = self.fp_conv5(x)
        x6 = self.fp_conv6(x5)
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        x = self.fp_bn(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        # Flatten for fully connected layer
        x = x.view(x.size(0), -1)
        x = x.float()
        if not hasattr(self, '_flattened_size'):
            self._flattened_size = x.shape[1]
            self.fc_main = nn.Linear(self._flattened_size, self.fc_main.out_features).to(device)
        x = self.fc_main(x).double()
        x = self.softmax(x)
        
        return x

class MSC_Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSC_Layer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding='same')
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=10, padding='same')
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=20, padding='same')

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        return torch.cat((x1, x2, x3), dim=1)

class IENet(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs, DR=0.5):
        super(IENet, self).__init__()
        self.msc1 = MSC_Layer(n_features, 40)
        self.msc2 = MSC_Layer(120, 40)
        self.msc3 = MSC_Layer(120, 40)
        self.msc4 = MSC_Layer(120, 40)
        self.msc5 = MSC_Layer(120, 40)
        self.msc6 = MSC_Layer(120, 40)
        
        self.batch_norm = nn.BatchNorm1d(120)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(120, n_outputs)
        self.dropout = nn.Dropout(DR)

    def forward(self, x):
        device = x.device
        x = x.permute(0,2,1)  # Swap the dimensions to [batch, channels, timesteps]
        
        x = self.msc1(x)
        x = self.msc2(x)
        x = self.msc3(x)
        x = self.msc4(x)
        x = self.msc5(x)
        x = self.msc6(x)
        
        x = self.batch_norm(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return F.softmax(x, dim=1)
    
class MI_EEGNet(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs=4, DR=0.5):
        super(MI_EEGNet, self).__init__()

        # Block 1: Temporal Convolution
        self.temporal_conv = nn.Conv2d(1, 16, (1, 16), padding=0)
        self.temporal_bn = nn.BatchNorm2d(16)
        
        # Block 1: Spatial Depthwise Convolution
        self.spatial_conv = nn.Conv2d(16, 16 * n_features, (n_features, 1), groups=16)
        self.spatial_bn = nn.BatchNorm2d(16 * n_features)
        
        # Inception-like block
        self.inception_conv1 = nn.Conv2d(16 * n_features, 32, (1, 3), padding=0)
        self.inception_conv2 = nn.Conv2d(16 * n_features, 32, (1, 5), padding=(0,1))
        self.inception_conv3 = nn.Conv2d(16 * n_features, 32, (1, 7), padding=(0,2))
        
        # Depthwise Separable Convolution Block
        self.sep_conv = nn.Conv2d(96, 96, (1, 3), groups=96, padding=0)
        self.pointwise_conv = nn.Conv2d(96, 96, (1, 1))
        self.sep_bn = nn.BatchNorm2d(96)
        
        # Global Average Pooling and Fully Connected Layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(DR)
        self.fc = nn.Linear(96, n_outputs)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        device = x.device
        x = x.unsqueeze(0).permute(1,0,2,3)  # Add a channel dimension
        
        # Block 1
        x = self.temporal_conv(x)
        x = self.temporal_bn(x)
        x = F.elu(x)
        
        x = self.spatial_conv(x)
        x = self.spatial_bn(x)
        x = F.elu(x)
        
        # Inception-like block
        x1 = self.inception_conv1(x)
        x2 = self.inception_conv2(x)
        x3 = self.inception_conv3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        
        # Depthwise Separable Convolution Block
        x = self.sep_conv(x)
        x = self.pointwise_conv(x)
        x = self.sep_bn(x)
        x = F.elu(x)
        
        # Global Average Pooling and Fully Connected Layer
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.softmax(x)
        
        return x

class MI_BMInet(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs, DR=0.5):
        super(MI_BMInet, self).__init__()

        # Spatial Convolution
        self.spatial_conv = nn.Conv2d(1, 32, (n_features, 1), padding=(0, 0), bias=False)
        self.spatial_bn = nn.BatchNorm2d(32)

        # Temporal Convolution
        self.temporal_conv = nn.Conv2d(32, 32, (1, 10), padding=(0, 0), bias=False)
        self.temporal_bn = nn.BatchNorm2d(32)

        # Separable Convolution Block
        self.depthwise_conv = nn.Conv2d(32, 32, (1, 16), groups=32, padding=(0, 0), bias=False)
        self.pointwise_conv = nn.Conv2d(32, 32, (1, 1), padding=(0, 0), bias=False)
        self.sep_bn = nn.BatchNorm2d(32)

        # Fully Connected Layer
        reduced_timesteps = n_timesteps // 8 // 8
        self.fc = nn.Linear(1,n_outputs)
        self.dropout = nn.Dropout(DR)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Add a channel dimension
        x = x.unsqueeze(1)
        device = x.device
        # Spatial Convolution Block
        x = self.spatial_conv(x)
        x = self.spatial_bn(x)
        x = F.relu(x)

        # Temporal Convolution Block
        x = self.temporal_conv(x)
        x = self.temporal_bn(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, (1, 4), stride=1)

        # Separable Convolution Block
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.sep_bn(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, (1, 4))

        # Global Average Pooling
        x = torch.flatten(x, start_dim=1).float()
        x = self.dropout(x)
        # Capture shape dynamically
        if not hasattr(self, '_flattened_size'):
            self._flattened_size = x.shape[1]
            self.fc = nn.Linear(self._flattened_size, self.fc.out_features).to(device)
        x = self.fc(x).double()
        x = self.softmax(x)

        return x

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SEEGNet(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs=4, DR=0.5):
        super(SEEGNet, self).__init__()

        # First convolution block
        self.conv1 = nn.Conv2d(1, 16, (1, 16), padding=0)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Depthwise separable convolution blocks
        self.depthwise_separable_conv1 = DepthwiseSeparableConv2d(16, 32, (n_features, 1))
        self.bn2 = nn.BatchNorm2d(32)
        
        self.depthwise_separable_conv2 = DepthwiseSeparableConv2d(32, 64, (1, 1))
        self.bn3 = nn.BatchNorm2d(64)

        # Additional layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(DR)
        self.fc = nn.Linear(64, n_outputs)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        device = x.device
        x = x.unsqueeze(0).permute(1, 0, 2, 3)
        # print(x.shape)
        # x = x.unsqueeze(1)  # Add a channel dimension
        
        # First convolution block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.elu(x)
        
        # Depthwise separable convolution blocks
        x = self.depthwise_separable_conv1(x)
        x = self.bn2(x)
        x = F.elu(x)
        
        x = self.depthwise_separable_conv2(x)
        x = self.bn3(x)
        x = F.elu(x)
        
        # Global average pooling and fully connected layer
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.softmax(x)
        
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, in_channels, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([nn.Conv2d(in_channels, in_channels, 1) for _ in range(num_heads)])
        # print(self.attention_heads)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, P, T = x.size()
        print('size',x.size())
        head_outputs = []
        for head in self.attention_heads:
            print('test')
            print(x.size(),batch_size, P * T, C // self.num_heads)
            query = head(x).permute(0, 2, 3, 1).reshape(batch_size, P * T, C // self.num_heads)
            key = head(x).permute(0, 2, 3, 1).reshape(batch_size, P * T, C // self.num_heads)
            value = head(x).permute(0, 2, 3, 1).reshape(batch_size, P * T, C // self.num_heads)
            attention_scores = self.softmax(torch.bmm(query, key.transpose(1, 2)) / (C // self.num_heads) ** 0.5)
            head_output = torch.bmm(attention_scores, value).reshape(batch_size, P, T, C // self.num_heads)
            head_outputs.append(head_output.permute(0, 3, 1, 2))
        return torch.cat(head_outputs, dim=1)

class ConvolutionalFeatureExpansion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionalFeatureExpansion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, (1, 3), padding=(0, 1))
        self.conv2 = nn.Conv2d(in_channels, out_channels // 2, (1, 5), padding=(0, 2))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = F.elu(self.conv1(x))
        x2 = F.elu(self.conv2(x))
        x = torch.cat((x1, x2), dim=1)
        return self.bn(x)

class EEGConvTransformer(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs=4, dropout_rate=0.5):
        super(EEGConvTransformer, self).__init__()

        # Local Feature Extractor (LFE) module
        self.lfe_conv1 = nn.Conv2d(1, 16, (8, 8), stride=(4, 4), padding=(2, 2))
        self.lfe_conv2 = nn.Conv2d(1, 16, (8, 8), stride=(4, 4), padding=(2, 2))
        self.lfe_bn = nn.BatchNorm2d(16)
        self.lfe_elu = nn.ELU()

        # ConvTransformer module
        self.mha = MultiHeadAttention(32, num_heads=4)
        self.cfe = ConvolutionalFeatureExpansion(16, 32)

        # Convolutional Encoder
        self.conv_encoder = nn.Conv2d(32, 64, (3, 3), padding=(1, 1))
        self.bn_encoder = nn.BatchNorm2d(64)
        self.elu_encoder = nn.ELU()

        # Fully Connected Classifier
        self.fc1 = nn.Linear(64 * (n_timesteps // 4) * (n_features // 4), 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, n_outputs)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(0).permute(1,0,2,3)  # Add a channel dimension
        print(x.shape)

        # Local Feature Extractor (LFE) module
        x1 = self.lfe_elu(self.lfe_bn(self.lfe_conv1(x)))
        x2 = self.lfe_elu(self.lfe_bn(self.lfe_conv2(x)))
        x = torch.cat((x1, x2), dim=1)  # Concatenate along the channel dimension

        # ConvTransformer module
        x = self.mha(x)
        x = self.cfe(x)

        # Convolutional Encoder
        x = self.elu_encoder(self.bn_encoder(self.conv_encoder(x)))

        # Fully Connected Classifier
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return self.softmax(x)