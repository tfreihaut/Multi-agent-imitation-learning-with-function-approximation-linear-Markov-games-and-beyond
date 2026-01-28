import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import time

from torch.utils.data import Dataset, DataLoader

class BehavioralDataset(Dataset):
    def __init__(self, expert_trajectories):
        # expert_trajectories is a list of (state_tensor, action_tensor)
        self.states = [s for s, _ in expert_trajectories]
        self.actions = [torch.as_tensor(a, dtype=torch.long) for _, a in expert_trajectories]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        x = self.states[idx]
        y = self.actions[idx]
        
        # Convert to float32 if not already (ensure CNN compatibility)
        if isinstance(x, torch.Tensor):
            x = x.float()
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        # ensure correct shapes - always convert to CHW format
        if x.ndim == 3:
            # Check if it's HWC by looking at dimensions
            # Common case: H and W are larger than C (e.g., 3x3x2 or 6x7x2)
            # If first dimension is smallest and last dimension is also small (<=3), likely HWC
            if x.shape[0] > x.shape[2] and x.shape[2] <= 3:
                # Detected HWC (e.g., 3x3x2) -> convert to CHW
                x = x.permute(2, 0, 1)
            # If already CHW (first dim is small channel count), keep as is
            # e.g., 2x3x3 is already CHW
        elif x.ndim == 2:  # (H,W) - add channel dimension
            x = x.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected state shape: {x.shape}")
        
        return x, y
    

class SmallCNN(nn.Module):
    """CNN for small games, outputs logits for 7 actions."""

    def __init__(self, in_channels=2, num_actions=7): 
            super().__init__()
            self.conv = nn.Sequential(
                # Layer 1: Increase channels immediately
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                
                # Layer 2: Deeper features
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                
                # Layer 3: Maintain high feature count
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            )
            
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.LazyLinear(256), # Increased from 128
                nn.BatchNorm1d(256), # Batch norm for dense layer
                nn.ReLU(),
                nn.Dropout(0.3),    # Slightly higher dropout
                nn.Linear(256, num_actions),
            )

    def forward(self, x):
        x = self.conv(x)
        return self.classifier(x)


    def get_action_probs(self, state):
        """
        Calculates the action probability distribution for a given state index.

        Args:
            state_index (int or torch.Tensor): The index of the state.

        Returns:
            torch.Tensor: A 1D tensor representing the probability distribution over actions.
        """
        # Set the model to evaluation mode (important for layers like dropout, batchnorm, etc.)
        self.eval()

        # Disable gradient calculations for inference
        with torch.no_grad():
            
            # Get the logits from the forward pass
            logits = self.forward(state)

            # Apply softmax to convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Remove the batch dimension before returning
            return probs.squeeze(0)

class SoftLinPolicy(nn.Module):
    """
    Implements the softmax-linear policy class using an explicit feature map φ(x, a).
    π(a|x) ∝ exp(η * φ(x,a)^T * θ)
    """
    def __init__(self, feature_map, feature_dim, num_actions, eta=1.0):
        super(SoftLinPolicy, self).__init__()
        self.feature_map = feature_map
        self.feature_dim = feature_dim
        self.num_actions = num_actions
        self.eta = eta
        self.theta = nn.Linear(feature_dim, 1, bias=False)

    def forward(self, states):
        batch_size = states.shape[0]
        all_actions = torch.arange(self.num_actions, device=states.device)
        states_expanded = states.unsqueeze(1).expand(-1, self.num_actions)
        actions_expanded = all_actions.unsqueeze(0).expand(batch_size, -1)
        
        feature_vectors = self.feature_map(states_expanded, actions_expanded)
        feature_vectors_flat = feature_vectors.view(-1, self.feature_dim)
        scores_flat = self.theta(feature_vectors_flat)
        scores = scores_flat.view(batch_size, self.num_actions)
        logits = self.eta * scores
        return logits
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Allows the network to use global context (entire board) to weight channels.
    """
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # Optional: Squeeze-and-Excitation
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.use_se:
            out = self.se(out)
            
        out += residual
        out = F.relu(out)
        return out

class Connect4ResNet(nn.Module):
    def __init__(self, in_channels=2, num_actions=7, num_res_blocks=5, num_filters=128):
        super().__init__()
        
        # 1. Start Block
        self.start_block = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )
        
        # 2. Backbone (ResNet)
        self.backbone = nn.Sequential(
            *[ResidualBlock(num_filters, use_se=True) for _ in range(num_res_blocks)]
        )
        
        # 3. Policy Head (Optimized)
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 32, kernel_size=1), # Compress channels
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            # OPTIMIZATION: Removed Dropout, Increased Width (256 -> 512)
            nn.Linear(32 * 6 * 7, 512), 
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.start_block(x)
        x = self.backbone(x)
        return self.policy_head(x)
    
class DeepSoftmaxPolicy(nn.Module):
    """
    Implements feedforward neural network policy with softmax output.
    π(a|x) = Softmax( NN(x)[a] )
    """
    def __init__(self, state_dim, num_actions, hidden_layers=[128, 128], eta=1.0):
        super(DeepSoftmaxPolicy, self).__init__()
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, num_actions))
        self.network = nn.Sequential(*layers)
        self.eta = eta

    def forward(self, states):
        logits = self.network(states)
        logits = self.eta * logits
        return logits

class BehavioralCloningSingleAgent:
    def __init__(self, num_actions, feature_map=None, 
                 feature_map_args=None, lr=1e-3, eta=1.0, cnn_policy=False, in_channels=1, device='cpu'):
        
        self.cnn = cnn_policy
        self.device = device
        self.num_actions = num_actions
        
        # Initialize ResNet
        if self.cnn:
            self.policy_net = Connect4ResNet(in_channels=in_channels, num_actions=num_actions)
            self.policy_net.to(self.device)
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            # Placeholder for tabular/linear policy init if needed
            self.policy_net = None 

    def train(self, dataset, epochs=100, batch_size=4096, device='cuda'):
        if not dataset:
            return None, 0.0
        
        if self.cnn:
            self.policy_net.to(device)
            self.policy_net.train()
            
            # Use num_workers=2 as requested; ensure your RAM can handle the pre-fetching
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
            
            final_loss = 0.0
            
            # Outer loop for Epochs
            for epoch in range(epochs):
                total_epoch_loss = 0.0
                num_batches = 0
                
                # Inner loop for Batches with tqdm
                # unit="batch" tells us the speed in batches/sec
                pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch", leave=True)
                
                epoch_start_time = time.time()
                
                for batch_states, batch_actions in pbar:
                    batch_states = batch_states.to(device, non_blocking=True)
                    batch_actions = batch_actions.to(device, non_blocking=True)
                    
                    if batch_states.dtype != torch.float32:
                        batch_states = batch_states.float()

                    self.optimizer.zero_grad()
                    logits = self.policy_net(batch_states)
                    loss = self.loss_fn(logits, batch_actions)
                    loss.backward()
                    self.optimizer.step()
                    
                    current_loss = loss.item()
                    total_epoch_loss += current_loss
                    num_batches += 1
                    
                    # Update the progress bar with the current batch loss
                    if num_batches % 10 == 0: # Update text every 10 batches to save CPU
                        pbar.set_postfix({"loss": f"{current_loss:.4f}"})
                
                avg_loss = total_epoch_loss / num_batches if num_batches > 0 else 0.0
                final_loss = avg_loss
                
                epoch_duration = time.time() - epoch_start_time
                throughput = (num_batches * batch_size) / epoch_duration
                print(f"Epoch {epoch+1} Complete - Avg Loss: {avg_loss:.4f} - Speed: {throughput:.0f} samples/sec")

            return self.policy_net, final_loss

        # --- Tabular / Non-Deep Path (Keep as is) ---
        else:
            # compute weights for each state-action pair in the dataset
            state_action_counts = defaultdict(int)
            # Note: If passing a Tensor Dataset to this tabular code, 
            # you might need to convert tensors to tuples to make them hashable.
            for s, a in dataset:
                # Basic conversion for tabular hashing
                if isinstance(s, torch.Tensor): s = s.item()
                if isinstance(a, torch.Tensor): a = a.item()
                state_action_counts[(s, a)] += 1
                
            weights = [1.0 / state_action_counts[(s, a)] for s, a in dataset]
            weights = torch.tensor(weights, dtype=torch.float32)

            states_indices = torch.LongTensor([s for s, a in dataset])
            actions_indices = torch.LongTensor([a for s, a in dataset])
            
            if isinstance(self.policy_net, DeepSoftmaxPolicy):
                states_input = F.one_hot(states_indices, num_classes=self.num_states).float()
            else:
                states_input = states_indices
            
            self.policy_net.train()
            
            final_loss = 0.0
            for epoch in tqdm(range(epochs), desc="Training Imitation Policy"):
                self.optimizer.zero_grad()
                logits = self.policy_net(states_input)
                log_probs = F.log_softmax(logits, dim=-1)
                expert_log_probs = log_probs.gather(1, actions_indices.unsqueeze(1)).squeeze(1)
                nll_loss_per_sample = -expert_log_probs
                weighted_loss = nll_loss_per_sample * weights
                loss = weighted_loss.mean()
                loss.backward()
                self.optimizer.step()
                final_loss = loss.item()

            return self.get_policy_table(), final_loss
        
    def get_policy_table(self):
        """
        Extracts the learned policy as a numpy probability table.
        """
        self.policy_net.eval()
        with torch.no_grad():
            all_states_indices = torch.arange(self.num_states)
            
            # If using DeepSoftmaxPolicy, convert state indices to one-hot encoding
            if isinstance(self.policy_net, DeepSoftmaxPolicy):
                states_input = F.one_hot(all_states_indices, num_classes=self.num_states).float()
            else:
                states_input = all_states_indices
            
            logits = self.policy_net(states_input)
            # Apply softmax to the logits to get the final action probabilities
            probs = F.softmax(logits, dim=-1)
            return probs.numpy()