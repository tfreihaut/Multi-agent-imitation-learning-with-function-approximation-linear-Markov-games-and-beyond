import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from collections import defaultdict

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
        
        # ensure correct shapes
        if x.ndim == 3:  # (C,H,W)
            pass
        elif x.ndim == 2:  # (H,W) - add channel dimension
            x = x.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected state shape: {x.shape}")
        
        return x, y
    

class LinearPolicy(nn.Module):
    """
    Multi-layer linear policy for grid games.
    Uses multiple linear transformations with layer normalization for better representation learning
    while maintaining linearity in the feature space.
    """
    
    def __init__(self, state_dim=9, num_actions=4, hidden_dim=64):
        """
        Args:
            state_dim: Dimension of flattened state (e.g., 3x3=9 for a 3x3 grid)
            num_actions: Number of possible actions
            hidden_dim: Dimension of hidden linear layer
        """
        super().__init__()
        self.flatten = nn.Flatten()
        
        # Multi-layer linear architecture with layer normalization
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Normalize but keep linearity
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, num_actions)
        )
    
    def forward(self, x):
        """
        Args:
            x: State tensor of shape (batch, channels, height, width) or (batch, state_dim)
        
        Returns:
            logits: Action logits of shape (batch, num_actions)
        """
        x = self.flatten(x)
        return self.network(x)
    
    def get_action_distribution(self, state_index):
        """
        Calculates the action probability distribution for a given state index.

        Args:
            state_index (int or torch.Tensor): The index of the state.

        Returns:
            torch.Tensor: A 1D tensor representing the probability distribution over actions.
        """
        self.eval()
        
        with torch.no_grad():
            if not isinstance(state_index, torch.Tensor):
                state_tensor = torch.tensor([state_index], dtype=torch.long)
            else:
                state_tensor = state_index.view(1)
            
            logits = self.forward(state_tensor)
            probs = F.softmax(logits, dim=-1)
            
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
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, num_actions))
        self.network = nn.Sequential(*layers)
        self.eta = eta

    def forward(self, states):
        logits = self.network(states)
        logits = self.eta * logits
        return logits

class BehavioralCloningSingleAgent:
    """
    Implements Behavioral Cloning for a single agent from a fixed dataset.
    This is used for the final Imitation Learning step of the main algorithm.
    """
    def __init__(self, num_states, num_actions, feature_map=None, 
                 feature_map_args=None, lr=1e-3, eta=1.0):
        """
        Initializes the single-agent Behavioral Cloning learner.

        Args:
            num_states (int): The total number of states in the environment.
            num_actions (int): The number of actions for the agent.
            feature_map (callable): The feature map function φ(s, a, ...).
            feature_map_args (dict, optional): Extra arguments for the feature map.
            lr (float): Learning rate for the optimizer.
            eta (float): Temperature parameter for the SoftmaxPolicy.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.feature_map = feature_map
        self.feature_map_args = feature_map_args or {}
        self.lr = lr

        if feature_map is not None:
            # Define a lambda to consistently call the feature map with its arguments
            wrapped_feature_map = lambda s, a: self.feature_map(
                s, a, self.num_states, self.num_actions, **self.feature_map_args
            )

            # Determine the feature dimension by passing a dummy input
            dummy_state = torch.tensor([0])
            dummy_action = torch.tensor([0])
            feature_dim = wrapped_feature_map(dummy_state, dummy_action).shape[-1]
            self.policy_net = SoftLinPolicy(
                feature_map=wrapped_feature_map,
                feature_dim=feature_dim,
                num_actions=self.num_actions,
                eta=eta
            )

        else:
            self.policy_net = DeepSoftmaxPolicy(
                state_dim=self.num_states, # assuming states are represented as one-hot vectors
                num_actions=self.num_actions,
                eta=eta
            )
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

    def train(self, dataset, epochs=10000, batch_size=64, device='cpu'):
        """
        Trains the policy by minimizing the negative log-likelihood on the dataset.
        This is equivalent to: argmin Σ -log π(a|s)
        
        Args:
            dataset (list): A list of (state, action) tuples.
            epochs (int): The number of training epochs.
            
        Returns:
            np.ndarray: The learned policy as a probability table of shape (num_states, num_actions).
        """
        if not dataset:
            print("Warning: Training dataset is empty. Returning a uniform random policy.")
            return np.ones((self.num_states, self.num_actions)) / self.num_actions
        
            # compute weights for each state-action pair in the dataset
        state_action_counts = defaultdict(int)
        for s, a in dataset:
            state_action_counts[(s, a)] += 1
        weights = [1.0 / state_action_counts[(s, a)] for s, a in dataset]
        weights = torch.tensor(weights, dtype=torch.float32)

        # Convert the dataset into PyTorch tensors
        states_indices = torch.LongTensor([s for s, a in dataset])
        actions_indices = torch.LongTensor([a for s, a in dataset])
        
        # If using DeepSoftmaxPolicy, convert state indices to one-hot encoding
        if isinstance(self.policy_net, DeepSoftmaxPolicy):
            states_input = F.one_hot(states_indices, num_classes=self.num_states).float()
        else:
            states_input = states_indices
        
        self.policy_net.train()
        
        final_loss = 0.0
        for epoch in tqdm(range(epochs), desc="Training Imitation Policy"):
            self.optimizer.zero_grad()
            
            # The forward pass of SoftmaxPolicy calculates logits for all actions for each state
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