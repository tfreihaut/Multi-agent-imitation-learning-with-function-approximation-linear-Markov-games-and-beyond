import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class SmallCNN(nn.Module):
    """CNN for small games, outputs logits for 7 actions."""

    def __init__(self, in_channels=2, num_actions=7):
        super().__init__()
        self.conv = nn.Sequential(
            # First layer: Sees 3x3 local patterns (like 3-in-a-row)
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Second layer: Sees how those patterns relate to each other
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Third layer: Deeper features
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_actions),
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


class BehavioralCloningSingleAgent:
    """
    Implements Behavioral Cloning for a single agent from a fixed dataset.
    This is used for the final Imitation Learning step of the main algorithm.
    """

    def __init__(
        self,
        num_actions,
        feature_map=None,
        feature_map_args=None,
        lr=1e-3,
        eta=1.0,
        cnn_policy=False,
        in_channels=1,
        device="cpu",
    ):
        """
        Initializes the single-agent Behavioral Cloning learner.

        Args:
            num_states (int): The total number of states in the environment.
            num_actions (int): The number of actions for the agent.
            feature_map (callable): The feature map function φ(s, a, ...).
            feature_map_args (dict, optional): Extra arguments for the feature map.
            lr (float): Learning rate for the optimizer.
            eta (float): Temperature parameter for the SoftmaxPolicy.
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        self.num_actions = num_actions
        self.feature_map = feature_map
        self.feature_map_args = feature_map_args or {}
        self.lr = lr
        self.cnn = cnn_policy

        self.policy_net = SmallCNN(
            in_channels=in_channels, num_actions=self.num_actions
        )

        # After initializing your model
        self.policy_net = self.policy_net.to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

    def train(self, dataset, epochs=100, batch_size=64, device="cuda"):
        """
        Optimized Behavior Cloning Trainer
        Handles both NumPy and Tensor inputs automatically.
        """
        if not dataset:
            return None, 0.0

        # We process the entire dataset ONCE before the epoch loop starts.
        processed_states = []
        processed_actions = []

        for obs, act in dataset:
            # Handle State (Force to Tensor + Float + CHW)
            if isinstance(obs, np.ndarray):
                s = torch.from_numpy(obs).float()
            else:
                s = obs.detach().float()  # Detach is safety for memory leaks

            # Ensure Channels-First (2, 6, 7) for Connect4 and (2,3,3) for TicTacToe
            if s.shape[-1] == 2:
                s = s.permute(2, 0, 1)

            processed_states.append(s)
            processed_actions.append(torch.tensor(act, dtype=torch.long, device=device))

        states_all = torch.stack(processed_states).to(device)
        # Convert actions: handle both scalars and tensors robustly
        if isinstance(processed_actions[0], torch.Tensor):
            actions_all = torch.stack(processed_actions).to(device, dtype=torch.long)
        else:
            actions_all = torch.as_tensor(
                processed_actions, dtype=torch.long, device=device
            )

        num_samples = len(dataset)
        self.policy_net.to(device).train()
        loss_fn = nn.CrossEntropyLoss()

        for _ in range(epochs):
            # Shuffle indices directly on the GPU
            indices = torch.randperm(num_samples, device=device)

            for i in range(0, num_samples, batch_size):
                idx = indices[i : i + batch_size]

                # Slicing the pre-loaded GPU tensors is instant
                batch_states = states_all[idx]
                batch_actions = actions_all[idx]

                self.optimizer.zero_grad(set_to_none=True)
                logits = self.policy_net(batch_states)
                loss = loss_fn(logits, batch_actions)
                loss.backward()
                self.optimizer.step()

        return self.policy_net, loss.item()
