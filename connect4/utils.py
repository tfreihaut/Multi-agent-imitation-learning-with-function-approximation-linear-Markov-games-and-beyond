import sys
from pathlib import Path

import torch
import random

import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from collections import deque, namedtuple
import os
import pickle
from datetime import datetime
import numpy as np

from torch.utils.data import Dataset


Transition = namedtuple(
    "Transition", ("state", "action", "expert_action", "reward", "next_state")
)


def setup_logging(log_file="deep_mail_training.log"):
    """
    Set up logging configuration for training.

    Args:
        log_file: Name of the log file to write to

    Returns:
        logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    # Create a timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{timestamp}_{log_file}"

    # Configure logger
    logger = logging.getLogger("DeepMAIL")
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Log file: {log_path}")

    return logger, log_path


def format_time(seconds):
    """
    Format seconds into a human-readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        str: Formatted time string (e.g., "1h 23m 45s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


class Connect4Dataset(Dataset):
    def __init__(self, trajectories, limit_traj=None):
        self.indices = []
        self.trajectories = trajectories

        # Flatten the structure: map index -> (traj_idx, step_idx)
        for i, traj in enumerate(trajectories):
            if limit_traj is not None and i >= limit_traj:
                break
            for step_idx in range(len(traj)):
                self.indices.append((i, step_idx))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        traj_idx, step_idx = self.indices[idx]
        transition = self.trajectories[traj_idx][step_idx]

        # 1. Get data (It is now a Numpy Array, likely int8)
        state_np = transition.state
        expert_action_np = transition.expert_action

        # 2. Inflate to Tensor for the Neural Network
        state = torch.from_numpy(state_np).float()
        action = torch.tensor(int(expert_action_np), dtype=torch.long)

        return state, action


class ReplayBuffer:
    """A simple replay buffer."""

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition."""
        self.memory.append(Transition(*args))
        if len(self.memory) > self.memory.maxlen:
            self.memory.popleft()

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def clear(self):
        self.memory.clear()

    def iterator(self):
        return iter(self.memory)

    def __len__(self):
        return len(self.memory)


class TrajectoryBuffer:
    def __init__(self, player_id):
        self.player_id = player_id
        # Optimization: We will still store lists of trajectories,
        # but the inner data will be lightweight numpy, not torch tensors.
        self.trajectories = []
        self.current_trajectory = []

    def push(self, state, action, expert_action, reward, next_state):
        """
        Compresses data to CPU NumPy arrays immediately to save RAM.
        """

        # Helper to safely convert any tensor/array to a detached cpu numpy array
        def clean(x, dtype):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                # .detach() cuts the graph (prevents memory leak)
                # .cpu() ensures it's in RAM
                x = x.detach().cpu().numpy()
            return x.astype(dtype)

        # 1. Compress State: Float32 (4 bytes) -> Int8 (1 byte)
        # Assuming your board data is discrete (0, 1, 2, etc.)
        # If your observations are normalized 0.0-1.0, use float16 or keep float32.
        cpu_state = clean(state, np.int8)
        cpu_next_state = clean(next_state, np.int8)

        # 2. Compress Actions: Long (8 bytes) -> Int16 (2 bytes)
        cpu_action = clean(action, np.int16)
        cpu_expert_action = clean(expert_action, np.int16)

        # 3. Reward: Keep as float32
        cpu_reward = clean(reward, np.float32)

        # Store pure lightweight data
        self.current_trajectory.append(
            Transition(
                cpu_state, cpu_action, cpu_expert_action, cpu_reward, cpu_next_state
            )
        )

    def end_trajectory(self):
        """
        Mark the current trajectory as complete and start a new one.
        """
        if self.current_trajectory:
            self.trajectories.append(self.current_trajectory)
            self.current_trajectory = []

    def get_all_transitions(self):
        """
        Get all transitions from all trajectories as a flat list.

        Returns:
            List of all Transition objects
        """
        all_transitions = []
        for trajectory in self.trajectories:
            all_transitions.extend(trajectory)
        # Include current trajectory if it's not empty
        if self.current_trajectory:
            all_transitions.extend(self.current_trajectory)
        return all_transitions

    def get_trajectories(self):
        """
        Get all complete trajectories.

        Returns:
            List of trajectories (list of lists of Transitions)
        """
        return self.trajectories

    def num_trajectories(self):
        """Return the number of complete trajectories."""
        return len(self.trajectories)

    def num_transitions(self):
        """Return the total number of transitions across all trajectories."""
        return sum(len(traj) for traj in self.trajectories) + len(
            self.current_trajectory
        )

    def save(self, filepath):
        """
        Save the trajectory buffer to a file.

        Args:
            filepath: Path where the buffer should be saved
        """
        # Create directory if it doesn't exist
        os.makedirs(
            os.path.dirname(filepath) if os.path.dirname(filepath) else ".",
            exist_ok=True,
        )

        # Convert trajectories to a serializable format
        data = {
            "player_id": self.player_id,
            "num_trajectories": self.num_trajectories(),
            "num_transitions": self.num_transitions(),
            "trajectories": [],
        }

        for trajectory in self.trajectories:
            traj_data = []
            for transition in trajectory:
                # Convert tensors to numpy for serialization
                traj_data.append(
                    {
                        "state": transition.state.cpu().numpy()
                        if isinstance(transition.state, torch.Tensor)
                        else transition.state,
                        "action": transition.action.cpu().numpy()
                        if isinstance(transition.action, torch.Tensor)
                        else transition.action,
                        "expert_action": transition.expert_action.cpu().numpy()
                        if isinstance(transition.expert_action, torch.Tensor)
                        else transition.expert_action,
                        "reward": transition.reward.cpu().numpy()
                        if isinstance(transition.reward, torch.Tensor)
                        else transition.reward,
                        "next_state": transition.next_state.cpu().numpy()
                        if isinstance(transition.next_state, torch.Tensor)
                        else transition.next_state,
                    }
                )
            data["trajectories"].append(traj_data)

        # Save to file using numpy
        np.save(filepath, data, allow_pickle=True)
        print(f"Saved trajectory buffer for Player {self.player_id} to {filepath}")
        print(f"  - {data['num_trajectories']} trajectories")
        print(f"  - {data['num_transitions']} total transitions")

    def save_optimized(self, filepath):
        # Open the file in binary write mode
        with open(filepath, "wb") as f:
            # 1. Save metadata first
            metadata = {
                "player_id": self.player_id,
                "num_trajectories": self.num_trajectories(),
                "num_transitions": self.num_transitions(),
            }
            pickle.dump(metadata, f)

            # 2. Save trajectories one by one
            # No massive 'data' variable is ever created!
            for trajectory in self.trajectories:
                # If you applied the int8 optimization, 'trajectory' is already
                # lightweight numpy arrays. Just dump it.
                pickle.dump(trajectory, f)

        print(f"Saved to {filepath} using stream optimization.")

    def load(self, filepath, device="cpu"):
        """
        Load the trajectory buffer from a file.

        Args:
            filepath: Path to the saved buffer file
            device: Torch device to load tensors to
        """
        data = np.load(filepath, allow_pickle=True).item()

        self.player_id = data["player_id"]
        self.trajectories = []

        for traj_data in data["trajectories"]:
            trajectory = []
            for trans_data in traj_data:
                # Convert numpy arrays back to tensors
                transition = Transition(
                    state=torch.from_numpy(trans_data["state"]).to(device),
                    action=torch.from_numpy(trans_data["action"]).to(device)
                    if isinstance(trans_data["action"], np.ndarray)
                    else torch.tensor(trans_data["action"], device=device),
                    expert_action=torch.from_numpy(trans_data["expert_action"]).to(
                        device
                    )
                    if isinstance(trans_data["expert_action"], np.ndarray)
                    else torch.tensor(trans_data["expert_action"], device=device),
                    reward=torch.from_numpy(trans_data["reward"]).to(device)
                    if isinstance(trans_data["reward"], np.ndarray)
                    else torch.tensor(trans_data["reward"], device=device),
                    next_state=torch.from_numpy(trans_data["next_state"]).to(device),
                )
                trajectory.append(transition)
            self.trajectories.append(trajectory)

        print(f"Loaded trajectory buffer for Player {self.player_id} from {filepath}")
        print(f"  - {self.num_trajectories()} trajectories")
        print(f"  - {self.num_transitions()} total transitions")

    def clear(self):
        """Clear all trajectories."""
        self.trajectories = []
        self.current_trajectory = []

    def __len__(self):
        """Return the total number of transitions."""
        return self.num_transitions()

    def iterator(self):
        """Iterator over all transitions in the buffer."""
        return iter(self.get_all_transitions())
