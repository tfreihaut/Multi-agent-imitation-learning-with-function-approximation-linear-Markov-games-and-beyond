import torch
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
import random
import logging
from datetime import datetime
import os

from itertools import product
from tictactoe.generate_perfect_policy import legal


import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Setup for tictactoe environment
SYMMETRIES = [
    (0, 1, 2, 3, 4, 5, 6, 7, 8),  # identity
    (6, 3, 0, 7, 4, 1, 8, 5, 2),  # rotate 90
    (8, 7, 6, 5, 4, 3, 2, 1, 0),  # rotate 180
    (2, 5, 8, 1, 4, 7, 0, 3, 6),  # rotate 270
    (2, 1, 0, 5, 4, 3, 8, 7, 6),  # reflect vertical
    (6, 7, 8, 3, 4, 5, 0, 1, 2),  # reflect horizontal
    (0, 3, 6, 1, 4, 7, 2, 5, 8),  # reflect main diagonal
    (8, 5, 2, 7, 4, 1, 6, 3, 0),  # reflect anti-diagonal
]


def transform(board, sym):
    return tuple(board[i] for i in sym)


def canonical(board):
    return min(transform(board, s) for s in SYMMETRIES)


def agent_symbol(agent):
    if agent == "player_1":
        return 1  # X
    elif agent == "player_2":
        return -1  # O
    else:
        raise ValueError(f"Unknown agent: {agent}")


def obs_to_board(obs, agent):
    """
    Converts PettingZoo TicTacToe observation
    into absolute board tuple (0, 1, -1)
    """
    own = obs[:, :, 0]
    opp = obs[:, :, 1]

    board = np.zeros((3, 3), dtype=int)
    board[own == 1] = agent_symbol(agent)
    board[opp == 1] = -agent_symbol(agent)

    return tuple(board.reshape(-1))


def board_to_obs(board):
    """
    Converts absolute board tuple (0, 1, -1)
    into PettingZoo TicTacToe observation
    """
    own = np.array(board).reshape(3, 3) == 1
    opp = np.array(board).reshape(3, 3) == -1

    obs = np.zeros((3, 3, 2), dtype=int)
    obs[:, :, 0] = own.astype(int)
    obs[:, :, 1] = opp.astype(int)

    return obs


def player_to_move(board):
    x = board.count(1)
    o = board.count(-1)
    return 1 if x == o else -1


def canonical_action(board, canon_board, canon_action):
    """
    Finds the actual action index corresponding
    to the canonical action.
    """
    for sym in SYMMETRIES:
        if transform(board, sym) == canon_board:
            return sym[canon_action]
    raise ValueError("No symmetry match found")


def canonical_action_to_board_action(board, canon_board, canon_action):
    """
    Given:
      - original board
      - its canonical form
      - an action index in canonical space
    Return:
      - action index in original board space
    """
    for sym in SYMMETRIES:
        if transform(board, sym) == canon_board:
            return sym[canon_action]
    raise ValueError("No matching symmetry found")


def get_vectorized_policy(policy, expert_policy):
    """
    Build a mapping from canonical board -> action using `policy`.

    Important: `policy` was trained as a single agent that expects its `own`
    channel to correspond to the agent to move (player_1 in our training).
    For canonical boards where the player to move is -1, present the network
    with the *flipped* board (multiply by -1) so that the network always
    sees the acting player's pieces as `own`.
    """
    policy.eval()
    all_boards = [
        b
        for b in product([0, 1, -1], repeat=9)
        if legal(b) and b.count(1) == b.count(-1)
    ]

    vec_policy = {}
    device = next(policy.parameters()).device
    for board in all_boards:
        # Get action probabilities from the policy network using the actor-perspective board
        state_np = board_to_obs(board)  # Convert board to observation format (H,W,C)
        # Convert to CHW for CNN and add batch dim
        state_tensor = (
            torch.from_numpy(state_np).float().permute(2, 0, 1).unsqueeze(0).to(device)
        )
        with torch.no_grad():
            logits = policy(state_tensor)
            probs = (
                F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            )  # Convert to numpy array

        # only take legal actions into account (legality doesn't change under sign flip)
        mask = np.array([1 if board[i] == 0 else 0 for i in range(9)], dtype=bool)
        probs = probs * mask  # Mask illegal actions

        # Renormalize to ensure probabilities sum to 1
        prob_sum = probs.sum()
        if prob_sum > 0:
            probs = probs / prob_sum
        else:
            # If all probabilities are zero, use uniform over legal actions
            probs = mask.astype(float) / mask.sum()
        # Ensure that probs sum to 1
        probs = probs / probs.sum()
        # Store in vectorized policy: store action probabilities for the canonical board
        vec_policy[board] = {a: float(probs[a]) for a in range(9) if mask[a]}

    return vec_policy


# Define the structure for storing transitions in the replay buffer
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
    """
    A replay buffer without capacity limit that stores all trajectories collected
    during the exploration phase (before running BC).
    """

    def __init__(self, player_id):
        """
        Initialize the trajectory buffer.

        Args:
            player_id: Integer identifier for the player (e.g., 1 or 2)
        """
        self.player_id = player_id
        self.trajectories = []  # List of trajectories, where each trajectory is a list of transitions
        self.current_trajectory = []  # Current trajectory being built

    def push(self, *args):
        """
        Add a transition to the current trajectory.

        Args:
            *args: Transition components (state, action, expert_action, reward, next_state, state_idx)
        """
        self.current_trajectory.append(Transition(*args))

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
