import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np




from torch.utils.data import Dataset, DataLoader

class BehavioralDataset(Dataset):
    def __init__(self, expert_trajectories):
        # expert_trajectories is a list of (state_tensor, action_tensor)
        self.states = [s for s, _ in expert_trajectories]
        self.actions = [a for _, a in expert_trajectories]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        x = self.states[idx]
        y = self.actions[idx]
        # ensure correct shapes
        if x.ndim == 3:  # (C,H,W)
            pass
        else:
            x = x.unsqueeze(0)
        return x, y


class LinearPolicy(nn.Module):
    """
    Multi-layer linear policy for grid games.
    Uses multiple linear transformations with layer normalization for better representation learning
    while maintaining linearity in the feature space.
    """
    
    def __init__(self, state_dim=9, num_actions=4, hidden_dim=12):
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


class BehavioralCloningDeep:
    """
    Implements the Behavioral Cloning algorithm for a two-player zero-sum game.
    It learns policies for both players by mimicking expert demonstrations.
    Supports both linear and CNN architectures.
    """
    def __init__(self, expert_policies, env, dataset_size, transitions, initial_dist, rewards, gamma, 
                 lr=1e-3, eta=1.0, device='cpu', use_linear=True):
        """
        Initializes the BehavioralCloning agent.
        
        Args:
            expert_policies: Tuple of expert policies for both players
            dataset_size: Number of state-action pairs to generate
            env: GridZeroSum environment
            transitions: Transition matrix
            initial_dist: Initial state distribution  
            rewards: Reward matrix
            gamma: Discount factor
            lr: Learning rate
            eta: Temperature parameter
            device: Device to run on ('cpu' or 'cuda')
            use_linear: If True, use LinearPolicy; if False, use SmallCNN
        """
        self.expert_policy_p1 = expert_policies[0]
        self.expert_policy_p2 = expert_policies[1]
        self.dataset_size = dataset_size
        self.transitions = transitions
        self.initial_dist = initial_dist
        self.rewards = rewards
        self.gamma = gamma
        self.eta = eta
        self.env = env
        self.use_linear = use_linear

        self.num_states = transitions.shape[0]
        self.num_actions_p1 = self.expert_policy_p1.shape[1]
        self.num_actions_p2 = self.expert_policy_p2.shape[1]

        state_dim = env.length * env.width  # for linear model
        
        self.policy_p0 = LinearPolicy(state_dim, self.num_actions_p1).to(device)
        self.policy_p1 = LinearPolicy(state_dim, self.num_actions_p2).to(device)


        self.optimizer_p0 = optim.Adam(self.policy_p0.parameters(), lr=lr)
        self.optimizer_p1 = optim.Adam(self.policy_p1.parameters(), lr=lr)



    def generate_expert_trajectories(self):
        """
        Sample expert trajectories under expert policies using geometric episode lengths.
        
        This method generates training data by sampling trajectories from the expert policies.
        Each trajectory starts from the initial state distribution and continues for a 
        geometrically distributed number of steps (with parameter 1-gamma).
        
        The sampling process:
        1. Sample initial state s₀ from initial_state_dist
        2. Sample episode length T ~ Geometric(1-gamma)
        3. For each timestep t = 0, 1, ..., T-1:
           - Sample actions a₁ᵗ ~ expert1[sᵗ], a₂ᵗ ~ expert2[sᵗ]
           - Record state-action pairs (sᵗ, a₁ᵗ) and (sᵗ, a₂ᵗ)
           - Sample next state sᵗ⁺¹ ~ P(·|sᵗ, a₁ᵗ, a₂ᵗ)
        4. Repeat until total_samples state-action pairs are collected
        
        Returns:
            tuple: (data1, data2) where:
                - data1: List of (state, action) pairs for player 1
                - data2: List of (state, action) pairs for player 2
                Each list contains exactly total_samples tuples.
                
        Note:
            The geometric episode length models the discounted future importance
            and ensures that trajectories have finite expected length.
        """
        data1, data2 = [], []
        samples = 0

        while samples < self.dataset_size:
            s = np.random.choice(self.initial_dist.shape[0], p=self.initial_dist)
            random_length = np.random.geometric(1 - self.gamma)

            for _ in range(random_length):
                a1 = np.random.choice(self.expert_policy_p1.shape[1], p=self.expert_policy_p1[s])
                a2 = np.random.choice(self.expert_policy_p2.shape[1], p=self.expert_policy_p2[s])
                positions = self.env.all_states[s]
                state_array = self.env.array_state_representation(positions)  # Already (1,3,3) channels-first

                # state_array is already in correct format (C, H, W)
                # No need to transpose

                # Normalize to float
                state_tensor = torch.tensor(state_array, dtype=torch.float32)

                action_tensor1 = torch.tensor(a1, dtype=torch.long)
                action_tensor2 = torch.tensor(a2, dtype=torch.long)

                data1.append((state_tensor, action_tensor1))
                data2.append((state_tensor, action_tensor2))

                # sample next state
                prob = self.transitions[s, a1, a2]
                s = np.random.choice(self.initial_dist.shape[0], p=prob)

            samples += random_length
            data1 = data1[:self.dataset_size]
            data2 = data2[:self.dataset_size]

        return data1, data2


    def train(self, epochs=100, batch_size=64, device='cpu'):
        expert_trajectories_p1, expert_trajectories_p2 = self.generate_expert_trajectories()

        print(f"Generated {len(expert_trajectories_p1)} samples for Player 1 and {len(expert_trajectories_p2)} for Player 2")

        self._train_player(self.policy_p0, self.optimizer_p0, expert_trajectories_p1, epochs, batch_size, device)
        self._train_player(self.policy_p1, self.optimizer_p1, expert_trajectories_p2, epochs, batch_size, device)

        final_policy_p0 = self.get_policy_table(self.policy_p0)
        final_policy_p1 = self.get_policy_table(self.policy_p1)
        return final_policy_p0, final_policy_p1

    def _train_player(self, policy_net, optimizer, expert_trajectories, epochs, batch_size=64, device='cpu'):
        dataset = BehavioralDataset(expert_trajectories)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        policy_net.train()
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for states, actions in loader:
                states = states.to(device)
                actions = actions.to(device)

                optimizer.zero_grad()
                logits = policy_net(states)
                loss = loss_fn(logits, actions)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * states.size(0)

            avg_loss = epoch_loss / len(dataset)
            if (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"[Player] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    def get_policy_table(self, policy_net):
        policy_net.eval()
        all_states = []

        for positions in self.env.all_states:
            state_array = self.env.array_state_representation(positions)  # Already (1,3,3) channels-first
            state_tensor = torch.tensor(state_array, dtype=torch.float32)
            all_states.append(state_tensor)

        all_states = torch.stack(all_states)  # (num_states, 1, 3, 3)

        with torch.no_grad():
            logits = policy_net(all_states)
            probs = F.softmax(logits, dim=-1)

        return probs.cpu().numpy()

        
    def _policy_value_zero_sum(self, mu_pi, nu_pi, reward, gamma, init_dist, P) -> float:
        """
        Evaluate the *joint* value V^{mu, \nu} = E[ ∑ gamma^t r(s_t,a_t,b_t) ] 
        by solving the linear system (I - gammaP_π)^-1 r_π and averaging over
        initial uniform state.
        """
        # Build 1-step expected reward per state
        r_s = (reward * mu_pi[:, :, None] * nu_pi[:, None, :]).sum(axis=(1,2))
        # Build joint transition matrix P_π[s,s'] = ∑_{a,b} μ(a|s) ν(b|s) P[s,a,b,s']
        P_joint = (P * mu_pi[:, :, None, None] * nu_pi[:, None, :, None]).sum(axis=(1,2))
        I = np.eye(P.shape[0])
        V = np.linalg.solve(I - gamma * P_joint, r_s)
        
        return float(init_dist @ V)

    def _value_iteration(self, R: np.ndarray, P: np.ndarray, gamma, init_dist, axis: int,
                         tol: float = 1e-6, max_iter: int = 10_000, ) -> float:
        """
        Standard value iteration for single-agent MDP:
            R: shape (S, A)
            P: shape (S, A, S')
        Returns optimal state-value averaged under uniform start.
        """
        S, A = R.shape
        V = np.zeros(S)
        for _ in range(max_iter):
            # Q(s,a) = R[s,a] + gamma ∑ P[s,a,s'] V[s']
            Q = R + gamma * (P @ V)
            V_new = Q.max(axis=1)
            if np.max(np.abs(V_new - V)) < tol:
                V = V_new
                break
            V = V_new
        return float(init_dist @ V)

    def calc_exploitability_true(self, mu_pi, nu_pi , reward: np.ndarray, P: np.ndarray, gamma: float) -> float:
        """
        Compute exploitability under the *true* reward r[s,a1,a2]:
        
        1) V^{mu,nu} = value under the current joint policy
        2) V^{BR_mu} = best-response value if mu could re-optimize against nu
        3) V^{BR_nu} = best-response value if nu could re-optimize against mu
        
        Return max{ V^{BR_mu} - V^{mu,nu}, V^{BR_nu} - V^{mu,nu} }.
        """
        # 1) Compute V^{μ,ν} by solving the full zero‐sum game via simple policy eval
        V_joint = self._policy_value_zero_sum(mu_pi, nu_pi, reward=reward, gamma=gamma,P=P, init_dist=self.initial_dist)

        # 2) Build single‐agent MDP for μ as decision‐maker:
        #    R_μ(s,a) = E_{b∼ν_pi(s)}[ r(s,a,b) ]
        R_mu = (reward * nu_pi[:, None, :]).sum(axis=2)  # shape (S, A1)
            #    P_mu[s,a,s'] = ∑_b ν_pi(b|s) P[s,a,b,s']
        P_mu = (P * nu_pi[:, None, :, None]).sum(axis=2)  # shape (S, A1, S')
        # best‐response value for μ:
        V_br_mu = self._value_iteration(R_mu, P_mu, axis=1, gamma=gamma, init_dist=self.initial_dist)
        

        # 3) ν's induced MDP (with negated rewards):
        #    R_nu[s,b] = - E_{a∼μ_pi(s)}[ r(s,a,b) ]
        R_nu = - (reward * mu_pi[:, :, None]).sum(axis=1)  # (S, A2)
        #    P_nu[s,b,s'] = ∑_a μ_pi(a|s) P[s,a,b,s']
        P_nu = (P * mu_pi[:, :, None, None]).sum(axis=1)  # (S, A2, S')
        V_br_nu = self._value_iteration(R_nu, P_nu, gamma=gamma, init_dist=self.initial_dist, axis=1)

        return float(max(V_br_mu - V_joint, V_br_nu - V_joint))
    