import numpy as np
from linear.bc import BehavioralCloningSingleAgent


class BehavioralCloning:
    """
    Implements the Behavioral Cloning algorithm for a two-player zero-sum game.
    It learns policies for both players by mimicking expert demonstrations.
    
    This class wraps two BehavioralCloningSingleAgent instances, one for each player,
    and handles the data generation from expert policies.
    """
    def __init__(self, expert_policies, dataset_size, transitions, initial_dist, rewards, gamma, 
                 state_action_feature_map_p1, state_action_feature_map_p2, 
                 feature_map_args_p1=None, feature_map_args_p2=None, lr=1e-3, eta=1.0):
        """
        Initializes the BehavioralCloning agent.
        
        Args:
            expert_policies: Tuple of expert policies for both players
            dataset_size: Number of state-action pairs to generate
            transitions: Transition matrix
            initial_dist: Initial state distribution  
            rewards: Reward matrix
            gamma: Discount factor
            state_action_feature_map_p1: Feature map function for player 1
            state_action_feature_map_p2: Feature map function for player 2
            feature_map_args_p1: Dictionary of additional arguments for player 1's feature map
            feature_map_args_p2: Dictionary of additional arguments for player 2's feature map
            lr: Learning rate
            eta: Temperature parameter
        """
        self.expert_policy_p1 = expert_policies[0]
        self.expert_policy_p2 = expert_policies[1]
        self.dataset_size = dataset_size
        self.transitions = transitions
        self.initial_dist = initial_dist
        self.rewards = rewards
        self.gamma = gamma

        self.num_states = transitions.shape[0]
        self.num_actions_p1 = self.expert_policy_p1.shape[1]
        self.num_actions_p2 = self.expert_policy_p2.shape[1]
        
        # Create BehavioralCloningSingleAgent for each player
        self.learner_p1 = BehavioralCloningSingleAgent(
            num_states=self.num_states,
            num_actions=self.num_actions_p1,
            feature_map=state_action_feature_map_p1,
            feature_map_args=feature_map_args_p1,
            lr=lr,
            eta=eta
        )
        
        self.learner_p2 = BehavioralCloningSingleAgent(
            num_states=self.num_states,
            num_actions=self.num_actions_p2,
            feature_map=state_action_feature_map_p2,
            feature_map_args=feature_map_args_p2,
            lr=lr,
            eta=eta
        )


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
        data1 = []  # list of (s,a)
        data2 = []
        samples = 0
        while samples < self.dataset_size:
            s = np.random.choice(self.initial_dist.shape[0], p=self.initial_dist)
            random_length = np.random.geometric(1-self.gamma)
            for _ in range(random_length):
                a1 = np.random.choice(self.expert_policy_p1.shape[1], p=self.expert_policy_p1[s])
                a2 = np.random.choice(self.expert_policy_p2.shape[1], p=self.expert_policy_p2[s])
                data1.append((s, a1))
                data2.append((s, a2))
                # sample next state
                prob = self.transitions[s, a1, a2]
                s = np.random.choice(self.initial_dist.shape[0], p=prob)
            samples += random_length
            data1 = data1[:self.dataset_size]
            data2 = data2[:self.dataset_size]

        return data1, data2

    def train(self, epochs=1000):
        """
        Train both players' policies using behavioral cloning.
        
        Args:
            epochs: Number of training epochs            
        Returns:
            tuple: (policy_p1, policy_p2, (loss_p1, loss_p2), avg_unique_states)
        """
        expert_trajectories_p1, expert_trajectories_p2 = self.generate_expert_trajectories()
        
        # Count unique states for each player
        unique_states_p1 = len(set(s for (s, _) in expert_trajectories_p1))
        unique_states_p2 = len(set(s for (s, _) in expert_trajectories_p2))
        
        # Train player 1
        policy_p1, loss_p1 = self.learner_p1.train(expert_trajectories_p1, epochs=epochs)
        
        # Train player 2
        policy_p2, loss_p2 = self.learner_p2.train(expert_trajectories_p2, epochs=epochs)
        
        print(f"Final Loss - Player 1: {loss_p1:.6f}, Player 2: {loss_p2:.6f}")
        print(f"BC Unique States - Player 1 (μ): {unique_states_p1}, Player 2 (ν): {unique_states_p2}")
        
        return policy_p1, policy_p2, (loss_p1, loss_p2), (unique_states_p1, unique_states_p2)