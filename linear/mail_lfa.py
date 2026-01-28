import numpy as np
from collections import defaultdict
from linear.bc import BehavioralCloningSingleAgent   

class TwoPlayerLSVI_UCB:
    """
    Implements the Reward-Free RL exploration algorithm (LSVI-UCB style)
    for a two-player MDP where Player 2 follows a fixed policy.
    """

    def __init__(self, feature_map, d, H, num_states, num_actions, beta, lambda_reg=1.0, feature_map_args={}):
        """
        Initializes the LSVI-UCB agent.

        Args:
            feature_map (callable): Function mapping (state, action) to a d-dim vector.
            d (int): Dimension of the feature map.
            H (int): The horizon for each episode.
            num_actions (int): The number of actions available to Player 1.
            beta (float): Hyperparameter for the UCB exploration bonus.
            lambda_reg (float): Regularization parameter (I in the paper).
        """
        self.feature_map = feature_map
        self.d = d
        self.H = H
        self.num_states = num_states
        self.num_actions = num_actions
        self.beta = beta
        self.lambda_reg = lambda_reg
        self.feature_map_args = feature_map_args

        # Data storage, indexed by timestep h (from 1 to H)
        # dataset[h] stores list of (s, a, s_next) tuples
        self.dataset = defaultdict(list)
        # Stores tuples of (phi_vector, next_state_value) for each timestep h.
        self.history = {h: [] for h in range(1, H + 1)}

    def perform_sanity_check(self, P_matrix, policy_p2_matrix, initial_state_sampler):
        """
        Solves the MDP using the learned exploration rewards via Value Iteration
        to verify that the value of the initial state goes to zero as K increases.
        """
        # print("\n--- RUNNING SANITY CHECK ---")

        # --- Step 1: Finalize the learned models (Lambda_h) after all K episodes ---
        final_Lambda = {}
        for h in range(1, self.H + 1):
            sum_of_outers = np.zeros((self.d, self.d))
            for prev_phi, _ in self.history[h]:
                sum_of_outers += np.outer(prev_phi, prev_phi)
            final_Lambda[h] = self.lambda_reg * np.identity(self.d) + sum_of_outers

        final_Lambda_inv = {h: np.linalg.inv(final_Lambda[h]) for h in final_Lambda}

        # --- Step 2: Define the learned average reward function r_bar ---
        def r_bar(h, s, a):
            phi_sa = self.feature_map(s, a, self.num_states, self.num_actions, **self.feature_map_args)
            phi_sa = np.array(phi_sa, dtype=np.float64)
            bonus = self.beta * np.sqrt(phi_sa.T @ final_Lambda_inv[h] @ phi_sa)
            u_h = min(bonus, self.H)
            return u_h / self.H

        # --- Step 3: Compute the single-agent transition model P(s'|s,a) ---
        num_opponent_actions = policy_p2_matrix.shape[1]
        P_single_agent = np.zeros((self.num_states, self.num_actions, self.num_states))
        for s in range(self.num_states):
            for a in range(self.num_actions):
                for b in range(num_opponent_actions):
                    prob_b = policy_p2_matrix[s, b]
                    # Assumes P_matrix is of shape (S, A1, A2, S)
                    P_single_agent[s, a, :] += prob_b * P_matrix[s, a, b, :]

        # --- Step 4: Solve for V* using finite-horizon Value Iteration ---
        V = np.zeros(self.num_states)  # V represents V_{h+1}
        for h in range(self.H, 0, -1):
            Q_h = np.zeros((self.num_states, self.num_actions))
            # Calculate expected future value for all state-action pairs
            expected_future_V = P_single_agent @ V
            
            # Calculate rewards for all state-action pairs at step h
            rewards_h = np.array([[r_bar(h, s, a) for a in range(self.num_actions)] for s in range(self.num_states)])
            
            Q_h = rewards_h + expected_future_V
            V_new = np.max(Q_h, axis=1)
            V = V_new # V is now V_h, ready for the next iteration (h-1)

        # --- Step 5: Perform the check on the initial state ---
        s0 = initial_state_sampler()
        initial_state_value = V[s0] # This is V_1(s0)

        num_episodes = len(self.history[1])
        # print(f"Solved MDP with learned reward r_bar after K={num_episodes} episodes.")
        # print(f"Value of initial state s0={s0}: V*(s0) = {initial_state_value:.6f}")
        # print("This value should decrease as you increase K in your experiment.")
        # print("--------------------------")

    def run_exploration(self, K, transition_P, policy_p2, initial_state_sampler, player_idx):
        """
        Runs the main exploration loop for K episodes.

        Args:
            K (int): The total number of episodes to run (see Line 3 in Algo 1).
            transition_P (callable or np.ndarray): The transition function P(s, a, b) -> s_next.
                                                   If np.ndarray, it should be a transition matrix of shape 
                                                   (num_states, num_actions_p1, num_actions_p2, num_states).
            policy_p2 (callable or np.ndarray): The fixed policy for Player 2, pi2(s) -> b.
                                                If np.ndarray, it should be a policy table of shape (num_states, num_actions).
            initial_state_sampler (callable): A function that samples an initial state s1.

        Returns:
            list: A dataset containing tuples of (h, s, a, s_next) for all observed transitions.
        """
        
        # Handle policy_p2 - convert policy table to callable if needed
        if isinstance(policy_p2, np.ndarray):
            # If policy_p2 is a policy table, convert it to a callable function
            def policy_p2_func(s):
                return np.random.choice(policy_p2.shape[1], p=policy_p2[s])
            policy_p2_callable = policy_p2_func
        else:
            # If policy_p2 is already callable, use it directly
            policy_p2_callable = policy_p2
            
        # Handle transition_P - convert transition matrix to callable if needed
        if isinstance(transition_P, np.ndarray):
            # If transition_P is a transition matrix, convert it to a callable function
            # Expected shape: (num_states, num_actions_p1, num_actions_p2, num_states)
            def transition_func(s, a, b):
                # Sample next state according to transition probabilities
                return np.random.choice(transition_P.shape[-1], p=transition_P[s, a, b])
            transition_P_callable = transition_func
        else:
            # If transition_P is already callable, use it directly
            transition_P_callable = transition_P
        
        def random_argmax(q_values):
            """Selects an action randomly from the set of optimal actions."""
            max_q = np.max(q_values)
            # Find all indices where the q_value is close to the max
            best_actions = np.where(np.isclose(q_values, max_q))[0]
            # Randomly choose one from the best actions
            return np.random.choice(best_actions)

        # print(f"Running exploration for {K} episodes...")
        for k in range(1, K + 1):
            # Define Q and V functions for the current step h (Line 11)
            def get_q_func(current_h, current_w, current_Lambda_inv):
                def q_func(s, a):
                    phi_sa = self.feature_map(s, a, self.num_states, self.num_actions, **self.feature_map_args)
                    # Convert to numpy array with consistent dtype
                    if hasattr(phi_sa, 'numpy'):
                        phi_sa = phi_sa.numpy().astype(np.float64)
                    elif hasattr(phi_sa, 'detach'):
                        phi_sa = phi_sa.detach().numpy().astype(np.float64)
                    else:
                        phi_sa = np.array(phi_sa, dtype=np.float64)
                    
                    # 1. Optimism Bonus (u_h^k) - Corrected with min(..., H)
                    optimism_bonus = self.beta * np.sqrt(phi_sa.T @ current_Lambda_inv @ phi_sa)
                    
                    u_h = min(optimism_bonus, self.H)

                    # 2. Exploration Reward (r_h^k) - This was missing
                    r_h = u_h / self.H

                    # 3. Q-value update - Corrected to include both terms
                    q_val = np.dot(current_w, phi_sa) + r_h + u_h
                    return min(q_val, self.H)
                return q_func

            # Planning Phase (Backward loop from H to 1)
            # Corresponds to Lines 6-12 in Algorithm 1
            V_k = {} # Value function for current episode k, V_k[h](s)
            Q_k = {} # Q-function for current episode k, Q_k[h](s, a)
            policy_k = {} # Policy for current episode k, policy_k[h](s)

            # V_{H+1} is always zero
            V_k[self.H + 1] = lambda s: 0.0

            for h in range(self.H, 0, -1):
                sum_of_outers = np.zeros((self.d, self.d))
                sum_of_phi_v = np.zeros(self.d)
                
                current_v_func = V_k[h + 1]
                for prev_phi, prev_s_next in self.history[h]:
                    sum_of_outers += np.outer(prev_phi, prev_phi)
                    v_label = current_v_func(prev_s_next)
                    sum_of_phi_v += prev_phi * v_label

                Lambda_h_k = self.lambda_reg * np.identity(self.d) + sum_of_outers
                # This sum is the second term in the w_h^k calculation
                Lambda_inv = np.linalg.inv(Lambda_h_k)

                # Calculate weights w_h^k
                # w = Lambda_inv * sum(phi(s,a) * V_{h+1}(s_next))
                w_h = Lambda_inv @ sum_of_phi_v

                Q_k[h] = get_q_func(h, w_h, Lambda_inv)
                V_k[h] = lambda s, h_val=h, q_dict=Q_k: max(q_dict[h_val](s, a) for a in range(self.num_actions))
                policy_k[h] = lambda s, h_val=h, q_dict=Q_k: random_argmax([q_dict[h_val](s, a) for a in range(self.num_actions)])

            # Phase 2: Data Collection & Storing History
            s_h = initial_state_sampler()
            for h in range(1, self.H + 1):
                a_h = policy_k[h](s_h)
                b_h = policy_p2_callable(s_h)
                if player_idx == 1:
                    s_next = transition_P_callable(s_h, a_h, b_h)
                else:
                    s_next = transition_P_callable(s_h, b_h, a_h)
                phi_h = self.feature_map(s_h, a_h, self.num_states, self.num_actions, **self.feature_map_args)
                if hasattr(phi_h, 'numpy'):
                    phi_h = phi_h.numpy().astype(np.float64)
                elif hasattr(phi_h, 'detach'):
                    phi_h = phi_h.detach().numpy().astype(np.float64)
                else:
                    phi_h = np.array(phi_h, dtype=np.float64)
                self.history[h].append((phi_h, s_next))
                self.dataset[h].append((s_h, a_h, b_h, s_next))
                s_h = s_next
    
        flat_dataset = []
        for h in self.dataset:
            flat_dataset.extend(self.dataset[h])
        return flat_dataset

class MAIL_LFA:
    """
    Implements the main Reward-Free Exploration and Imitation algorithm.
    """
    def __init__(self, H, K, num_states, num_actions_p1, num_actions_p2,
                 feature_map_p1, feature_map_p2, d, expert_policy_p1, expert_policy_p2,
                 transition_P, initial_state_sampler, beta, feature_map_args_p1={}, feature_map_args_p2={}):
        """
        Initializes the main algorithm orchestrator.
        """
        self.H = H
        self.K = K
        self.num_states = num_states
        self.num_actions_p1 = num_actions_p1
        self.num_actions_p2 = num_actions_p2
        self.feature_map_p1 = feature_map_p1
        self.feature_map_p2 = feature_map_p2
        self.d = d
        self.feature_map_args_p1 = feature_map_args_p1
        self.feature_map_args_p2 = feature_map_args_p2
        self.expert_policy_p1 = expert_policy_p1
        self.expert_policy_p2 = expert_policy_p2
        self.transition_P = transition_P
        self.initial_state_sampler = initial_state_sampler
        self.beta = beta

    def run(self, lambda_reg=1.0, epochs=None):
        """
        Executes the full exploration and imitation pipeline.
        """
        # --- Phase 1: Exploration from Player 1's perspective ---
        
        # Prepare feature map for Player 1
        wrapped_feature_map_p1_explorer = self.feature_map_p1
        wrapped_feature_map_p1_learner = self.feature_map_p1
        feature_map_args_p1 = self.feature_map_args_p1
        
        # Initialize the explorer for Player 1
        explorer_p1 = TwoPlayerLSVI_UCB(
            feature_map=wrapped_feature_map_p1_explorer,
            d=self.d,
            H=self.H,
            num_states=self.num_states,
            num_actions=self.num_actions_p1,
            beta=self.beta,
            feature_map_args=feature_map_args_p1,
            lambda_reg=lambda_reg
        )
        
        # Player 1 explores against the fixed expert policy of Player 2
        D_mu_nuE = explorer_p1.run_exploration(
            K=self.K,
            transition_P=self.transition_P,
            policy_p2=self.expert_policy_p2,
            initial_state_sampler=self.initial_state_sampler,
            player_idx=1
        )
        explorer_p1.perform_sanity_check(
            self.transition_P,
            self.expert_policy_p2,
            self.initial_state_sampler
        )

        # --- Phase 2: Exploration from Player 2's perspective ---

        # Prepare feature map for Player 2
        wrapped_feature_map_p2_explorer = self.feature_map_p2
        wrapped_feature_map_p2_learner = self.feature_map_p2
        feature_map_args_p2 = self.feature_map_args_p2

        # Initialize the explorer for Player 2
        explorer_p2 = TwoPlayerLSVI_UCB(
            feature_map=wrapped_feature_map_p2_explorer,
            d=self.d,
            H=self.H,
            num_states=self.num_states,
            num_actions=self.num_actions_p2,
            beta=self.beta,
            feature_map_args=feature_map_args_p2,
            lambda_reg=lambda_reg
        )
        
        # Player 2 explores against the fixed expert policy of Player 1
        # Note: The 'policy_p2' argument in run_exploration always refers to the opponent.
        D_nu_muE = explorer_p2.run_exploration(
            K=self.K,
            transition_P=self.transition_P,
            policy_p2=self.expert_policy_p1, # Opponent is now Player 1
            initial_state_sampler=self.initial_state_sampler,
            player_idx=2
        )

        # --- Phase 3: Imitation Learning ---

        # Use D_mu_nuE to learn Player 2's policy (hat_nu)
        # We need (state, player_2_action) pairs. The expert action for P2 is `b`.
        dataset_for_nu = [(s, b) for (s, a, b, s_next) in D_mu_nuE]
        
        # Count unique states in dataset for nu
        unique_states_nu = len(set(s for (s, _) in dataset_for_nu))
        
        learner_nu = BehavioralCloningSingleAgent(
            num_states=self.num_states,
            num_actions=self.num_actions_p2,
            feature_map=wrapped_feature_map_p2_learner,
            feature_map_args=feature_map_args_p2
        )
        hat_nu, loss_nu = learner_nu.train(dataset_for_nu, epochs=epochs)

        # Use D_nu_muE to learn Player 1's policy (hat_mu)
        # We need (state, player_1_action) pairs.
        dataset_for_mu = [(s, b) for (s, a, b, s_next) in D_nu_muE]
        
        # Count unique states in dataset for mu
        unique_states_mu = len(set(s for (s, _) in dataset_for_mu))
        
        learner_mu = BehavioralCloningSingleAgent(
            num_states=self.num_states,
            num_actions=self.num_actions_p1,
            feature_map=wrapped_feature_map_p1_learner,
            feature_map_args=feature_map_args_p1
        )
        hat_mu, loss_mu = learner_mu.train(dataset_for_mu, epochs=epochs)
        
        print(f"MAIL Final Loss - Player 1: {loss_mu:.6f}, Player 2: {loss_nu:.6f}")
        print(f"MAIL Unique States - Player 1 (μ): {unique_states_mu}, Player 2 (ν): {unique_states_nu}")

        return (hat_mu, hat_nu), (loss_mu, loss_nu), (unique_states_mu, unique_states_nu)
