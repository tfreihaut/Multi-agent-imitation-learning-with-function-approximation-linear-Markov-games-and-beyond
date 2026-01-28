import numpy as np
from scipy.optimize import linprog
from tqdm import tqdm

class MarkovGameValueIteration:
    """
    Implementation of value iteration for two-player zero-sum Markov Games
    to find Nash equilibrium policies using the maximin Bellman operator.
    """
    
    def __init__(self, num_states, num_actions_player1, num_actions_player2, 
                 rewards, transitions, discount_factor=0.9):
        """
        Initialize the Markov Game.
        
        Args:
            num_states: Number of states in the game
            num_actions_player1: Number of actions available to player 1 (max player)
            num_actions_player2: Number of actions available to player 2 (min player)
            rewards: Array of shape (num_states, num_actions_player1, num_actions_player2)
                    representing immediate rewards for player 1
            transitions: Array of shape (num_states, num_actions_player1, num_actions_player2, num_states)
                        representing transition probabilities P(s'|s,a1,a2)
            discount_factor: Discount factor for future rewards (gamma)
        """
        self.num_states = num_states
        self.num_actions_player1 = num_actions_player1
        self.num_actions_player2 = num_actions_player2
        self.rewards = np.array(rewards)
        self.transitions = np.array(transitions)
        self.discount_factor = discount_factor
        
        # Initialize value function and policies
        self.values = np.zeros(num_states)
        self.policies_player1 = np.ones((num_states, num_actions_player1)) / num_actions_player1
        self.policies_player2 = np.ones((num_states, num_actions_player2)) / num_actions_player2
        
    def solve_state_game(self, state, next_values):
        """
        Solve the matrix game for a specific state using the current value estimates.
        
        Args:
            state: The current state
            next_values: The current value function estimates
            
        Returns:
            value: The updated value for this state
            policy1: Optimal policy for player 1 in this state
            policy2: Optimal policy for player 2 in this state
        """
        # Construct the stage game (matrix game) for this state
        game_matrix = np.zeros((self.num_actions_player1, self.num_actions_player2))
        
        for a1 in range(self.num_actions_player1):
            for a2 in range(self.num_actions_player2):
                # Immediate reward
                immediate_reward = self.rewards[state, a1, a2]
                
                # Expected future reward
                future_reward = 0
                for next_state in range(self.num_states):
                    transition_prob = self.transitions[state, a1, a2, next_state]
                    future_reward += transition_prob * next_values[next_state]
                
                # Q-value = immediate reward + discounted future reward
                game_matrix[a1, a2] = immediate_reward + self.discount_factor * future_reward
        
        # Solve the matrix game using linear programming

        # For player 1 (row player - maximizing)
        # min c^T x subject to A_ub x <= b_ub, A_eq x = b_eq, lb <= x <= ub
        # the elements of x are the value of the state concatenated with the probabilities for each action 
        c1 = np.zeros(self.num_actions_player1 + 1)
        c1[0] = -1  # Maximize the value (negated for minimization of the negation of the value)
        
        # Constraints: v ≤ ∑_j x_j * Q(s,i,j) for all i
        A_ub1 = np.zeros((self.num_actions_player2, self.num_actions_player1 + 1))
        A_ub1[:, 0] = 1  # v term
        for a2 in range(self.num_actions_player2):
            A_ub1[a2, 1:] = -game_matrix[:, a2]
        b_ub1 = np.zeros(self.num_actions_player2)
        
        # Constraint (equality): sum of probabilities = 1
        A_eq1 = np.zeros((1, self.num_actions_player1 + 1))
        A_eq1[0, 1:] = 1
        b_eq1 = np.ones(1)
        
        # Bounds: v is unbounded, probabilities are between 0 and 1
        bounds1 = [(None, None)] + [(0, 1)] * self.num_actions_player1
        
        # Solve the LP for player 1
        result1 = linprog(c1, A_ub=A_ub1, b_ub=b_ub1, A_eq=A_eq1, b_eq=b_eq1, bounds=bounds1, method='highs')
        
        if result1.success:
            value = result1.x[0]
            policy1 = result1.x[1:]
            
            # For player 2 (column player - minimizing)
            # max c^T y subject to A_ub y <= b_ub, A_eq y = b_eq, lb <= y <= ub
            c2 = np.zeros(self.num_actions_player2 + 1)
            c2[0] = 1  # Minimize the value
            
            # Constraints: v ≥ ∑_i y_i * Q(s,i,j) for all j
            A_ub2 = np.zeros((self.num_actions_player1, self.num_actions_player2 + 1))
            A_ub2[:, 0] = -1  # v term
            for a1 in range(self.num_actions_player1):
                A_ub2[a1, 1:] = game_matrix[a1, :]
            b_ub2 = np.zeros(self.num_actions_player1)
            
            # Constraint: sum of probabilities = 1
            A_eq2 = np.zeros((1, self.num_actions_player2 + 1))
            A_eq2[0, 1:] = 1
            b_eq2 = np.ones(1)
            
            # Bounds: v is unbounded, probabilities are between 0 and 1
            bounds2 = [(None, None)] + [(0, 1)] * self.num_actions_player2
            
            # Solve the LP for player 2
            result2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2, bounds=bounds2, method='highs')
            
            if result2.success:
                policy2 = result2.x[1:]
                return value, policy1, policy2
        
        # If LP fails, return uniform policies
        return 0, np.ones(self.num_actions_player1) / self.num_actions_player1, np.ones(self.num_actions_player2) / self.num_actions_player2
    
    def value_iteration(self, max_iterations=1000, tolerance=1e-6):
        """
        Perform value iteration to find optimal policies for both players.
        
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence threshold
            
        Returns:
            values: The state values
            policies_player1: Optimal policies for player 1
            policies_player2: Optimal policies for player 2
        """
        for iteration in tqdm(range(max_iterations)):
            max_diff = 0
            new_values = np.zeros(self.num_states)
            new_policies_player1 = np.zeros((self.num_states, self.num_actions_player1))
            new_policies_player2 = np.zeros((self.num_states, self.num_actions_player2))
            
            # Update each state
            for state in range(self.num_states):
                new_value, new_policy1, new_policy2 = self.solve_state_game(state, self.values)
                
                new_values[state] = new_value
                new_policies_player1[state] = new_policy1
                new_policies_player2[state] = new_policy2
                
                max_diff = max(max_diff, abs(new_value - self.values[state]))
            
            # Update values and policies
            self.values = new_values
            self.policies_player1 = new_policies_player1
            self.policies_player2 = new_policies_player2
            
            # Check for convergence
            if max_diff < tolerance:
                print(f"Converged after {iteration+1} iterations")
                break
                
            if iteration == max_iterations - 1:
                print(f"Did not converge after {max_iterations} iterations")
        
        return self.values, self.policies_player1, self.policies_player2
    
    def get_Q_values(self):
        """
        Compute Q-values for all state-action pairs.
        
        Returns:
            Q: Array of shape (num_states, num_actions_player1, num_actions_player2)
               representing Q-values
        """
        Q = np.zeros((self.num_states, self.num_actions_player1, self.num_actions_player2))
        
        for s in range(self.num_states):
            for a1 in range(self.num_actions_player1):
                for a2 in range(self.num_actions_player2):
                    # Immediate reward
                    Q[s, a1, a2] = self.rewards[s, a1, a2]
                    
                    # Expected future reward
                    for next_s in range(self.num_states):
                        Q[s, a1, a2] += self.discount_factor * self.transitions[s, a1, a2, next_s] * self.values[next_s]
        
        return Q
    
    def verify_nash_equilibrium(self, policy_p1, policy_p2):
        """
        Verify that the computed policies form a Nash equilibrium.
        
        Returns:
            is_nash: Boolean indicating whether policies form a Nash equilibrium
            deviations: Maximum gain from deviating for each player
        """
        Q = self.get_Q_values()
        max_deviation_p1 = 0
        max_deviation_p2 = 0
        
        for s in range(self.num_states):
            # Expected value under current policies
            current_value = 0
            for a1 in range(self.num_actions_player1):
                for a2 in range(self.num_actions_player2):
                    current_value += policy_p1[s, a1] * policy_p2[s, a2] * Q[s, a1, a2]
            
            # Best response for player 1:
            # Assuming Player 2 sticks to their policy (policy_p2), what is the highest 
            # possible payoff I can get by choosing my single best action (a1)?
            best_response_value_p1 = float('-inf')
            for a1 in range(self.num_actions_player1):
                response_value = 0
                for a2 in range(self.num_actions_player2):
                    response_value += policy_p2[s, a2] * Q[s, a1, a2]
                best_response_value_p1 = max(best_response_value_p1, response_value)
            
            # Best response for player 2
            best_response_value_p2 = float('inf')
            for a2 in range(self.num_actions_player2):
                response_value = 0
                for a1 in range(self.num_actions_player1):
                    response_value += policy_p1[s, a1] * Q[s, a1, a2]
                best_response_value_p2 = min(best_response_value_p2, response_value)
            
            # Calculate maximum deviations
            max_deviation_p1 = max(max_deviation_p1, best_response_value_p1 - current_value)
            max_deviation_p2 = max(max_deviation_p2, current_value - best_response_value_p2)
        
        # Small tolerance for floating point errors
        eps = 1e-5
        is_nash = (max_deviation_p1 < eps) and (max_deviation_p2 < eps)
        
        return is_nash, (max_deviation_p1, max_deviation_p2)


# Example usage with a simple grid world game
if __name__ == "__main__":
    # Define a simple 2x2 grid world game with 4 states
    # State 0: Initial state
    # State 1: Goal state for player 1 (+1 reward)
    # State 2: Goal state for player 2 (-1 reward)
    # State 3: Terminal state (0 reward)
    
    num_states = 4
    num_actions_p1 = 2  # Actions: 0=Move Right, 1=Move Down
    num_actions_p2 = 2  # Actions: 0=Block Right, 1=Block Down
    
    # Rewards: player 1 wants to reach state 1, player 2 wants to prevent this
    rewards = np.zeros((num_states, num_actions_p1, num_actions_p2))
    
    # In state 0, player 1 gets reward only if they reach state 1
    # Player 2 can block one direction
    rewards[0, 0, 0] = 0  # Blocked right
    rewards[0, 0, 1] = 1  # Can move right to state 1
    rewards[0, 1, 0] = -1  # Can move down to state 2
    rewards[0, 1, 1] = 0  # Blocked down
    
    # States 1, 2, 3 are terminal states
    
    # Transitions: P(s'|s,a1,a2)
    transitions = np.zeros((num_states, num_actions_p1, num_actions_p2, num_states))
    
    # From state 0
    # If player 1 tries right and player 2 blocks right, stay in state 0
    transitions[0, 0, 0, 0] = 1.0
    # If player 1 tries right and player 2 doesn't block right, go to state 1
    transitions[0, 0, 1, 1] = 1.0
    # If player 1 tries down and player 2 blocks down, stay in state 0
    transitions[0, 1, 1, 0] = 1.0
    # If player 1 tries down and player 2 doesn't block down, go to state 2
    transitions[0, 1, 0, 2] = 1.0
    
    # Terminal states always transition to state 3 (terminal)
    for a1 in range(num_actions_p1):
        for a2 in range(num_actions_p2):
            transitions[1, a1, a2, 3] = 1.0
            transitions[2, a1, a2, 3] = 1.0
            transitions[3, a1, a2, 3] = 1.0
    
    # Create and solve the game
    markov_game = MarkovGameValueIteration(num_states, num_actions_p1, num_actions_p2, 
                                         rewards, transitions, discount_factor=0.9)
    
    values, policies_p1, policies_p2 = markov_game.value_iteration()
    
    print("State values:")
    for s in range(num_states):
        print(f"State {s}: {values[s]:.4f}")
    
    print("\nPlayer 1 policies:")
    for s in range(num_states):
        print(f"State {s}: {policies_p1[s]}")
    
    print("\nPlayer 2 policies:")
    for s in range(num_states):
        print(f"State {s}: {policies_p2[s]}")
    
    # Verify Nash equilibrium
    is_nash, deviations = markov_game.verify_nash_equilibrium(policy_p1=policies_p1, policy_p2=policies_p2)
    print(f"\nPolicies form a Nash equilibrium: {is_nash}")
    print(f"Maximum deviation for player 1: {deviations[0]:.8f}")
    print(f"Maximum deviation for player 2: {deviations[1]:.8f}")