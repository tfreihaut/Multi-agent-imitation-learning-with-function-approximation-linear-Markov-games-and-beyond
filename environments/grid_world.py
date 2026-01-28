# Implementation of the Nash Q Learning algorithm for simple games with two agents
# In this simple version we don't have obstacles
# The implementation is for a zero-sum markov game
# In case of collision the players will remain where they are
# In this implementation we have the reward in only one cell

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from environments.player import Player
import os
from matplotlib import patches
import torch.nn.functional as F
import torch

import pygame


def get_image(path):
    from os import path as os_path

    import pygame

    cwd = os_path.dirname(__file__)
    image = pygame.image.load(cwd + "/" + path)
    sfc = pygame.Surface(image.get_size(), flags=pygame.SRCALPHA)
    sfc.blit(image, (0, 0))
    return sfc

class GridZeroSum:
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "gridworld-v0",
        "is_parallelizable": False,
        "render_fps": 2,
    }
    def __init__(
        self,
        length=2,
        width=2,
        players=[Player(), Player()],
        reward_coordinates=[[0, 1]],
        reward_value=20,
        reward_neutral=0,
        gamma=1,
        random=False,
        reward_dicts = None,
        random_dict = None,
        screen_scaling=5,
        render_mode="rgb_array",
        starting_state: list = None,
        fixed_expert: np.ndarray = None,
        fixed_player: int = None,
    ):
        """
        This class is a representation of the game grid.

        length (int) : horizontal dimension of the grid
        width (int)  : vertical dimension of the grid
        players (list) : list of 2 Player objects.
        reward_coordinates (list) : list of lists of 2 integers giving the coordinates of the reward
        reward_value (int) : value obtained by reaching the reward coordinates
        reward_neutral (int)  : neutral reward for not hitting a wall or reaching the reward coordinates
        gamma (int) : discountig factor, equals 1 if not discounted.
        random (bool): Indicates if the environment is stochastic or not
        reward_dicts (list): list of two dictionaries, one for each player that maps joint states to rewards for that player
        random_dict (dict): dictionary that maps joint states to random transitions
        """
        assert len(reward_coordinates) == 1
        self.length = length
        self.width = width
        self.players = players
        # In this simple scenario players must have the same action space
        assert self.players[0].movements == self.players[1].movements
        self.movements = self.players[0].movements
        self.reward_coordinates = reward_coordinates
        self.reward_value = reward_value
        self.reward_neutral = reward_neutral
        self.gamma = gamma
        self.random = random
        self.joint_player_coordinates = [players[0].position, players[1].position]
        self.all_states = self.joint_states()
        if reward_dicts is None:
            # reward is a list of dictionaries, one for each player that maps joint states to rewards for that player
            self.reward = [self.create_reward_dict(player_idx) for player_idx in range(len(self.players))]
        else:
            self.reward = reward_dicts
        # random dict has the following structures:
        # {
        #     (position, action): action success probability 
        # }
        self.transition_matrix = self.create_transition_matrix(random_dict)

        # New properties for protoRL
        # In case of induced MDP
        self.fixed_expert = fixed_expert
        self.fixed_player = fixed_player
        
        # Reset information
        self.starting_state = starting_state

        self.screen = None
        self.screen_scaling = screen_scaling
        self.render_mode = render_mode
        if self.render_mode == "human":
            self.clock = pygame.time.Clock()
    
    def get_player_0(self):
        return self.players[0]

    def get_player_1(self):
        return self.players[1]

    def create_reward_dict(self, player_idx):
        """
        Creates a reward dictionary for a specific player.
        This method ensures the game is zero-sum by defining rewards based on
        Player 0's outcome and negating it for Player 1.
        """
        reward_dict = {}  # Maps joint state indices to rewards for player_idx
        for state in self.all_states:
            state_idx = self.all_states.index(state)
            
            # 1. Define a base reward from Player 0's (the maximizer's) perspective.
            #    The outcome of the game depends only on Player 0's position relative to the reward.
            base_reward = self.reward_neutral
            if state[0] in self.reward_coordinates:
                base_reward = self.reward_value
                # 2. Assign rewards to players to enforce the zero-sum property.
                if player_idx == 0:
                    # Player 0 (maximizer) gets the base reward.
                    reward_dict[state_idx] = base_reward
                else:
                    # Player 1 (minimizer) gets the negative of the base reward.
                    reward_dict[state_idx] = -base_reward
            elif state[1] in self.reward_coordinates:
                base_reward = self.reward_value
                if player_idx == 0:
                    reward_dict[state_idx] = -base_reward
                else:
                    reward_dict[state_idx] = base_reward
            else:
                reward_dict[state_idx] = self.reward_neutral

        return reward_dict

    def joint_states(self):
        """
        Returns a list of all possible joint states in the game.
        """
        # Agents are only allowed to collide on the reward cell, whether they arrive there at the same time or not
        joint_states = [
            [[i, j], [k, l]]
            for i in range(self.length)
            for j in range(self.width)
            for k in range(self.length)
            for l in range(self.width)
            if [i, j] != [k, l]
        ]
        # note that by construction we don't have the reward state as a joint state
        return joint_states
    
    def _get_transition_outcome(self, state, m0, m1, walls):
        """
        Calculates the single, consistent outcome of a joint move.
        This helper method should be used by both transition and reward functions.
        """
        # 1. Determine tentative next positions, handling walls.
        pos0 = state[0]
        if [m0, state[0]] not in walls:
            pos0 = Player(state[0], movements=self.movements).move(m0)

        pos1 = state[1]
        if [m1, state[1]] not in walls:
            pos1 = Player(state[1], movements=self.movements).move(m1)

        # 2. Apply the absorbing state rule. A player on a reward tile cannot move.
        if state[0] in self.reward_coordinates:
            pos0 = state[0]
        if state[1] in self.reward_coordinates:
            pos1 = state[1]

        # 3. Check for collisions based on the final intended positions.
        is_collision = False
        if pos0 == pos1: # Players try to move to the same cell
            is_collision = True
        elif pos0 == state[1] and pos1 == state[0]: # Players try to swap positions
            is_collision = True

        # 4. Determine the final state based on the outcome.
        if is_collision:
            return state # On collision, players stay in their original positions.
        else:
            return [pos0, pos1] # No collision, the move is successful.

    def create_transition_matrix_old(self, random_dict):
        # transition matrix is a matrix of shape (num_joint_states * num_actions_player1 * num_actions_player2, num_joint_states)
        transition_matrix = np.zeros((len(self.all_states) * len(self.players[0].movements) * len(self.players[1].movements), len(self.all_states)))
        # walls is a list of lists, each list contains the direction and position where you can't move using that direction
        walls = self.identify_walls(self.movements)
        # these are basically the actions
        player0_movements = self.players[0].movements
        player1_movements = self.players[1].movements
        idx = 0
        for state in self.all_states:
            # state is a list of two elements, each representing a player's position
            for m0 in player0_movements:
                for m1 in player1_movements:
                    if [m1, state[1]] in walls or state[1] == self.reward_coordinates[0]:
                        if [m0, state[0]] in walls or state[0] == self.reward_coordinates[0]:
                            # If both players are either hitting a wall or on the reward cell both players stay in place
                            new_state = state
                        else:
                            # Only second player stays in place
                            new_state = [Player(state[0], movements=self.movements).move(m0), state[1]]
                    else:
                        if [m0, state[0]] in walls or state[0] == self.reward_coordinates[0]:
                            # Only first player stays in place
                            new_state = [state[0], Player(state[1], movements=self.movements).move(m1)]
                        else:
                            # Both players can move
                            new_state = [
                                Player(state[0], movements=self.movements).move(m0),
                                Player(state[1], movements=self.movements).move(m1),
                            ]
                    if (
                        new_state[0] == state[1] and new_state[1] == state[0]
                    ) or new_state not in self.all_states:
                        # There is a collision or a swap of positions
                        new_state = state  # Return to previous state
                    if not random_dict:
                        # If no randomness is specified, deterministically transition to the new state
                        transition_matrix[idx, self.all_states.index(new_state)] = 1.0
                    elif state == new_state:
                        # If the state hasn't changed, stay in the same state
                        transition_matrix[idx, self.all_states.index(new_state)] = 1.0
                    elif (tuple(state[0]),m0) in random_dict.keys() and (tuple(state[1]),m1) in random_dict.keys():
                        # If both actions are stochastic, use the random dict to determine the transition probabilities
                        p1 = random_dict[(tuple(state[0]),m0)]
                        p2 = random_dict[(tuple(state[1]),m1)]
                        # 1. Both actions succeed
                        transition_matrix[idx, self.all_states.index(new_state)] = p1 * p2
                        # 2. Both actions fail
                        transition_matrix[idx, self.all_states.index(state)] = (1 - p1) * (1-p2)
                        # 3. First action succeeds, second fails
                        transition_matrix[idx, self.all_states.index([new_state[0], state[1]])] = p1 * (1-p2)
                        # 4. First action fails, second succeeds
                        transition_matrix[idx, self.all_states.index([state[0], new_state[1]])] = (1-p1) * p2

                    elif (tuple(state[0]),m0) in random_dict.keys():
                        # If the first action is stochastic, use the random dict to determine the transition probabilities
                        p1 = random_dict[(tuple(state[0]),m0)]
                        transition_matrix[idx, self.all_states.index(new_state)] = p1
                        if [state[0], new_state[1]] in self.all_states:
                            transition_matrix[idx, self.all_states.index([state[0], new_state[1]])] = 1 - p1
                        else:
                            transition_matrix[idx, self.all_states.index([state[0], state[1]])] = 1 - p1

                    elif (tuple(state[1]),m1) in random_dict.keys():
                        p2 = random_dict[(tuple(state[1]),m1)]
                        transition_matrix[idx, self.all_states.index(new_state)] = p2
                        if [new_state[0], state[1]] in self.all_states:
                            transition_matrix[idx, self.all_states.index([new_state[0], state[1]])] = 1 - p2
                        else:
                            transition_matrix[idx, self.all_states.index([state[0], state[1]])] = 1 - p2
                    else:
                        # If neither action is stochastic, deterministically transition to the new state
                        transition_matrix[idx, self.all_states.index(new_state)] = 1.0

                    idx +=1

        return transition_matrix

    def array_state_representation(self, positions):
        """
        Returns a structured representation of the state given its index.
        The structured representation is a tuple of player positions.
        Returns shape (1, length, width) for PyTorch compatibility (channels first).
        """
        # Init empty grid

        grid = np.array([0] * (self.length * self.width)).reshape(self.length, self.width)
        # Mark positions of both players
        grid[positions[0][0]][positions[0][1]] = 1
        grid[positions[1][0]][positions[1][1]] = 2
        # Return with channels first (1, length, width) for PyTorch
        return np.stack([grid], axis=0).astype(np.int8)

    def render(self, state):
        """
        Renders the grid world with current player positions.
        """

        # Initialize grid representation

        # Compute screen size based on your PNG ratio (≈ 99×86)
        screen_width = int(99 * self.screen_scaling)
        screen_height = int(86 / 99 * screen_width)

        # Initialize pygame surface
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.set_caption("Zero-sum Grid World")
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            elif self.render_mode == "rgb_array":
                self.screen = pygame.Surface((screen_width, screen_height))
        joint_position = self.map_state_idx_to_state(state)
        observation = self.array_state_representation(joint_position)
        

        # Load and scale images
        grid_img = get_image(os.path.join("imgs", "Grid.png"))
        grid_img = pygame.transform.scale(grid_img, (screen_width, screen_height))

        self.screen.blit(grid_img, (0, 0))

        # Compute cell dimensions
        cell_width = screen_width / self.width
        cell_height = screen_height / self.length

        # Determine agent size relative to the cell
        agent_size = int(min(cell_width, cell_height) * 0.7)

        agent_1 = get_image(os.path.join("imgs", "agent1.png"))
        agent_1 = pygame.transform.scale(agent_1, (agent_size, agent_size))

        agent_2 = get_image(os.path.join("imgs", "agent2.png"))
        agent_2 = pygame.transform.scale(agent_2, (agent_size, agent_size))

        # Draw agents centered in their grid cells
        for row in range(self.length):
            for col in range(self.width):
                if observation[0][row, col].item() == 1:
                    x = col * cell_width + (cell_width - agent_1.get_width()) / 2
                    y = row * cell_height + (cell_height - agent_1.get_height()) / 2
                    self.screen.blit(agent_1, (x, y))
                elif observation[0][row, col].item() == 2:
                    x = col * cell_width + (cell_width - agent_2.get_width()) / 2
                    y = row * cell_height + (cell_height - agent_2.get_height()) / 2
                    self.screen.blit(agent_2, (x, y))

        # Render updates
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        observation = np.array(pygame.surfarray.pixels3d(self.screen))
        return np.transpose(observation, axes=(1, 0, 2)) if self.render_mode == "rgb_array" else None

    def create_transition_matrix(self, random_dict):
        # The shape of the transition matrix
        num_states = len(self.all_states)
        num_actions_p1 = len(self.players[0].movements)
        num_actions_p2 = len(self.players[1].movements)
        transition_matrix = np.zeros((num_states * num_actions_p1 * num_actions_p2, num_states))
        
        walls = self.identify_walls(self.movements)
        player0_movements = self.players[0].movements
        player1_movements = self.players[1].movements
        
        idx = 0
        # NOTE: The logic for stochastic transitions (random_dict) has been removed for clarity.
        # You would need to re-integrate it based on the 'final_new_state' if needed.
        if random_dict:
            print("Warning: The provided corrected code does not include the stochastic logic.")

        for state_idx, state in enumerate(self.all_states):
            for m0_idx, m0 in enumerate(player0_movements):
                for m1_idx, m1 in enumerate(player1_movements):
                    
                    # Get the single, correct outcome for this move
                    final_new_state = self._get_transition_outcome(state, m0, m1, walls)
                    
                    # Update the transition matrix
                    final_state_idx = self.all_states.index(final_new_state)
                    transition_matrix[idx, final_state_idx] = 1.0
                    
                    idx += 1

        return transition_matrix

    def identify_walls(self, movements):
        """
        Identify all impossible transitions due to the grid walls.
        """
        walls = []
        for row in range(self.length):
            for col in range(self.width):
                fictitious_player = Player(
                    position=[row, col],
                    movements=movements
                )
                for move in movements:
                    new_pos = fictitious_player.move(move)
                    new_row, new_col = new_pos[0], new_pos[1]
                    
                    # Check both vertical and horizontal boundaries
                    if new_row not in range(self.length) or \
                    new_col not in range(self.width):
                        walls.append([move, fictitious_player.position])

        # remove duplicates
        unique_walls = []
        for wall in walls:
            if wall not in unique_walls:
                unique_walls.append(wall)
        return unique_walls

    def map_state_idx_to_state(self, idx):
        """
        Map a state index to its corresponding state representation
        """
        return self.all_states[idx]
    
    def map_action_idx_to_action(self, idx):
        """
        Map an action index to its corresponding action representation
        """
        return self.movements[idx]

    def compute_reward(
        self, old_state ,state, player_idx
    ):
        """
        Compute the reward obtained by a player for transitioning from its old state to its new state
        """
        # This is needed to make the finite game infinite
        # if old_state[player_idx] == self.reward_coordinates[0]:
        #     return 0
        
        reward_dict = self.reward[player_idx]
        reward = reward_dict[self.all_states.index(state)]

        return reward

    def create_stage_games_old(self):
        """
        Creates the stage game tables which contains the reward obtained by the players for each pair of joint states and joint movements.
        The stage game tables are represented as 3-dimensional tensors.
        """
        joint_states = self.all_states
        walls = self.identify_walls(self.movements)
        # Action space of the players
        player0_movements = self.players[0].movements
        player1_movements = self.players[1].movements

        # stage_games0 will store the rewards for player 0:
        # In particular for [current_state, m0, m1] stores the immediate reward for player 0
        # for taking actions m0 and m1 in current_state
        stage_games0 = np.zeros(
            (
                len(joint_states),
                len(player0_movements),
                len(player1_movements),
            )
        )

        stage_games1 = np.zeros(
            (
                len(joint_states),
                len(player0_movements),
                len(player1_movements),
            )
        )
        for state in joint_states:  # Removed tqdm to avoid clutter when called repeatedly
            for m0 in player0_movements:
                for m1 in player1_movements:
                    if [m1, state[1]] in walls:
                        if [m0, state[0]] not in walls:
                            # Only the first player can move
                            new_state = [Player(state[0], movements=self.movements).move(m0), state[1]]
                        else:
                            new_state = state
                    else:
                        if [m0, state[0]] in walls:
                            # Only the second player can move
                            new_state = [state[0], Player(state[1], movements=self.movements).move(m1)]
                        else:
                            # Both players can move
                            new_state = [
                                Player(state[0], movements=self.movements).move(m0),
                                Player(state[1], movements=self.movements).move(m1),
                            ]
                    
                    if (
                        new_state[0] == state[1] and new_state[1] == state[0]
                    ) or new_state not in joint_states:
                        # There is a collision
                        new_state = state  # Return to previous state

                    reward0 = self.compute_reward(state, new_state, 0)
                    reward1 = self.compute_reward(state, new_state, 1)

                    stage_games0[joint_states.index(state)][player0_movements.index(m0)][player1_movements.index(m1)] = reward0
                    stage_games1[joint_states.index(state)][player0_movements.index(m0)][player1_movements.index(m1)] = reward1

        return stage_games0, stage_games1

    def create_stage_games(self):
        joint_states = self.all_states
        walls = self.identify_walls(self.movements)
        player0_movements = self.players[0].movements
        player1_movements = self.players[1].movements

        stage_games0 = np.zeros((len(joint_states), len(player0_movements), len(player1_movements)))
        stage_games1 = np.zeros((len(joint_states), len(player0_movements), len(player1_movements)))
        
        for state_idx, state in enumerate(joint_states):  # Removed tqdm to avoid clutter when called repeatedly
            for m0_idx, m0 in enumerate(player0_movements):
                for m1_idx, m1 in enumerate(player1_movements):
                    
                    # Get the single, correct outcome for this move
                    final_new_state = self._get_transition_outcome(state, m0, m1, walls)
                    
                    # Calculate rewards based on the actual final state
                    reward0 = self.compute_reward(state, final_new_state, 0)
                    reward1 = self.compute_reward(state, final_new_state, 1)

                    stage_games0[state_idx, m0_idx, m1_idx] = reward0
                    stage_games1[state_idx, m0_idx, m1_idx] = reward1

        return stage_games0, stage_games1
    
    # New functions for ProtoRL
    def step(self, joint_action):
        """
        Executes a joint action for both players and returns the new state and rewards.
        joint_action: list of two actions, one for each player
        """

        if self.fixed_expert is not None:
            # In case of induced MDP, we override with an action sampled from the fixed expert policy
            if isinstance(self.fixed_expert, np.ndarray):
                # If fixed_expert is a policy table, convert it to a callable function
                def fixed_expert_func(s):
                    return np.random.choice(self.fixed_expert.shape[1], p=self.fixed_expert[s])
                fixed_expert_callable = fixed_expert_func
            else:
                # If fixed_expert is already callable, use it directly
                fixed_expert_callable = self.fixed_expert
            
        if self.fixed_player is not None:
            # This means that the joint action given is only a single agent's action for the non-fixed player
            expert_action = fixed_expert_callable(self.all_states.index(self.joint_player_coordinates))
            # Translate to movement
            expert_movement_fixed = self.players[self.fixed_player].movements[expert_action]
            # get other player index
            other_idx = 1 - self.fixed_player
            # translate action to movement
            action_played = self.players[other_idx].movements[joint_action]

            joint_action = [expert_movement_fixed if i == self.fixed_player else action_played for i in range(2)]
            
        
        m0 = joint_action[0]
        m1 = joint_action[1]
        # 
        current_state = self.joint_player_coordinates
        self.get_player_0().position = current_state[0]
        self.get_player_1().position = current_state[1]
        new_state = self._get_transition_outcome(current_state, m0, m1, self.identify_walls(self.movements))
        
        self.joint_player_coordinates = new_state
        
        rewards = [
            self.compute_reward(current_state, new_state, player_idx)
            for player_idx in range(len(self.players))
        ]
        self.observation = self.array_state_representation(self.joint_player_coordinates)
        
        if self.render_mode is not None:
            state_idx = self.all_states.index(self.joint_player_coordinates)
            self.render(state=state_idx)
        done = False
        if np.any(np.array(rewards) > 0):
            done = True
        rewards_other_player = rewards[other_idx]
        return self.observation, rewards_other_player, expert_action, done

    def reset(self):
        self.observation = self.array_state_representation(self.starting_state)
        self.joint_player_coordinates = self.starting_state
        return self.observation

def visualize_gridzero_board(grid_game, current_state=None, title="Grid World Board", filename="grid_state.png", output_dir="images"):
    """
    Visualizes the state of a GridZeroSum game with a board-like appearance
    and saves it as a PNG image to a specified folder.

    - Interprets coordinates as (row, column).
    - Grid lines are drawn to form cells, like a Tic-Tac-Toe board.
    - The reward location is marked with a red cross.
    - Player 1 is a blue dot.
    - Player 2 is a green dot.

    Args:
        grid_game (GridZeroSum): The game instance containing grid dimensions and rewards.
        current_state (list, optional): A list of two positions, e.g., [[row1, col1], [row2, col2]].
                                        If None, uses the initial positions of the players.
        title (str, optional): The title for the plot.
        filename (str, optional): The name of the PNG file to save.
        output_dir (str, optional): The directory where the image will be saved.
                                    It will be created if it doesn't exist.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a figure and axes for the plot
    fig, ax = plt.subplots(figsize=(6, 6))

    # Get grid dimensions
    length = grid_game.length # Interpreted as num_rows
    width = grid_game.width   # Interpreted as num_cols
    
    # --- Grid and Axis Setup ---
    
    # Set plot limits to encompass the grid cells (x for columns, y for rows)
    ax.set_xlim(-0.5, width - 0.5)  # x-axis for columns
    ax.set_ylim(-0.5, length - 0.5) # y-axis for rows

    # Draw the cell borders (x for columns, y for rows)
    for x_line in range(width + 1):
        ax.axvline(x_line - 0.5, color='black', linewidth=2)
    for y_line in range(length + 1):
        ax.axhline(y_line - 0.5, color='black', linewidth=2)

    # Remove the default axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Invert the y-axis so that (0,0) is at the top-left corner
    ax.invert_yaxis()
    
    # Make the grid cells perfectly square
    ax.set_aspect('equal', adjustable='box')

    # --- Plot the Game Elements (Coordinates swapped to (col, row)) ---

    # 1. Plot the reward location with a red cross
    reward_coord = grid_game.reward_coordinates[0]
    # plot(column, row)
    ax.plot(reward_coord[1], reward_coord[0], marker='X', color='red', markersize=25,
            linestyle='None', label='Reward', markeredgewidth=4)

    # 2. Determine and plot player positions
    if current_state:
        p1_pos, p2_pos = current_state[0], current_state[1]
    else:
        p1_pos = grid_game.get_player_0().position
        p2_pos = grid_game.get_player_1().position

    # Plot Player 1 (blue dot)
    # plot(column, row)
    ax.plot(p1_pos[1], p1_pos[0], marker='o', color='blue', markersize=22,
            linestyle='None', label='Player 1')

    # Plot Player 2 (green dot)
    # plot(column, row)
    ax.plot(p2_pos[1], p2_pos[0], marker='o', color='green', markersize=22,
            linestyle='None', label='Player 2')

    # --- Final Touches ---
    ax.set_title(title, fontsize=16)
    ax.legend()
    
    # Construct the full path for saving the image
    full_path = os.path.join(output_dir, filename)
    
    # Save the figure as a PNG
    plt.savefig(full_path, bbox_inches='tight', dpi=100)
    
    # Close the plot to free up memory and prevent it from appearing on screen
    plt.close(fig)
    print(f"Plot saved to: {full_path}")

def state_to_image_array(grid_game, current_state):
    """
    Converts a specific game state into an RGB image NumPy array.

    This function is optimized for machine learning use cases, generating a clean
    visual representation of the board without titles, legends, or axes.

    Args:
        grid_game (GridZeroSum): The game instance containing grid dimensions and rewards.
        current_state (int): state index in the grid.

    Returns:
        numpy.ndarray: The game state as an RGB NumPy array of shape (height, width, 3),
                       with pixel values in the range [0, 255].
    """
    # get the actual state from the index
    current_state = grid_game.map_state_idx_to_state(current_state)
    # Create a figure and axes for the plot
    fig, ax = plt.subplots(figsize=(6, 6))
    # Get grid dimensions
    length = grid_game.length
    width = grid_game.width
    
    # --- Grid and Axis Setup ---
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, length - 0.5)
    for x_line in range(width + 1):
        ax.axvline(x_line - 0.5, color='black', linewidth=2)
    for y_line in range(length + 1):
        ax.axhline(y_line - 0.5, color='black', linewidth=2)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')

    # --- Plot the Game Elements ---
    # reward_coord = grid_game.reward_coordinates[0]
    # ax.plot(reward_coord[1], reward_coord[0], marker='X', color='red', markersize=25,
    #         linestyle='None', markeredgewidth=4)

    p1_pos, p2_pos = current_state[0], current_state[1]
    ax.plot(p1_pos[1], p1_pos[0], marker='o', color='blue', markersize=22, linestyle='None')
    ax.plot(p2_pos[1], p2_pos[0], marker='o', color='green', markersize=22, linestyle='None')

    # --- Convert plot to NumPy array ---
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    
    # Calculate actual dimensions from buffer size
    # Buffer size = width * height * 4 (RGBA channels)
    buffer_size = len(img_array)
    total_pixels = buffer_size // 4
    
    # Get reported canvas dimensions (might differ on retina displays)
    width, height = fig.canvas.get_width_height()
    
    # Check if we need to account for device pixel ratio (retina display)
    if width * height != total_pixels:
        # Recalculate assuming square aspect ratio
        actual_dim = int(np.sqrt(total_pixels))
        img_array = img_array.reshape(actual_dim, actual_dim, 4)
    else:
        img_array = img_array.reshape(height, width, 4)
    
    plt.close(fig) # IMPORTANT: Frees up memory
    
    # Return the RGB channels, slicing off the alpha (transparency) channel
    return img_array[:, :, :3]


def visualize_joint_policy_from_state(
    grid_game,
    policy_p1,
    policy_p2,
    current_state,
    title="Nash Equilibrium Joint Policy Visualization",
    filename="joint_policy_state.png",
    output_dir="images"
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Setup Plot ---
    fig, ax = plt.subplots(figsize=(8, 8))
    length, width = grid_game.length, grid_game.width
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, length - 0.5)
    ax.set_xticks(np.arange(-0.5, width, 1))
    ax.set_yticks(np.arange(-0.5, length, 1))
    # put grid behind everything
    ax.grid(True, which='both', color='black', linewidth=2, zorder=0)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')

    # --- Colors and Layout ---
    player_colors = ['#0077BE', '#009E73']  # Blue (P1), Green (P2)

    # --- Robust State Lookup ---
    try:
        p1_pos = list(map(int, current_state[0]))
        p2_pos = list(map(int, current_state[1]))
        state_for_lookup = [p1_pos, p2_pos]
        state_idx = grid_game.all_states.index(state_for_lookup)
    except (ValueError, IndexError):
        print(f"ERROR: The state {current_state} was not found in the game's list of valid states.")
        plt.close(fig)
        return

    # Plot reward and players (set zorder so arrows can be above/below as desired)
    reward_coord = grid_game.reward_coordinates[0]
    ax.scatter(reward_coord[1], reward_coord[0], marker='X', color='#D55E00', s=450, zorder=6, label=f'Reward at {reward_coord}')
    # players: put marker zorder a bit lower than arrow zorder we'll use below
    ax.scatter(p1_pos[1], p1_pos[0], s=700, color=player_colors[0], zorder=8, label=f'Player 1 at {p1_pos}')
    ax.scatter(p2_pos[1], p2_pos[0], s=700, color=player_colors[1], zorder=8, label=f'Player 2 at {p2_pos}')

    # --- Helper to draw policy arrows for one player ---
    def draw_policy_for_player(pos, policy_probs, movements, color, player_name='Player'):
        r, c = pos
        arrows_drawn = 0

        # defensive: allow different action naming
        movs = [m.upper() for m in movements]

        if len(policy_probs) != len(movs):
            print(f"Warning: policy length ({len(policy_probs)}) != movements length ({len(movs)}) for {player_name}.")

        for i, action in enumerate(movs):
            # protect index mismatch
            if i >= len(policy_probs):
                break
            prob = float(policy_probs[i])
            if prob < 1e-3:
                continue

            # compute destination cell (row,col)
            if action in ("UP", "NORTH"):
                r2, c2 = r - 1, c
            elif action in ("DOWN", "SOUTH"):
                r2, c2 = r + 1, c
            elif action in ("LEFT", "WEST"):
                r2, c2 = r, c - 1
            elif action in ("RIGHT", "EAST"):
                r2, c2 = r, c + 1
            elif action in ("STAY", "WAIT", "NOOP"):
                r2, c2 = r, c
            else:
                # unknown action (skip)
                # print(f"Unknown action '{action}' for {player_name} -- skipping.")
                continue

            # skip invalid move outside grid
            if not (0 <= r2 < length and 0 <= c2 < width):
                continue

            # STAY: draw a small loop / circle
            if action in ("STAY", "WAIT", "NOOP"):
                radius = 0.2
                circ = patches.Circle((c, r), radius=radius, fill=False,
                                      linewidth=max(1.2, 1 + 4 * prob), zorder=9)
                ax.add_patch(circ)
                ax.text(c + 0.3, r, f"{prob:.2f}",
                        fontsize=10, fontweight='bold', ha='left', va='center',
                        bbox=dict(facecolor='white', alpha=0.8, pad=0.12), zorder=10)
                arrows_drawn += 1
                continue

            # Draw arrow from center (c,r) -> center (c2,r2)
            start = (c, r)
            end = (c2, r2)
            # linewidth and head scaling by prob to make higher probs pop
            arrow_lw = max(1.0, 1 + 4 * prob)
            head_scale = 10 + 15 * prob

            ax.annotate("",
                        xy=end, xytext=start,
                        arrowprops=dict(arrowstyle="->",
                                        color=color,
                                        linewidth=arrow_lw,
                                        mutation_scale=head_scale),
                        zorder=11, clip_on=False)

            # Put the label slightly closer to the destination (60% of the way)
            label_x = c + 0.6 * (c2 - c)
            label_y = r + 0.6 * (r2 - r)
            ax.text(label_x, label_y, f"{prob:.2f}",
                    fontsize=10, fontweight='bold', ha='center', va='center',
                    color='black', bbox=dict(facecolor='white', alpha=0.85, pad=0.12), zorder=12)

            arrows_drawn += 1

        if arrows_drawn == 0:
            print(f"[debug] No arrows drawn for {player_name} at pos {pos}. Check movement names and nonzero probs.")

    # Draw for both players
    p1_policy = policy_p1[state_idx]
    p1_movements = grid_game.get_player_0().movements
    draw_policy_for_player(p1_pos, p1_policy, p1_movements, player_colors[0], player_name='Player 1')

    p2_policy = policy_p2[state_idx]
    p2_movements = grid_game.get_player_1().movements
    draw_policy_for_player(p2_pos, p2_policy, p2_movements, player_colors[1], player_name='Player 2')

    # Final touches
    ax.set_title(title, fontsize=16, pad=10)
    ax.legend(loc='upper right')
    full_path = os.path.join(output_dir, filename)
    plt.show()
    # fig.savefig(full_path, bbox_inches='tight')
    # print(f"Joint policy visualization saved to: {full_path}")

def state_action_feature_map_tabular_case(states, actions, num_states, num_actions):
    """
    Creates a feature vector by concatenating the one-hot encodings of state and action.
    This function is designed to be vectorized.

    Args:
        states (torch.Tensor): A tensor of state indices.
        actions (torch.Tensor): A tensor of action indices.
        num_states (int): The total number of states.
        num_actions (int): The total number of actions.

    Returns:
        torch.Tensor: A tensor of feature vectors.
    """
    # Ensure inputs are long tensors for one-hot encoding
    s_one_hot = F.one_hot(states.long(), num_classes=num_states)
    a_one_hot = F.one_hot(actions.long(), num_classes=num_actions)

    # Concatenate along the last dimension to form the feature vector φ(x, a)
    features = torch.cat([s_one_hot, a_one_hot], dim=-1).float()
    return features

def state_action_feature_map_interaction(states, actions, num_states, num_actions):
    """
    Creates a feature vector where each (state, action) pair has a unique dimension.
    This is a one-hot encoding of the flattened (state, action) index.
    """
    # Handle both scalar and tensor inputs
    if isinstance(states, (int, np.integer)):
        states = torch.tensor([states])
        actions = torch.tensor([actions])
        single_input = True
    else:
        single_input = False
    
    # Calculate a unique index for each (state, action) pair
    # E.g., for state x and action a, the index is x * num_actions + a
    indices = states * num_actions + actions
    
    # Total number of unique pairs
    feature_dim = num_states * num_actions
    
    # Create the one-hot encoded features
    features = F.one_hot(indices.long(), num_classes=feature_dim).float()
    
    # If input was a single scalar, return a 1D tensor
    if single_input:
        return features.squeeze(0)
    else:
        return features

def state_action_feature_map_splitted_case(states, actions, num_states, num_actions, grid_game):
    """
    Creates a feature representation where:
    - First length*width positions: one-hot encoding for player 1's position
    - Next length*width positions: one-hot encoding for player 2's position  
    - Last num_actions positions: one-hot encoding for the action
    
    Args:
        states: tensor of joint state indices 
        actions: tensor of action indices
        num_states: total number of joint states
        num_actions: number of possible actions
        grid_game: GridZeroSum instance to get length/width
        
    Returns:
        Feature tensor of shape (..., 2*length*width + num_actions)
    """
    batch_shape = states.shape
    length = grid_game.length
    width = grid_game.width
    
    # Total feature dimension: 2 * (length * width) + num_actions
    feature_dim = 2 * length * width + num_actions
    
    # Initialize feature tensor
    features = torch.zeros(batch_shape + (feature_dim,))
    
    # Convert state indices to joint states and extract positions
    states_flat = states.flatten()
    actions_flat = actions.flatten()
    
    for i, (state_idx, action_idx) in enumerate(zip(states_flat, actions_flat)):
        # Get the joint state from the index
        joint_state = grid_game.all_states[state_idx.item()]
        player1_pos = joint_state[0]  # [row, col]
        player2_pos = joint_state[1]  # [row, col]
        
        # Convert 2D positions to flat indices
        player1_flat_pos = player1_pos[0] * width + player1_pos[1]
        player2_flat_pos = player2_pos[0] * width + player2_pos[1]
        
        # Create flat index for the current batch element
        flat_idx = i
        
        # Set player 1 position (first length*width positions)
        features.view(-1, feature_dim)[flat_idx, player1_flat_pos] = 1.0
        
        # Set player 2 position (next length*width positions)
        features.view(-1, feature_dim)[flat_idx, length * width + player2_flat_pos] = 1.0
        
        # Set action (last num_actions positions)
        features.view(-1, feature_dim)[flat_idx, 2 * length * width + action_idx.item()] = 1.0
    
    return features

def state_action_feature_map_splitted_case_interaction(states, actions, num_states, num_actions, grid_game, player_idx):
    """
    Creates a feature representation with separate interaction terms for each player.
    - Part 1: One-hot encoding of the (Player 1 Position, Action) pair.
    - Part 2: One-hot encoding of the (Player 2 Position, Action) pair.
    
    Args:
        states (torch.Tensor): Tensor of joint state indices.
        actions (torch.Tensor): Tensor of action indices.
        num_states (int): Total number of joint states (not directly used, but part of the signature).
        num_actions (int): Number of possible actions for the acting player.
        grid_game (GridZeroSum): GridZeroSum instance to get grid dimensions.
        
    Returns:
        Feature tensor of shape (..., 2 * length * width * num_actions)
    """
    batch_shape = states.shape
    length = grid_game.length
    width = grid_game.width
    
    # The size of one interaction block (e.g., P1_pos x Actions)
    interaction_block_size = length * width * num_actions
    
    # Total feature dimension is two such blocks concatenated
    feature_dim = 2 * interaction_block_size
    
    # Initialize the final feature tensor
    features = torch.zeros(batch_shape + (feature_dim,), device=states.device)
    
    # Flatten inputs for easier iteration
    states_flat = states.flatten()
    actions_flat = actions.flatten()
    
    # Get all joint states from the game for mapping
    all_states = grid_game.all_states
    
    for i, (state_idx, action_idx) in enumerate(zip(states_flat, actions_flat)):
        # 1. Get player positions from the state index
        joint_state = all_states[state_idx.item()]
        player1_pos = joint_state[0]  # [row, col]
        player2_pos = joint_state[1]  # [row, col]
        
        # 2. Convert 2D player positions to flat 1D indices
        player1_flat_pos = player1_pos[0] * width + player1_pos[1]
        player2_flat_pos = player2_pos[0] * width + player2_pos[1]
        
        # 3. Calculate the unique index for each interaction term
        # Interaction index for (Player 1 Position, Action)
        interaction_idx1 = player1_flat_pos * num_actions + action_idx.item()
        
        # Interaction index for (Player 2 Position, Action)
        interaction_idx2 = player2_flat_pos * num_actions + action_idx.item()
        
        # 4. Set the corresponding bits in the feature vector
        # The .view(-1, feature_dim) allows us to use a single flat index `i`
        # for any batch shape.
        
        # Set the bit in the first block (for Player 1)
        features.view(-1, feature_dim)[i, interaction_idx1] = 1.0
        
        # Set the bit in the second block (for Player 2), making sure to add the offset
        features.view(-1, feature_dim)[i, interaction_block_size + interaction_idx2] = 1.0
    
    return features

def state_action_feature_map_player_centric(states, actions, num_states, num_actions, grid_game, player_idx):
    """
    Creates a player-centric feature representation with interaction terms.

    For the acting player (`player_idx`):
    - A one-hot encoding of their (Position, Action) pair is created.

    For the other player (the one not acting):
    - All features corresponding to their (Position, *) pairs are activated,
      where * represents every possible action.
    
    Args:
        states (torch.Tensor): Tensor of joint state indices.
        actions (torch.Tensor): Tensor of action indices for the acting player.
        num_states (int): Total number of joint states.
        num_actions (int): Number of possible actions for the acting player.
        grid_game (GridZeroSum): GridZeroSum instance to get grid dimensions.
        player_idx (int): The index (0 or 1) of the player taking the action.
        
    Returns:
        Feature tensor of shape (..., 2 * length * width * num_actions)
    """
    batch_shape = states.shape
    length = grid_game.length
    width = grid_game.width
    
    # The size of one player's interaction block (e.g., P1_pos x Actions)
    interaction_block_size = length * width * num_actions
    
    # Total feature dimension is the concatenation of both player blocks
    feature_dim = 2 * interaction_block_size
    
    # Initialize the final feature tensor
    features = torch.zeros(batch_shape + (feature_dim,), device=states.device)
    
    # Flatten inputs for efficient iteration
    states_flat = states.flatten()
    actions_flat = actions.flatten()
    
    all_states = grid_game.all_states
    
    for i, (state_idx, action_idx) in enumerate(zip(states_flat, actions_flat)):
        # 1. Get player positions from the state index
        joint_state = all_states[state_idx.item()]
        player1_pos_2d = joint_state[0]
        player2_pos_2d = joint_state[1]
        
        # 2. Convert 2D positions to flat 1D indices
        player1_flat_pos = player1_pos_2d[0] * width + player1_pos_2d[1]
        player2_flat_pos = player2_pos_2d[0] * width + player2_pos_2d[1]

        # 3. Determine who is the acting and who is the other player
        if player_idx == 0:
            acting_player_pos = player1_flat_pos
            other_player_pos = player2_flat_pos
        else: # player_idx == 1
            acting_player_pos = player2_flat_pos
            other_player_pos = player1_flat_pos
            
        # 4. Set features for the ACTING player
        # A single bit is activated for their specific (position, action) pair.
        acting_interaction_idx = acting_player_pos * num_actions + action_idx.item()
        
        # The offset is based on which player is acting
        acting_player_offset = player_idx * interaction_block_size
        features.view(-1, feature_dim)[i, acting_player_offset + acting_interaction_idx] = 1.0

        # 5. Set features for the OTHER player
        # All bits corresponding to their position and ANY action are activated.
        other_player_base_idx = other_player_pos * num_actions
        
        # The offset is based on which player is the "other" one
        other_player_idx = 1 - player_idx
        other_player_offset = other_player_idx * interaction_block_size
        
        for j in range(num_actions):
            other_interaction_idx = other_player_base_idx + j
            features.view(-1, feature_dim)[i, other_player_offset + other_interaction_idx] = 1.0
            
    return features

def state_action_feature_map_hybrid(states, actions, num_states, num_actions, grid_game, player_idx):
    """
    Creates an effective and compact feature vector by combining absolute and relative interactions.

    The vector is a concatenation of two parts:
    1. Absolute Interaction: A one-hot encoding of the (acting player's position, action) pair.
    2. Relative Interaction: A one-hot encoding of the (opponent's relative position, action) pair.

    This captures both location-specific and opponent-relative strategies.
    
    Args:
        states (torch.Tensor): Tensor of joint state indices.
        actions (torch.Tensor): Tensor of action indices for the acting player.
        ... (rest of the args are the same)
        
    Returns:
        A compact and expressive feature tensor.
    """
    batch_shape = states.shape
    length = grid_game.length
    width = grid_game.width
    grid_size = length * width
    
    # --- Calculate Feature Dimensions ---
    # Part 1: Acting player's absolute position x action
    absolute_block_size = grid_size * num_actions
    
    # Part 2: Relative position of opponent x action
    # Delta can range from -(dim-1) to +(dim-1), so there are 2*dim - 1 possible values
    num_relative_rows = 2 * length - 1
    num_relative_cols = 2 * width - 1
    num_relative_positions = num_relative_rows * num_relative_cols
    relative_block_size = num_relative_positions * num_actions

    feature_dim = absolute_block_size + relative_block_size
    
    # --- Vectorized Feature Creation ---
    states_flat = states.flatten()
    actions_flat = actions.flatten()
    
    # Get player positions
    all_states = grid_game.all_states
    joint_states_list = [all_states[idx.item()] for idx in states_flat]
    p1_pos = torch.tensor([s[0] for s in joint_states_list], device=states.device)
    p2_pos = torch.tensor([s[1] for s in joint_states_list], device=states.device)

    # Determine acting and other player positions
    acting_pos = p1_pos if player_idx == 0 else p2_pos
    other_pos = p2_pos if player_idx == 0 else p1_pos
    
    # --- 1. Absolute Interaction Part ---
    acting_flat_pos = acting_pos[:, 0] * width + acting_pos[:, 1]
    absolute_indices = acting_flat_pos * num_actions + actions_flat
    phi_absolute = F.one_hot(absolute_indices, num_classes=absolute_block_size).float()
    
    # --- 2. Relative Interaction Part ---
    delta_pos = acting_pos - other_pos
    # Shift deltas to be non-negative for indexing: e.g., range [-2, 2] -> [0, 4]
    relative_row_indices = delta_pos[:, 0] + (length - 1)
    relative_col_indices = delta_pos[:, 1] + (width - 1)
    
    # Flatten the 2D relative position into a 1D index
    relative_flat_pos = relative_row_indices * num_relative_cols + relative_col_indices
    relative_indices = relative_flat_pos * num_actions + actions_flat
    phi_relative = F.one_hot(relative_indices, num_classes=relative_block_size).float()
    
    # --- 3. Concatenate and Reshape ---
    all_features_flat = torch.cat([phi_absolute, phi_relative], dim=1)
    features = all_features_flat.view(batch_shape + (feature_dim,))
    
    return features

def state_action_feature_map_tiled_relational(states, actions, num_states, num_actions, grid_game, player_idx):
    """
    Creates a compact, expressive, and generalizable feature vector based on
    tiled relational concepts.

    The vector is a concatenation of interactions between the action and high-level concepts.
    """
    # Handle both scalar and tensor inputs
    if isinstance(states, (int, np.integer)):
        states = torch.tensor([states])
        actions = torch.tensor([actions])
        single_input = True
    else:
        single_input = False
        
    batch_shape = states.shape
    length = grid_game.length
    width = grid_game.width
    
    num_directions = 8  # N, S, E, W, NE, NW, SE, SW
    num_proximity = 2  # Opponent adjacent, Reward adjacent
    num_on_reward = 1
    num_in_corner = 1
    
    concepts_per_action = num_directions + num_directions + num_proximity + num_on_reward + num_in_corner
    feature_dim = concepts_per_action * num_actions
    
    states_flat = states.flatten()
    actions_flat = actions.flatten()
    batch_size = len(states_flat)
    
    features = torch.zeros((batch_size, feature_dim), device=states.device)

    all_states = grid_game.all_states
    joint_states_list = [all_states[idx.item()] for idx in states_flat]
    p1_pos = torch.tensor([s[0] for s in joint_states_list], device=states.device)
    p2_pos = torch.tensor([s[1] for s in joint_states_list], device=states.device)
    reward_pos = torch.tensor(grid_game.reward_coordinates[0], device=states.device).expand_as(p1_pos)

    acting_pos = p1_pos if player_idx == 0 else p2_pos
    other_pos = p2_pos if player_idx == 0 else p1_pos
    
    delta_opp = other_pos - acting_pos
    delta_rew = reward_pos - acting_pos
    
    def get_direction_idx(delta):
        dr, dc = delta[:, 0], delta[:, 1]
        # N=0, NE=1, E=2, SE=3, S=4, SW=5, W=6, NW=7
        idx = torch.full_like(dr, -1)
        idx[dr < 0] = 0  # North
        idx[dr > 0] = 4  # South
        idx[dc > 0] += 2 # East
        idx[dc < 0] += 6 # West
        idx[torch.logical_and(dr != 0, dc != 0)] %= 8 # Handle diagonals
        idx[torch.logical_and(dr == 0, dc > 0)] = 2
        idx[torch.logical_and(dr == 0, dc < 0)] = 6
        return idx

    dir_opp_idx = get_direction_idx(delta_opp)
    dir_rew_idx = get_direction_idx(delta_rew)

    base_offset = 0
    # Activate opponent direction features
    valid_opp = dir_opp_idx != -1
    if valid_opp.any():
        action_offset = actions_flat[valid_opp] * concepts_per_action
        concept_offset = base_offset + dir_opp_idx[valid_opp]
        features[valid_opp, action_offset + concept_offset] = 1.0
    
    base_offset += num_directions
    # Activate reward direction features
    valid_rew = dir_rew_idx != -1
    if valid_rew.any():
        action_offset = actions_flat[valid_rew] * concepts_per_action
        concept_offset = base_offset + dir_rew_idx[valid_rew]
        features[valid_rew, action_offset + concept_offset] = 1.0

    is_opp_adj = (torch.abs(delta_opp[:, 0]) <= 1) & (torch.abs(delta_opp[:, 1]) <= 1) & (dir_opp_idx != -1)
    is_rew_adj = (torch.abs(delta_rew[:, 0]) <= 1) & (torch.abs(delta_rew[:, 1]) <= 1) & (dir_rew_idx != -1)
    is_on_reward = (acting_pos[:, 0] == reward_pos[:, 0]) & (acting_pos[:, 1] == reward_pos[:, 1])
    
    r, c = acting_pos[:, 0], acting_pos[:, 1]
    is_in_corner = ((r == 0) | (r == length - 1)) & ((c == 0) | (c == width - 1))

    base_offset += num_directions
    # Activate features
    if is_opp_adj.any():
        features[is_opp_adj, actions_flat[is_opp_adj] * concepts_per_action + base_offset] = 1.0
    if is_rew_adj.any():
        features[is_rew_adj, actions_flat[is_rew_adj] * concepts_per_action + base_offset + 1] = 1.0
    if is_on_reward.any():
        features[is_on_reward, actions_flat[is_on_reward] * concepts_per_action + base_offset + 2] = 1.0
    if is_in_corner.any():
        features[is_in_corner, actions_flat[is_in_corner] * concepts_per_action + base_offset + 3] = 1.0

    result = features.view(batch_shape + (feature_dim,))
    
    # If input was a single scalar, return a 1D tensor
    if single_input:
        return result.squeeze(0)
    else:
        return result

def state_action_feature_map_svd(states, actions, num_states, num_actions, grid_game, rank, player_idx, experts, reward_pos) -> torch.Tensor:
    """
    Computes a low-rank SVD-based feature map for a specified player.

    This function isolates a player's transition dynamics, flattens it into a 
    (next_state, state-action) matrix, performs a singular value decomposition (SVD),
    and uses the right singular vectors to construct a feature map.
    """
    # Handle scalar inputs by converting to tensors
    if not isinstance(states, torch.Tensor):
        states = torch.tensor([states], dtype=torch.long)
        actions = torch.tensor([actions], dtype=torch.long)
        scalar_input = True
    else:
        scalar_input = False
    
    # 1. Get the full transition dynamics tensor T(s', a1, a2, s)
    _, transitions, _ = convert_gridzero_to_markov_game(grid_game)
    if isinstance(transitions, np.ndarray):
        transitions = torch.from_numpy(transitions).float()
    num_states = transitions.shape[0]
    # 2. Derive the induced transition matrix by marginalizing the other player's actions
    if player_idx == 0:
        # For player 0 we compute the transitions induced by the expert policy of the second player
        # Resulting shape: (s', a1, s)
        expert_policy_p2 = experts[1]  # Expert policy for player 2
        if isinstance(expert_policy_p2, np.ndarray):
            expert_policy_p2 = torch.from_numpy(expert_policy_p2).float()
        # transitions shape: (s', a1, a2, s)
        # expert_policy_p2 shape: (s, a2)
        # We need to weight transitions by the policy at the *source* state s (last dimension)
        # Reshape expert_policy_p2 to (1, 1, a2, s) for broadcasting
        policy_weights = expert_policy_p2.T.unsqueeze(0).unsqueeze(0)  # (1, 1, a2, s)
        induced_transitions = (transitions * policy_weights).sum(dim=2)  # sum over a2
        num_actions = induced_transitions.shape[1]
    elif player_idx == 1:
        # For player 1 we compute the transitions induced by the expert policy of the first player
        # Resulting shape: (s', a2, s)
        expert_policy_p1 = experts[0]  # Expert policy for player 1
        if isinstance(expert_policy_p1, np.ndarray):
            expert_policy_p1 = torch.from_numpy(expert_policy_p1).float()
        # transitions shape: (s', a1, a2, s)
        # expert_policy_p1 shape: (s, a1)
        # We need to weight transitions by the policy at the *source* state s (last dimension)
        # Reshape expert_policy_p1 to (1, a1, 1, s) for broadcasting
        policy_weights = expert_policy_p1.T.unsqueeze(0).unsqueeze(2)  # (1, a1, 1, s)
        induced_transitions = (transitions * policy_weights).sum(dim=1)  # sum over a1
        num_actions = induced_transitions.shape[1]
    else:
        raise ValueError("player_idx must be 0 or 1")

    # 3. Reshape the matrix for SVD
    # We want a 2D matrix of shape (s', s * a)
    # Current shape is (s', a, s). Permute to (s', s, a) then reshape.
    matrix_to_decompose = induced_transitions.permute(0, 2, 1).reshape(num_states, -1)

    # 4. Perform SVD and extract the right singular vectors (Vh)
    # Vh contains the feature vectors for the columns (the state-action pairs) in its rows
    # M = U @ S @ Vh
    _, _, Vh = torch.linalg.svd(matrix_to_decompose, full_matrices=True)

    # 5. Select the top 'rank' features (the first 'rank' rows of Vh)
    # Vh_r has shape (rank, num_states * num_actions)
    Vh_r = Vh[:rank, :]

    # 6. Reshape the features into an indexable map
    # We want a final tensor of shape (num_states, num_actions, rank)
    # First, reshape to (rank, num_states, num_actions)
    feature_tensor_transposed = Vh_r.reshape(rank, num_states, num_actions)
    # Then, permute the dimensions to get the desired final shape
    feature_map = feature_tensor_transposed.permute(1, 2, 0).contiguous()

    # compute features for the given states and actions
    batch_shape = states.shape
    states_flat = states.flatten()
    actions_flat = actions.flatten()
    feature_vectors = torch.zeros((len(states_flat), rank), device=states.device)
    for i, (state_idx, action_idx) in enumerate(zip(states_flat, actions_flat)):
        feature_vectors[i] = feature_map[state_idx.item(), action_idx.item(), :]
        # concatenate boolean indicator to indicate if on reward
        joint_state = grid_game.all_states[state_idx.item()]
        player_pos = joint_state[player_idx]
        if player_pos == reward_pos:
            feature_vectors[i, -1] = 1.0  # set last feature to 1 if on reward

    feature_map = feature_vectors.view(batch_shape + (rank,))

    # If input was scalar, return squeezed output
    if scalar_input:
        feature_map = feature_map.squeeze(0)

    return feature_map

def convert_gridzero_to_markov_game(grid_game: GridZeroSum):
    """
    Converts a GridZeroSum game instance into the format required by the
    MarkovGameValueIteration solver.

    Args:
        grid_game: An instance of the GridZeroSum class.

    Returns:
        A tuple containing:
        - rewards: An array of shape (num_states, num_actions_p1, num_actions_p2)
                   representing immediate rewards for player 1.
        - transitions: An array of shape (num_states, num_actions_p1, num_actions_p2, num_states)
                       representing transition probabilities P(s'|s,a1,a2).
        - game_params: A dictionary with game parameters ('num_states',
                       'num_actions_p1', 'num_actions_p2').
    """
    # 1. Extract game parameters from the GridZeroSum object
    num_states = len(grid_game.all_states)
    num_actions_p1 = len(grid_game.get_player_0().movements)
    num_actions_p2 = len(grid_game.get_player_1().movements)

    game_params = {
        'num_states': num_states,
        'num_actions_p1': num_actions_p1,
        'num_actions_p2': num_actions_p2,
        'num_actions': num_actions_p1
    }

    # 2. Get rewards for player 1 (pursuer)
    # Since the game is zero-sum, we only need one reward matrix.
    # The create_stage_games method returns rewards for both players. We'll use player 0's.
    rewards, _ = grid_game.create_stage_games()

    # 3. Get and reshape the transition matrix
    # The transition_matrix is already computed in the GridZeroSum constructor.
    # Its shape is (num_states * num_actions_p1 * num_actions_p2, num_states).
    flat_transitions = grid_game.transition_matrix

    # We reshape it to the required 4D format: (num_states, num_actions_p1, num_actions_p2, num_states)
    # The order 'C' (row-major) is default and matches the nested loop structure in create_transition_matrix.
    transitions = flat_transitions.reshape((num_states, num_actions_p1, num_actions_p2, num_states))

    return rewards, transitions, game_params

# test state_action_feature_map_splitted_case_interaction
if __name__ == "__main__":
    # Create a simple grid game for testing
    GAMMA = 0.9
    movements = ["left", "right", "up", "down"]
    grid_game = GridZeroSum(
        length=3, width=3,
        players=[Player(position=None, movements=movements), Player(position=None, movements=movements)],
        reward_coordinates=[[0, 2]], reward_value=1, gamma=GAMMA, render_mode='human', screen_scaling=9)
    start_state = [[1, 0], [2, 1]]
    start_state_idx = grid_game.all_states.index(start_state)
    grid_game.render(state=start_state_idx)

    # Example state and action tensors
    states = torch.tensor([0, 4])  # Example joint state indices
    actions = torch.tensor([0, 1])  # Example action indices
    
    # Get number of states and actions
    num_states = len(grid_game.all_states)
    num_actions = len(grid_game.get_player_0().movements)
    
    # Compute features
    features = state_action_feature_map_svd(grid_game, rank=30, player_idx=0)

    print("Feature shape:", features.shape)
    print("Features:\n", features)