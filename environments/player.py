import numpy as np


class Player:
    def __init__(
        self,
        position: list = [0, 0],
        movements: list = ["left", "right", "up", "down", "stay"],
    ):
        """
        This class represents a player in the gridworld.

        Args:
            position (list): A list of two integers representing the starting position coordinates of the player.
            movements (list): A list of strings representing the possible movements.
        """
        self.movements = movements
        self.position = position

    def move(self, movement: str):
        """
        Compute the new position of the player after performing a movement.

        Args:
            movement (str): The movement to perform. If the movement is invalid or not in the list of movements,
                            the player stays in the current position.

        Returns:
            list: The new position of the player after performing the movement.
        """
        if movement == "left" and "left" in self.movements:
            new_position = [self.position[0], self.position[1]-1]
        elif movement == "right" and "right" in self.movements:
            new_position = [self.position[0], self.position[1]+1]
        elif movement == "up" and "up" in self.movements:
            new_position = [self.position[0]-1, self.position[1]]
        elif movement == "down" and "down" in self.movements:
            new_position = [self.position[0]+1, self.position[1]]
        else:
            new_position = self.position

        return new_position


class Expert_Player(Player):
    def __init__(
        self,
        position: list = [0, 0],
        movements: list = ["left", "right", "up", "down", "stay"],
        starting_position: list = [0, 0],
    ):
        """
        This class represents an expert player in the gridworld.

        Args:
            position (list): A list of two integers representing the current position coordinates of the player.
            movements (list): A list of strings representing the possible movements.
            starting_position (list): A list of two integers representing the starting position coordinates of the player.
        """
        super().__init__(position, movements)
        self.starting_position = starting_position

        # NOTE: basically the expert has two policies that depend on how the expert is initalized or a fallback random policy

    def move_expert(self):
        """
        Computes the new position according to the expert policy.

        Returns:
            list: The new position of the expert agent.
        """
        if self.starting_position == [0, 0]:
            if self.position == [0, 0]:
                move = "up"
                new_position = self.move(movement=move)
            elif self.position == [0, 1]:
                move = "right"
                new_position = self.move(movement=move)
            elif self.position == [1, 1]:
                move = "right"
                new_position = self.move(movement=move)
            elif self.position == [2, 1]:
                move = "up"
                new_position = self.move(movement=move)
            else:
                move = np.random.choice(self.movements)
                new_position = self.move(movement=move)

        elif self.starting_position == [2, 0]:
            if self.position == [2, 0]:
                move = "up"
                new_position = self.move(movement=move)
            elif self.position == [2, 1]:
                move = "up"
                new_position = self.move(movement=move)
            elif self.position == [2, 2]:
                move = "left"
                new_position = self.move(movement=move)
            elif self.position == [1, 2]:
                move = "left"
                new_position = self.move(movement=move)
            else:
                move = np.random.choice(self.movements)
                new_position = self.move(movement=move)

        return new_position

    def get_action(self, position: list):
        """
        Retrieves the action according to the expert's policy based on the given position.

        Args:
            position (list): The current position of the expert agent.

        Returns:
            str: The action to be performed according to the expert's policy.
        """
        if self.starting_position == [0, 0]:
            if position == [0, 0]:
                move = "up"
            elif position == [0, 1]:
                move = "right"
            elif position == [1, 1]:
                move = "right"
            elif position == [2, 1]:
                move = "up"
            else:
                move = np.random.choice(self.movements)

        elif self.starting_position == [2, 0]:
            if position == [2, 0]:
                move = "up"
            elif position == [2, 1]:
                move = "up"
            elif position == [2, 2]:
                move = "left"
            elif position == [1, 2]:
                move = "left"
            else:
                move = np.random.choice(self.movements)

        return move
