import random
from gym import spaces
from gym_LTL_RL.envs.gridworld.gridworld_env import GridWorldEnv, GridWorldActions
from gym_LTL_RL.envs.utils.utils import get_param
from copy import deepcopy

class OfficeWorldObject:
    AGENT = "A"
    ROOM_A = "a"
    ROOM_B = "b"
    ROOM_C = "c"
    ROOM_D = "d"
    office = "o"
    mail = "m"
    tea = "t"


class OfficeWorldRoomVisits:
    VISITED_NONE = [False, False, False, False]

class OfficeWorldEnv(GridWorldEnv):
    """
    OfficeWorld Environment is based on Icarde's paper. This environment is cre-
    ated after some modifications to the original environment.
    """

    def __init__(self, params=None):
        super(OfficeWorldEnv, self).__init__(params)

        self.agent = None       # agent's location
        self.prev_agent = None  # previous agent location
        self.init_agent = None  # agent's initial position, for resetting

        self.locations = {}     # location of the rooms: a, b, c and d
        self.walls = set()
        self.slip_rate = 0.

        # grid size
        self.height = 8
        self.width = 8

        # state
        self.visited_rooms = OfficeWorldRoomVisits.VISITED_NONE
        self.visited_rooms_prev = None

        self.visited_B = False
        self.visited_BC = False
        self.visited_BCA = False

        self._load_map()

    def set_sliprate(self, new_slip_rate):
        if new_slip_rate <= 1. and new_slip_rate >= 0.:
            self.slip_rate = new_slip_rate
        else:
            print('Invalid Slip Rate Value: Values must be interval [0, 1]')

    def set_seed(self, seed):
        random.seed(seed)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        if self.is_game_over:
            return self._get_state(), 0.0, True, self.get_observations()

        target_x, target_y = self.agent

        if action == GridWorldActions.UP or action == GridWorldActions.DOWN:
            action_ = self.action_space.np_random.choice(a=[action, GridWorldActions.RIGHT, GridWorldActions.LEFT],
                p=[1-self.slip_rate, self.slip_rate/2., self.slip_rate/2.])
        elif action == GridWorldActions.RIGHT or action == GridWorldActions.LEFT:
            action_ = self.action_space.np_random.choice(a=[action, GridWorldActions.UP, GridWorldActions.DOWN],
                p=[1-self.slip_rate, self.slip_rate/2., self.slip_rate/2.])


        if action_ == GridWorldActions.UP:
            target_y += 1
        elif action_ == GridWorldActions.DOWN:
            target_y -= 1
        elif action_ == GridWorldActions.LEFT:
            target_x -= 1
        elif action_ == GridWorldActions.RIGHT:
            target_x += 1

        target_pos = (target_x, target_y)
        if self._is_valid_position(target_pos) and (self.agent, target_pos) not in self.walls:
            self.prev_agent = self.agent
            self.agent = target_pos
            self._update_state()
        reward, is_done = self._get_reward(), self.is_terminal()
        self.is_game_over = is_done
        # print(self.visited_rooms)

        return self._get_state(), reward, is_done, self.get_observations()

    def _get_num_states(self):
        num_states = self.width * self.height
        return num_states

    def _get_state(self):
        num_states = self._get_num_states()

        state_possible_values = [self.width, self.height]
        state_variables = [self.agent[0], self.agent[1]]

        state_id = self.get_state_id(num_states, state_possible_values, state_variables)

        if self.use_one_hot_vector:
            return self.get_one_hot_state(num_states, state_id)

        return state_id

    def is_goal_achieved(self):
        raise NotImplementedError()

    def _get_reward(self):
        if self.is_goal_achieved():
            return 1.0
        return 0.0

    def is_terminal(self):
        return self.is_goal_achieved()

    def get_observations(self):
        observations = set()

        for location in self.locations:
            if location == self.agent:
                observations.add(self.locations[location])

        return observations

    def get_observables(self):
        return [OfficeWorldObject.ROOM_A, OfficeWorldObject.ROOM_B,
            OfficeWorldObject.ROOM_C, OfficeWorldObject.ROOM_D]

    def get_restricted_observables(self):
        raise NotImplementedError()

    def _update_state(self):
        self.visited_rooms_prev = []
        for items in self.visited_rooms:
            self.visited_rooms_prev.append(items)
        if self.prev_agent in self.locations:
            if self.prev_agent != self.agent:
                location = self.locations[self.prev_agent]
                if location == OfficeWorldObject.ROOM_A:
                    self.visited_rooms[0] = True
                elif location == OfficeWorldObject.ROOM_B:
                    self.visited_rooms[1] = True
                elif location == OfficeWorldObject.ROOM_C:
                    self.visited_rooms[2] = True
                elif location == OfficeWorldObject.ROOM_D:
                    self.visited_rooms[3] = True

    def reset(self, all_map=False):
        super().reset()

        # set initial state
        # self.agent = self.init_agent
        if all_map:
            self.agent = (random.randint(0, 7), random.randint(0, 7))
        else:
            self.agent = (random.randint(5, 7), random.randint(0, 2))
        self.prev_agent = None
        self.visited_rooms = [False, False, False, False]

        # update initial state according to the map layout
        self._update_state()
        self._load_map()

        return self._get_state()

    def _load_map(self):
        self._load_walls()
        self._load_paper_map()

    def _load_paper_map(self):
        self.init_agent = (7, 0)

        self.locations[(1, 1)] = OfficeWorldObject.ROOM_A
        self.locations[(6, 1)] = OfficeWorldObject.ROOM_D
        self.locations[(6, 6)] = OfficeWorldObject.ROOM_C
        self.locations[(1, 6)] = OfficeWorldObject.ROOM_B

    def _is_adjacent_to_location(self, position):
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                if (x == 0 and y != 0) or (x != 0 and y == 0):
                    test_pos = (position[0] + x, position[1] + y)
                    if test_pos in self.locations:
                        return True
        return False

    def _is_adjacent_position(self, position1, position2):
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                if (x == 0 and y != 0) or (x != 0 and y == 0):
                    test_pos = (position1[0] + x, position1[1] + y)
                    if test_pos == position2:
                        return True
        return False

    def _load_walls(self):
        # Load Room1:
        for x in range(0, 3, 2):
            for y in [2]:
                self.walls.add(((x, y), (x, y + 1)))
                self.walls.add(((x, y + 1), (x, y)))
        for y in range(0,3):
            for x in [2]:
                self.walls.add(((x, y), (x + 1, y)))
                self.walls.add(((x + 1, y), (x, y)))

        # Load Room2:
        for x in range(0, 3, 2):
            for y in [4]:
                self.walls.add(((x, y), (x, y + 1)))
                self.walls.add(((x, y + 1), (x, y)))
        for y in range(5,8):
            for x in [2]:
                self.walls.add(((x, y), (x + 1, y)))
                self.walls.add(((x + 1, y), (x, y)))

        # Load Room3:
        for x in range(5, 8, 2):
            for y in [4]:
                self.walls.add(((x, y), (x, y + 1)))
                self.walls.add(((x, y + 1), (x, y)))
        for y in range(5,8):
            for x in [4]:
                self.walls.add(((x, y), (x + 1, y)))
                self.walls.add(((x + 1, y), (x, y)))

        # Load Room4:
        for x in range(5, 8, 2):
            for y in [2]:
                self.walls.add(((x, y), (x, y + 1)))
                self.walls.add(((x, y + 1), (x, y)))
        for y in range(0,3):
            for x in [4]:
                self.walls.add(((x, y), (x + 1, y)))
                self.walls.add(((x + 1, y), (x, y)))

        # Load Door1 as a wall:
        self.walls.add(((1, 2), (1, 3)))
        self.walls.add(((1, 3), (1, 2)))

        # Unload Door2 if it is loaded at previous iterations:
        self.remove_wall((6,4), (6,5))
        pass

    def _get_num_adjacent_walls(self, position):
        num_walls = 0
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                if (x == 0 and y != 0) or (x != 0 and y == 0):
                    test_pos = (position[0] + x, position[1] + y)
                    if self._find_wall(test_pos):
                        num_walls += 1
        return num_walls

    def add_wall(self, first_pos, adj_pos):
        if self._is_adjacent_position(first_pos, adj_pos):
            self.walls.add((first_pos, adj_pos))
            self.walls.add((adj_pos, first_pos))
            pass
        else:
            print("Positions are not adjacent, Wall is not added")
        return None

    def remove_wall(self, first_pos, adj_pos):
        if self._is_adjacent_position(first_pos, adj_pos):
            if (first_pos, adj_pos) in self.walls:
                self.walls.remove((first_pos, adj_pos))
                self.walls.remove((adj_pos, first_pos))
            else:
                # print("Wall does not exits already")
                pass
        else:
            print("Positions are not adjacent, Wall is not removed")
        return None

    def _find_wall(self, position):
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                if (x == 0 and y != 0) or (x != 0 and y == 0):
                    test_pos = (position[0] + x, position[1] + y)
                    test_wall = (position, test_pos)
                    if test_wall in self.walls:
                        return True
        return False

    def _is_valid_position(self, pos):
        return 0 <= pos[0] < self.width and 0 <= pos[1] < self.height

    def render(self, mode='human'):
        self._render_horizontal_line()
        for y in range(self.height - 1, -1, -1):
            print("|", end="")
            for x in range(0, self.width):
                position = (x, y)
                if position == self.agent:
                    print(OfficeWorldObject.AGENT, end="")
                elif position in self.locations:
                    print(self.locations[position], end="")
                else:
                    print(" ", end="")

                wall = ((x, y), (x + 1, y))
                if wall in self.walls:
                    print("*", end="")
                else:
                    print(" ", end="")
            print("|")

            if y > 0:
                self._render_horizontal_wall(y)

        self._render_horizontal_line()

    def _render_horizontal_wall(self, y):
        print("|", end="")
        for x in range(0, self.width):
            wall = ((x, y), (x, y - 1))
            if wall in self.walls:
                print("--", end="")
            else:
                print("  ", end="")
        print("|")

    def _render_horizontal_line(self):
        for x in range(0, 2 * self.width + 2):
            print("-", end="")
        print()

class OfficeWorldBigEnv(OfficeWorldEnv):
    """Bigger version of officeworld environment. Differences are map size and
    places of walls."""

    def __init__(self, params=None):
        super(OfficeWorldBigEnv, self).__init__()

        # Grid Size
        self.height = 9
        self.width = 12

        # State
        self.visited_rooms = OfficeWorldRoomVisits.VISITED_NONE
        self.visited_rooms_prev = None

        self._load_map()

    def reset(self, all_map=False):
        super().reset()

        # set initial state
        if all_map:
            self.agent = (random.randint(0, 8), random.randint(0, 11))
        else:
            self.agent = (random.randint(0, 2), random.randint(0, 2))

        self.prev_agent = None
        self.visited_rooms = [False, False, False, False, False, False, False]
        # Order of visited rooms roomA, roomB, roomC, roomD, office, mail, tea

        # update initial state according to the map layout
        self._update_state()
        self._load_map()

    def _load_map(self):
        self._load_walls()
        self._load_paper_map()

    def get_observables(self):
        return [OfficeWorldObject.ROOM_A, OfficeWorldObject.ROOM_B,
            OfficeWorldObject.ROOM_C, OfficeWorldObject.ROOM_D,
            OfficeWorldObject.office, OfficeWorldObject.mail,
            OfficeWorldObject.tea]

    def _load_paper_map(self):
        self.init_agent = (0, 0)

        # self.locations[(1, 1)] = OfficeWorldObject.ROOM_A
        # self.locations[(11, 1)] = OfficeWorldObject.ROOM_B
        self.locations[(11, 7)] = OfficeWorldObject.ROOM_C
        # self.locations[(1, 7)] = OfficeWorldObject.ROOM_D
        self.locations[(4, 4)] = OfficeWorldObject.office
        self.locations[(7, 4)] = OfficeWorldObject.mail
        self.locations[(8, 2)] = OfficeWorldObject.tea
        self.locations[(3, 6)] = OfficeWorldObject.tea

    def _load_walls(self):
        for j in [2, 5]:
            for i in range(12):
                super().add_wall((i,j),(i,j+1))

        for i in [2, 5, 8]:
            for j in range(9):
                super().add_wall((i,j),(i+1, j))

        for j in [1, 7]:
            for i in [2, 5, 8]:
                super().remove_wall((i,j),(i+1, j))

        for j in [2, 5]:
            if j == 2:
                for i in [1, 10]:
                    super().remove_wall((i,j),(i,j+1))
            else:
                for i in [1,4,7,10]:
                    super().remove_wall((i,j),(i,j+1))


    def _update_state(self):
        self.visited_rooms_prev = []
        for items in self.visited_rooms:
            self.visited_rooms_prev.append(items)
        if self.prev_agent in self.locations:
            if self.prev_agent != self.agent:
                location = self.locations[self.prev_agent]
                if location == OfficeWorldObject.ROOM_A:
                    self.visited_rooms[0] = True
                elif location == OfficeWorldObject.ROOM_B:
                    self.visited_rooms[1] = True
                elif location == OfficeWorldObject.ROOM_C:
                    self.visited_rooms[2] = True
                elif location == OfficeWorldObject.ROOM_D:
                    self.visited_rooms[3] = True
                elif location == OfficeWorldObject.office:
                    self.visited_rooms[4] = True
                elif location == OfficeWorldObject.mail:
                    self.visited_rooms[5] = True
                elif location == OfficeWorldObject.tea:
                    self.visited_rooms[6] = True


class OfficeWorldDoorsTask2(OfficeWorldEnv):
    """
    Go to ROOM_A without doors
    """
    def is_goal_achieved(self):
        if OfficeWorldObject.ROOM_A in self.get_observations() and \
            self.visited_rooms[1] and not self.visited_rooms[2]:
            return True
        else:
            return False

    def _load_walls(self):
        super()._load_walls()
        self.remove_wall((1, 2), (1, 3))

    def get_restricted_observables(self):
        return []

class OfficeWorldDoorsTask1(OfficeWorldEnv):
    """
    Go to ROOM_A
    """
    def is_goal_achieved(self):
        return OfficeWorldObject.ROOM_A in self.get_observations()

    def step(self, action):
        super().step(action)

        # Open and Close Doors according to rules:
        if self.visited_rooms[1] == True  and self.visited_rooms_prev[1] == False:
            self.remove_wall((1,2), (1,3))
        if self.visited_rooms[2] == True and self.visited_rooms_prev[2] == False:
            self.add_wall((6,4), (6,5))

        reward, is_done = self._get_reward(), self.is_terminal()
        self.is_game_over = is_done

        return self._get_state(), reward, is_done, self.get_observations()

    def get_restricted_observables(self):
        return []

class OfficeWorldBigTask1(OfficeWorldBigEnv):
    """
    Visit o and m first then go to C while avoiding B.
    """
    def is_goal_achieved(self):
        if (self.visited_rooms[4] and self.visited_rooms[5]
            and not self.visited_rooms[1]) and \
             (OfficeWorldObject.ROOM_C in self.get_observations()):
             return True
        else:
             return False

class OfficeWorldTaskBCA(OfficeWorldEnv):
    """
    Visit B and C(not ordered) then A
    """
    # def __init__(self, params=None):
    #     super(OfficeWorldTaskBCA, self).__init__()
    #     self.visited_B = False
    #     self.visited_BC = False
    #     self.visited_BCA = False

    def _update_state(self):
        super()._update_state()
        if OfficeWorldObject.ROOM_B in self.get_observations():
            self.visited_B = True

        if self.visited_B and OfficeWorldObject.ROOM_C in self.get_observations():
            self.visited_BC = True

        if self.visited_BC and OfficeWorldObject.ROOM_A in self.get_observations():
            self.visited_BCA = True

    def is_goal_achieved(self):
        if self.visited_BCA:
            return True
        else:
            return False

    def _load_walls(self):
        super()._load_walls()
        self.remove_wall((1, 2), (1, 3))
