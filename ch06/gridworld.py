import numpy as np
from numba import int32, float32
from numba.experimental import jitclass


spec = [
    ("ix_start", int32),
    ("ix_goal", int32),
    ("n_rows", int32),
    ("n_cols", int32),
    ("n_states", int32),
    ("reward_goal", float32),
]

@jitclass(spec)
class Gridworld:
    def __init__(self, ix_start, ix_goal, n_rows, n_cols, reward_goal):
        self.ix_start = ix_start
        self.ix_goal = ix_goal
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_states = n_rows * n_cols
        self.reward_goal = reward_goal

    
    @property
    def map_ix(self):
        return  np.arange(self.n_states, dtype=np.int32).reshape(self.n_rows, self.n_cols)
    

    def get_pos(self, ix):
        row = ix // self.n_cols
        col = ix % self.n_cols
        pos = np.array([row, col])
        return pos


    def get_ix(self, pos):
        row, col = pos
        ix = row * self.n_cols + col
        return ix


    def move_pos(self, pos, step):
        row, col = pos
        row_next, col_next = pos + step
        
        if (row_next < 0) or (row_next >= self.n_rows):
            row_next = row
        if (col_next < 0) or (col_next >= self.n_cols):
            col_next = col
        
        pos_next = np.array([row_next, col_next])
        return pos_next

    def wind_strength(self, n_row, n_col):
        c_strength = 0
        
        # s = np.random.uniform() > 1/3
        if n_col in [3, 4, 5, 8]:
            r_strength = -1 
        elif n_col in [6, 7]:
            r_strength = -2 
        elif n_col in [1]: # optional
            r_strength = 2
        else:
            r_strength = 0

        move_wind = np.array([r_strength, c_strength])
        return move_wind


    def move(self, ix, step):
        pos = self.get_pos(ix)
        row, col = pos
        wind_shift = self.wind_strength(row, col)
        pos_new = self.move_pos(pos, step + wind_shift)
        ix_new = self.get_ix(pos_new)
        return ix_new


    def move_and_reward(self, ix, step):
        ix_new = self.move(ix, step)

        reward = -1 if ix != self.ix_goal else self.reward_goal
        ix_new = ix_new if ix != self.ix_goal else self.ix_start
        
        return ix_new, reward