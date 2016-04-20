import numpy as np
from World2D import *

class Player:
    def __init__(self, world, start_loc):
        self.W = world
        self.Loc = start_loc
        
        # Define actions
        move_up    = lambda: self._move(0,-1)
        move_down  = lambda: self._move(0,1)
        move_left  = lambda: self._move(-1,0)
        move_right = lambda: self._move(1,0)
        place_torch = lambda: self._place(KEY['torch'])
        place_wall  = lambda: self._place(KEY['wall'])
        remove_bloc = lambda: self._remove()
        test_score  = lambda: self._score()
        
        # Create action lookup
        self.actions = {0: move_up, 1: move_left, 2: move_right,
                        3: move_down, 4: place_wall, 5: place_torch,
                        6: remove_block, 7: test_score}
        
    def action(self, a):
        return self.actions[a]()
        
    def _move(self, dx, dy):
        x0, y0 = self.Loc
        x1, y1 = x0 + dx, y1 + dy
        if x1 < 0 or x1 >= self.W.shape[1]:
            return False
        elif y1 < 0 or y1 >= self.W.shape[0]:
            return False
        else:
            self.Loc[0] = y1
            self.Loc[1] = x1
            return True
        
    def _place(self, val):
        if self.W[self.Loc] == KEY['space']:
            self.W[self.Loc] = val
            return True
        else:
            return False
        
    def _remove(self):
        if self.W[self.Loc] == KEY['space']:
            return False
        else:
            self.W[self.Loc] = KEY['space']
            return True
            
    def _score(self):
        L = generate_light(self.W, d_l = 2)
        P = generate_particles(self.W, L, Np=10)
        S = run_simulation(P)
        return scoreWorld(W,S)