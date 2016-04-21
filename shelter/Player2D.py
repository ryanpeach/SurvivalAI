from collections import defaultdict
import numpy as np
from World2D import *

class Player:
    def __init__(self, world, start_loc, inv):
        self.W = world
        self.Loc = start_loc
        
        # Create inventory
        self.INV = defaultdict(int)
        for k, v in inv.items():
            self.INV[k] = v
            
        # Default selection
        self.sel = KEY['wall']
        
        # Define actions
        move_up     = lambda: self._move(0,-1)
        move_down   = lambda: self._move(0,1)
        move_left   = lambda: self._move(-1,0)
        move_right  = lambda: self._move(1,0)
        sel_torch   = lambda: self._sel(KEY['torch'])
        sel_wall    = lambda: self._sel(KEY['wall'])
        sel_door    = lambda: self._sel(KEY['door'])
        sel_space   = lambda: self._sel(KEY['space'])
        place_up    = lambda: self._place(0,-1)
        place_down  = lambda: self._place(0,1)
        place_left  = lambda: self._place(-1,0)
        place_right = lambda: self._place(1,0)
        test_score  = self._score

        # Create action lookup
        self.actions = {0: move_up, 1: move_left, 2: move_right,
                        3: move_down, 4: sel_wall, 5: sel_torch, 6: sel_door,
                        7: sel_space, 8: place_up, 9: place_down, 10: place_right,
                        11: place_left, 12: test_score}
        
    def action(self, a):
        return self.actions[a]()
        
    def _sel(self, val):
        if self.sel != val:
            self.sel = val
            return True
        return False
        
    def _move(self, dx, dy):
        y0, x0 = self.Loc
        x1, y1 = x0 + dx, y0 + dy
        xvalid = not (x1 < 0 or x1 >= self.W.shape[1])                          # If new loc is not past the boundary
        yvalid = not (y1 < 0 or y1 >= self.W.shape[0])                          # -- Also for y
        passable = W[y1,x1] not in not_passable                                 # And the space is passable
        if xvalid and yvalid and passable:
            self.Loc = (y1, x1)                                                 # Move to new location
            return True                                                         # Return success
        return False                                                            # Otherwise, return failure
        
    def _place(self, dx, dy):
        x1, y1 = self.Loc[1] + dx, self.Loc[0] + dy

        if not (x1 < 0 or x1 >= self.W.shape[1]) \
            and not (y1 < 0 or y1 >= self.W.shape[0]):                          # If new loc is not past the boundaries
            if self.sel == KEY['space']:                                        # If you have selected to add a space (remove block)
                if self.W[y1,x1] not in not_removable:                          # You may only do this if the target is removable
                    self.INV[self.W[y1,x1]] += 1                                # Add the item to your inventory
                    self.W[y1,x1] = self.sel                                    # Remove it from the world
                    return True                                                 # Return success
            elif self.INV[self.sel] > 0:                                        # Else, if the item is in your inventory
                if self.W[y1,x1] == KEY['space']:                               # And the target is not filled
                    self.W[y1,x1] = self.sel                                    # Put the selection in the space
                    self.INV[KEY['space']] -= 1                                 # Remove it from your inventory
                    return True                                                 # Return success

        return False                                                            # Otherwise, return failure
        
    def _score(self):
        L = generate_light(self.W, d_l = 2)
        P = generate_particles(self.W, L, Np=10)
        S = run_simulation(P)
        return scoreWorld(W,S)
        
if __name__=="__main__":
    # Create World
    Ny, Nx = 10, 10
    W = np.full((Ny,Nx), KEY['space'])    # This is our world
    
    # Create a basic house
    H = [[KEY['wall'],KEY['wall'], KEY['wall'], KEY['wall']],
         [KEY['wall'],KEY['space'],KEY['space'],KEY['wall']],
         [KEY['wall'],KEY['torch'],KEY['space'],KEY['wall']],
         [KEY['wall'],KEY['wall'], KEY['wall'], KEY['wall']]]
    H = np.array(H)
    
    # Place the house in the world
    W[Ny/2:Ny/2+H.shape[0], Nx/2:Nx/2+H.shape[1]] = H
    
    # Create Player
    inv = {KEY['wall']: 10, KEY['torch']: 5, KEY['door']: 1}
    player = Player(W, (Ny/2,Nx/2), inv)
    
    # Actions
    # 0: move_up, 1: move_left, 2: move_right,
    # 3: move_down, 4: sel_wall, 5: sel_torch, 6: sel_door,
    # 7: sel_space, 8: place_up, 9: place_down, 10: place_right,
    # 11: place_left, 12: test_score
    
    # Create a game loop
    completed = False
    while not completed:
        D = W.copy()
        D[player.Loc] = -1
        print(D)
        a = False
        i = str(input())
        if i == 'mu':
            a = player.action(0)
        elif i == 'ml':
            a = player.action(1)
        elif i == 'mr':
            a = player.action(2)
        elif i == 'md':
            a = player.action(3)
        elif i == 'sw':
            a = player.action(4)
        elif i == 'st':
            a = player.action(5)
        elif i == 'sd':
            a = player.action(6)
        elif i == 'sr':
            a = player.action(7)
        elif i == 'pu':
            a = player.action(8)
        elif i == 'pd':
            a = player.action(9)
        elif i == 'pr':
            a = player.action(10)
        elif i == 'pl':
            a = player.action(11)
        elif i == 'test':
            sc = player.action(12)
            print("Score: {0}".format(sc))
            a = (sc > 0)
        elif i == 'exit':
            break
        else:
            print("Command not recognized")
            
        if a:
            print("Action successful")
        else:
            print("Action unsuccessful")
        print('\n')
    
    print('Done!')