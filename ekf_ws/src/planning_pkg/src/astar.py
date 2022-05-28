#!/usr/bin/env python3

"""
Set of static functions to perform A* path planning.
"""

from std_msgs.msg import Float32MultiArray

class Astar:
    # Always use the same map.
    occ_map = None
    # Use same params as node.
    params = None

    @staticmethod
    def astar(start, goal):
        """
        Use A* to generate a path from the current pose to the goal position.
        1 map grid cell = 0.1x0.1 units in ekf coords.
        map (0,0) = ekf (-10,10).
        """
        # define goal node.
        goal_cell = Cell(Astar.tf_ekf_to_map(goal))
        # add starting node to open list.
        open_list = [Cell(Astar.tf_ekf_to_map(start))]
        closed_list = []
        # iterate until reaching the goal or exhausting all cells.
        while len(open_list) > 0:
            # move first element of open list to closed list.
            open_list.sort(key=lambda cell: cell.f)
            cur_cell = open_list.pop(0)
            # stop if we've found the goal.
            if cur_cell == goal_cell:
                # recurse up thru parents to get reverse of path from start to goal.
                path_to_start = []
                while cur_cell.parent is not None:
                    path_to_start.append((cur_cell.i, cur_cell.j))
                    cur_cell = cur_cell.parent
                return path_to_start
            # add this node to the closed list.
            closed_list.append(cur_cell)
            # add its unoccupied neighbors to the open list.
            for chg in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                # check each cell for occlusion and if we've already checked it.
                nbr = Cell([cur_cell.i+chg[0], cur_cell.j+chg[1]], parent=cur_cell)
                # skip if out of bounds.
                if nbr.i < 0 or nbr.j < 0 or nbr.i >= Astar.params["OCC_MAP_SIZE"] or nbr.j >= Astar.params["OCC_MAP_SIZE"]: continue
                # skip if occluded.
                if Astar.occ_map[nbr.i][nbr.j] == 0: continue
                # skip if already in closed list.
                if any([nbr == c for c in closed_list]): continue
                # skip if already in open list, unless the cost is lower.
                seen = [nbr == open_cell for open_cell in open_list]
                try:
                    match_i = seen.index(True)
                    # this cell has been added to the open list already.
                    # check if the new or existing route here is better.
                    if nbr.g < open_list[match_i].g: 
                        # the cell has a shorter path this new way, so update its cost and parent.
                        open_list[match_i].set_cost(g=nbr.g)
                        open_list[match_i].parent = nbr.parent
                    continue
                except:
                    # there's no match, so proceed.
                    pass
                # compute heuristic "cost-to-go". keep squared to save unnecessary computation.
                nbr.set_cost(h=(goal_cell.i - nbr.i)**2 + (goal_cell.j - nbr.j)**2)
                # add cell to open list.
                open_list.append(nbr)


    @staticmethod
    def tf_map_to_ekf(pt):
        # transform x,y from occ map indices to ekf coords.
        return [(pt[1] - Astar.params["SHIFT"]) * Astar.params["SCALE"], -(pt[0] - Astar.params["SHIFT"]) * Astar.params["SCALE"]]
        

    @staticmethod
    def tf_ekf_to_map(pt):
        # transform x,y from ekf coords to occ map indices.
        return [int(Astar.params["SHIFT"] - pt[1] / Astar.params["SCALE"]), int(Astar.params["SHIFT"] + pt[0] / Astar.params["SCALE"])]


    @staticmethod
    def interpret_astar_path(path_to_start, goal_queue, pp):
        """
        A* gives a reverse list of index positions to get to goal.
        Reverse this, convert to EKF coords, and add all to goal_queue.
        """
        if path_to_start is None:
            return
        for i in range(len(path_to_start)-1, -1, -1):
            # convert pt to ekf coords.
            goal_queue.append(Astar.tf_map_to_ekf(path_to_start[i]))
            pp.add_point(Astar.tf_map_to_ekf(path_to_start[i]))
        # publish this path for the plotter.
        return Float32MultiArray(data=sum(goal_queue, []))



class Cell:
    def __init__(self, pos, parent=None):
        self.i = int(pos[0])
        self.j = int(pos[1])
        self.parent = parent
        self.g = 0 if parent is None else parent.g + 1
        self.f = 0
    
    def set_cost(self, h=None, g=None):
        # set/update either g or h and recompute the cost, f.
        if h is not None:
            self.h = h
        if g is not None:
            self.g = g
        # update the cost.
        self.f = self.g + self.h

    def __eq__(self, other):
        return self.i == other.i and self.j == other.j

    def __str__(self):
        return "Cell ("+str(self.i)+","+str(self.j)+") with costs "+str([self.g, self.f])
