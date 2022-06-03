#!/usr/bin/env python3

"""
Set of static functions to perform A* path planning.
"""
from math import cos, sin

class Astar:
    # Always use the same map.
    occ_map = None

    @staticmethod
    def local_planner(cur):
        """
        Choose a free point a small distance ahead to plan to.
        Simulates the veh only having access to a small local map.
        """
        # choose ideal point at the desired distance ahead of the vehicle.
        pt = (cur[0]+Astar.params["LOCAL_PLANNER_DIST"]*cos(cur[2]), cur[1]+Astar.params["LOCAL_PLANNER_DIST"]*sin(cur[2]))
        goal = Astar.tf_ekf_to_map(pt)
        # cap this point to make sure it's on the map.
        goal = [max(0, min(goal[0], Astar.params["OCC_MAP_SIZE"]-1)), max(0, min(goal[1], Astar.params["OCC_MAP_SIZE"]-1))]
        # if this point is free, use it.
        if Astar.occ_map[goal[0]][goal[1]] == 1:
            return Astar.tf_map_to_ekf(goal)

        # find the nearest free cell to this one to use as our goal.
        open_list = [Cell(goal)]
        closed_list = []
        # iterate until finding a free cell or exhausting all cells within region.
        while len(open_list) > 0:
            # move first element of open list to closed list.
            cur_cell = open_list.pop(0)

            # add this node to the closed list.
            closed_list.append(cur_cell)
            # add its neighbors to the open list.
            for chg in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                # check each cell for occlusion and if we've already checked it.
                nbr = Cell([cur_cell.i+chg[0], cur_cell.j+chg[1]], parent=cur_cell)
                # skip if out of bounds.
                if nbr.i < 0 or nbr.j < 0 or nbr.i >= Astar.params["OCC_MAP_SIZE"] or nbr.j >= Astar.params["OCC_MAP_SIZE"]: continue
                
                # stop if we've found a free cell to use as the goal.
                if Astar.occ_map[nbr.i][nbr.j] == 1:
                    # return this pt to be used for A* path planning.
                    return Astar.tf_map_to_ekf((nbr.i, nbr.j))
                    
                # skip if already in closed list.
                if any([nbr == c for c in closed_list]): continue
                # skip if already in open list.
                if any([nbr == c for c in open_list]): continue
                # add cell to open list.
                open_list.append(nbr)
        # no free cell was found.
        return None


    @staticmethod
    def astar(start, goal):
        """
        Use A* to generate a path from the current pose to the goal position.
        1 map grid cell = 0.1x0.1 units in ekf coords.
        map (0,0) = ekf (-10,10).
        """
        # define goal node.
        goal_cell = Cell(Astar.tf_ekf_to_map(goal))
        # define start node.
        start_cell = Cell(Astar.tf_ekf_to_map(start))
        # make sure starting pose is on the map.
        if start_cell.i < 0 or start_cell.j < 0 or start_cell.i >= Astar.params["OCC_MAP_SIZE"] or start_cell.j >= Astar.params["OCC_MAP_SIZE"]:
            print("Starting position for A* not within map bounds.")
            return
        # check if starting node (veh pose) is in collision.
        start_cell.in_collision = Astar.occ_map[start_cell.i][start_cell.j] == 0
        # add starting node to open list.
        open_list = [start_cell]
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
            for chg in Astar.nbrs:
                nbr = Cell([cur_cell.i+chg[0], cur_cell.j+chg[1]], parent=cur_cell)
                # skip if out of bounds.
                if nbr.i < 0 or nbr.j < 0 or nbr.i >= Astar.params["OCC_MAP_SIZE"] or nbr.j >= Astar.params["OCC_MAP_SIZE"]: continue
                # skip if occluded, unless parent is occluded.
                nbr.in_collision = Astar.occ_map[nbr.i][nbr.j] == 0
                if nbr.in_collision and not nbr.parent.in_collision: continue
                # if Astar.occ_map[nbr.i][nbr.j] == 0: continue
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
                # compute heuristic "cost-to-go"
                if Astar.params["ASTAR_INCL_DIAGONALS"]:
                    # chebyshev heuristic
                    nbr.set_cost(h=max(abs(goal_cell.i - nbr.i), abs(goal_cell.j - nbr.j)))
                else:
                    # euclidean heuristic. (keep squared to save unnecessary computation.)
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
    def interpret_astar_path(path_to_start):
        """
        A* gives a reverse list of index positions to get to goal.
        Reverse this, convert to EKF coords, and add all to goal_queue.
        """
        if path_to_start is None:
            return
        goal_queue = []
        for i in range(len(path_to_start)-1, -1, -1):
            # convert pt to ekf coords and add to path.
            goal_queue.append(Astar.tf_map_to_ekf(path_to_start[i]))
            # pp.add_point(Astar.tf_map_to_ekf(path_to_start[i]))
        return goal_queue



class Cell:
    def __init__(self, pos, parent=None):
        self.i = int(pos[0])
        self.j = int(pos[1])
        self.parent = parent
        self.g = 0 if parent is None else parent.g + 1
        self.f = 0
        self.in_collision = False
    
    def set_cost(self, h=None, g=None):
        # set/update either g or h and recompute the cost, f.
        if h is not None:
            self.h = h
        if g is not None:
            self.g = g
        # update the cost.
        self.f = self.g + self.h
        # give huge penalty if in collision to encourage leaving occluded cells ASAP.
        if self.in_collision: self.f += 1000

    def __eq__(self, other):
        return self.i == other.i and self.j == other.j

    def __str__(self):
        return "Cell ("+str(self.i)+","+str(self.j)+") with costs "+str([self.g, self.f])
