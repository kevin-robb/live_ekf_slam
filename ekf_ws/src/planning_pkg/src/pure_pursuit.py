#!/usr/bin/env python3

"""
Set of static functions to perform pure pursuit navigation.
"""

from data_pkg.msg import Command
from math import remainder, tau, pi, atan2, sqrt

class PurePursuit:
    goal_queue = []
    # PID vars.
    integ = 0
    err_prev = 0.0
    # we're synched to the EKF's "clock", so all time increments are the same.

    @staticmethod
    def get_next_cmd(cur):
        """
        Determine odom command to stay on the path.
        """
        # pare the path up to current veh pos.
        PurePursuit.pare_path(cur)

        cmd_msg = Command(fwd=0, ang=0)
        if len(PurePursuit.goal_queue) < 1: 
            # if there's no path yet, just wait. (send 0 cmd)
            return cmd_msg

        # define lookahead point.
        lookahead_pt = None
        lookahead_dist = PurePursuit.params["LOOKAHEAD_DIST_INITIAL"] # starting search radius.
        # look until we find the path, or give up at the maximum dist.
        while lookahead_pt is None and lookahead_dist <= PurePursuit.params["LOOKAHEAD_DIST_MAX"]: 
            lookahead_pt = PurePursuit.choose_lookahead_pt(cur, lookahead_dist)
            lookahead_dist *= 1.25
        # make sure we actually found the path.
        if lookahead_pt is None:
            # we can't see the path, so just try to go to the first pt.
            lookahead_pt = PurePursuit.goal_queue[0]
        
        # compute global heading to lookahead_pt
        gb = atan2(lookahead_pt[1] - cur[1], lookahead_pt[0] - cur[0])
        # compute hdg relative to veh pose.
        beta = remainder(gb - cur[2], tau)

        # update global integral term.
        PurePursuit.integ += beta * PurePursuit.params["DT"]

        P = 0.5 * beta # proportional to hdg error.
        I = 0 #.01 * PurePursuit.integ # integral to correct systematic error.
        D = 0 #.01 * (beta - PurePursuit.err_prev) / PurePursuit.params["DT"] # slope to reduce oscillation.

        # set forward and turning commands.
        cmd_msg.fwd = 0.02 * (1 - abs(beta / pi))**12 + 0.01
        cmd_msg.ang = P + I + D

        PurePursuit.err_prev = beta
        

        # ensure commands are capped within constraints.
        cmd_msg.fwd= max(0, min(cmd_msg.fwd, PurePursuit.params["ODOM_D_MAX"]))
        cmd_msg.ang = max(-PurePursuit.params["ODOM_TH_MAX"], min(cmd_msg.ang, PurePursuit.params["ODOM_TH_MAX"]))
        return cmd_msg


    @staticmethod
    def pare_path(cur):
        """
        If the vehicle is near a path pt, cut the path off up to this pt.
        """
        for i in range(len(PurePursuit.goal_queue)):
            r = ((cur[0]-PurePursuit.goal_queue[i][0])**2 + (cur[1]-PurePursuit.goal_queue[i][1])**2)**(1/2)
            if r < 0.15:
                # remove whole path up to this pt.
                del PurePursuit.goal_queue[0:i+1]
                return


    @staticmethod
    def choose_lookahead_pt(cur, lookahead_dist):
        """
        Find the point on the path at the specified radius from the current veh pos.
        """
        # if there's only one path point, go straight to it.
        if len(PurePursuit.goal_queue) == 1:
            return PurePursuit.goal_queue[0]
        lookahead_pt = None
        # check the line segments between each pair of path points.
        for i in range(1, len(PurePursuit.goal_queue)):
            # get vector between path pts.
            diff = [PurePursuit.goal_queue[i][0]-PurePursuit.goal_queue[i-1][0], PurePursuit.goal_queue[i][1]-PurePursuit.goal_queue[i-1][1]]
            # get vector from veh to first path pt.
            v1 = [PurePursuit.goal_queue[i-1][0]-cur[0], PurePursuit.goal_queue[i-1][1]-cur[1]]
            # compute coefficients for quadratic eqn to solve.
            a = diff[0]**2 + diff[1]**2
            b = 2*(v1[0]*diff[0] + v1[1]*diff[1])
            c = v1[0]**2 + v1[1]**2 - lookahead_dist**2
            try:
                discr = sqrt(b**2 - 4*a*c)
            except:
                # discriminant is negative, so there are no real roots.
                # (line segment is too far away)
                continue
            # compute solutions to the quadratic.
            # these will tell us what point along the 'diff' line segment is a solution.
            q = [(-b-discr)/(2*a), (-b+discr)/(2*a)]
            # check validity of solutions.
            valid = [q[i] >= 0 and q[i] <= 1 for i in range(2)]
            # compute the intersection pt. it's the first seg pt + q percent along diff vector.
            if valid[0]: lookahead_pt = [PurePursuit.goal_queue[i-1][0]+q[0]*diff[0], PurePursuit.goal_queue[i-1][1]+q[0]*diff[1]]
            elif valid[1]: lookahead_pt = [PurePursuit.goal_queue[i-1][0]+q[1]*diff[0], PurePursuit.goal_queue[i-1][1]+q[1]*diff[1]]
            else: continue # no intersection pt in the allowable range.
        return lookahead_pt


    @staticmethod
    def direct_nav(cur):
        """
        Navigate directly point-to-point without using pure pursuit.
        """
        cmd_msg = Command(fwd=0, ang=0)
        if len(PurePursuit.goal_queue) < 1: # if there's no path yet, just wait.
            return cmd_msg
        goal = PurePursuit.goal_queue[0]
        # check how close veh is to our current goal pt.
        r = ((cur[0]-goal[0])**2 + (cur[1]-goal[1])**2)**(1/2) 
        # compute vector from veh pos est to goal.
        diff_vec = [goal[0]-cur[0], goal[1]-cur[1]]
        # calculate bearing difference.
        gb = atan2(diff_vec[1], diff_vec[0]) # global bearing.
        beta = remainder(gb - cur[2], tau) # bearing rel to robot

        # go faster the more aligned the hdg is.
        cmd_msg.fwd= 1 * (1 - abs(beta)/PurePursuit.params["ODOM_TH_MAX"])**3 + 0.05 if r > 0.1 else 0.0
        P = 0.03 if r > 0.2 else 0.2
        cmd_msg.ang = beta #* P if r > 0.05 else 0.0
        # ensure commands are capped within constraints.
        cmd_msg.fwd= max(0, min(cmd_msg.fwd, PurePursuit.params["ODOM_D_MAX"]))
        cmd_msg.ang = max(-PurePursuit.params["ODOM_TH_MAX"], min(cmd_msg.ang, PurePursuit.params["ODOM_TH_MAX"]))
        # remove the goal from the queue if we've arrived.
        if r < 0.15:
            PurePursuit.goal_queue.pop(0)
        return cmd_msg

