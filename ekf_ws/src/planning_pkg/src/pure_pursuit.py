#!/usr/bin/env python3

"""
Set of static functions to perform pure pursuit navigation.
"""

from geometry_msgs.msg import Vector3
from math import remainder, tau, pi, atan2, sqrt

class PurePursuit:
    params = None
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
        cmd_msg = Vector3(x=0, y=0)
        if len(PurePursuit.goal_queue) < 1: 
            # if there's no path yet, just wait. (send 0 cmd)
            return cmd_msg

        # define lookahead point.
        lookahead_pt = None
        radius = 0.2 # starting search radius.
        # look until we find the path at the increasing radius or at the maximum dist.
        while lookahead_pt is None and radius <= 3: 
            # TODO make functions to actually do the pure pursuit thing.
            lookahead_pt = PurePursuit.get_lookahead_point(cur[0], cur[1], radius)
            radius *= 1.25
        # make sure we actually found the path.
        if lookahead_pt is not None:
            # compute global heading to lookahead_pt
            gb = atan2(lookahead_pt[1] - cur[1], lookahead_pt[0] - cur[0])
            # compute hdg relative to veh pose.
            beta = remainder(gb - cur[2], tau)

            # update global integral term.
            PurePursuit.integ += beta * PurePursuit.params["DT"]

            P = 0.08 * beta # proportional to hdg error.
            I = 0.009 * PurePursuit.integ # integral to correct systematic error.
            D = 0.01 * (beta - PurePursuit.err_prev) / PurePursuit.params["DT"] # slope to reduce oscillation.

            # set forward and turning commands.
            cmd_msg.x = (1 - abs(beta / pi))**5 + 0.05
            cmd_msg.y = P + I + D

            PurePursuit.err_prev = beta

        # ensure commands are capped within constraints.
        cmd_msg.x = max(0, min(cmd_msg.x, PurePursuit.params["ODOM_D_MAX"]))
        cmd_msg.y = max(-PurePursuit.params["ODOM_TH_MAX"], min(cmd_msg.y, PurePursuit.params["ODOM_TH_MAX"]))
        return cmd_msg

    @staticmethod
    def direct_nav(cur):
        """
        Navigate directly point-to-point without using pure pursuit.
        """
        cmd_msg = Vector3(x=0, y=0)
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
        cmd_msg.x = 1 * (1 - abs(beta)/PurePursuit.params["ODOM_TH_MAX"])**3 + 0.05 if r > 0.1 else 0.0
        P = 0.03 if r > 0.2 else 0.2
        cmd_msg.y = beta #* P if r > 0.05 else 0.0
        # ensure commands are capped within constraints.
        cmd_msg.x = max(0, min(cmd_msg.x, PurePursuit.params["ODOM_D_MAX"]))
        cmd_msg.y = max(-PurePursuit.params["ODOM_TH_MAX"], min(cmd_msg.y, PurePursuit.params["ODOM_TH_MAX"]))
        # remove the goal from the queue if we've arrived.
        if r < 0.15:
            PurePursuit.goal_queue.pop(0)
        return cmd_msg


# this stuff taken from pure_pursuit.py from the nrc_software 2020 repo

    # def __init__(self):
    #     PurePursuit.goal_queue = []
    
    # def add_point(self, pt):
    #     PurePursuit.goal_queue.append(pt)

    # def set_points(self, pts):
    #     PurePursuit.goal_queue = pts

    @staticmethod
    def get_lookahead_point(x, y, r):
        # create a counter that will stop searching after counter_max checks after finding a valid lookahead
        # this should prevent seeing the start and end of the path simultaneously and going backwards
        counter = 0
        counter_max = 50
        counter_started = False

        lookahead = None

        for i in range(len(PurePursuit.goal_queue)-1):
            # increment counter if at least one valid lookahead point has been found
            if counter_started:
                counter += 1
            # stop searching for a lookahead if the counter_max has been hit
            if counter >= counter_max:
                #break
                return lookahead

            segStart = PurePursuit.goal_queue[i]
            segEnd = PurePursuit.goal_queue[i+1]

            p1 = (segStart[0] - x, segStart[1] - y)
            p2 = (segEnd[0] - x, segEnd[1] - y)

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]

            d = sqrt(dx * dx + dy * dy)
            D = p1[0] * p2[1] - p2[0] * p1[1]

            discriminant = r * r * d * d - D * D
            if discriminant < 0 or p1 == p2:
                continue

            sign = lambda x: (1, -1)[x < 0]

            x1 = (D * dy + sign(dy) * dx * sqrt(discriminant)) / (d * d)
            x2 = (D * dy - sign(dy) * dx * sqrt(discriminant)) / (d * d)

            y1 = (-D * dx + abs(dy) * sqrt(discriminant)) / (d * d)
            y2 = (-D * dx - abs(dy) * sqrt(discriminant)) / (d * d)

            validIntersection1 = min(p1[0], p2[0]) < x1 and x1 < max(p1[0], p2[0]) or min(p1[1], p2[1]) < y1 and y1 < max(p1[1], p2[1])
            validIntersection2 = min(p1[0], p2[0]) < x2 and x2 < max(p1[0], p2[0]) or min(p1[1], p2[1]) < y2 and y2 < max(p1[1], p2[1])

            if validIntersection1 or validIntersection2:
                # we are within counter_max, so reset the counter if it has been started, or start it if not
                if counter_started:
                    counter = 0
                else:
                    counter_started = True
                    counter = 0

                lookahead = None

            if validIntersection1:
                lookahead = (x1 + x, y1 + y)

            if validIntersection2:
                if lookahead == None or abs(x1 - p2[0]) > abs(x2 - p2[0]) or abs(y1 - p2[1]) > abs(y2 - p2[1]):
                    lookahead = (x2 + x, y2 + y)

        if len(PurePursuit.goal_queue) > 0:
            lastPoint = PurePursuit.goal_queue[len(PurePursuit.goal_queue) - 1]

            endX = lastPoint[0]
            endY = lastPoint[1]

            if sqrt((endX - x) * (endX - x) + (endY - y) * (endY - y)) <= r:
                return (endX, endY)

        return lookahead