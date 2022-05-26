# this file taken from pure_pursuit.py from the nrc_software 2020 repo
import math

class PurePursuit:

    def __init__(self):
        self.path = []
    
    def add_point(self, x, y):
        self.path.append((x,y))

    def set_points(self, pts):
        self.path = pts

    def get_lookahead_point(self, x, y, r):
        # create a counter that will stop searching after counter_max checks after finding a valid lookahead
        # this should prevent seeing the start and end of the path simultaneously and going backwards
        counter = 0
        counter_max = 50
        counter_started = False

        lookahead = None

        for i in range(len(self.path)-1):
            # increment counter if at least one valid lookahead point has been found
            if counter_started:
                counter += 1
            # stop searching for a lookahead if the counter_max has been hit
            if counter >= counter_max:
                #break
                return lookahead

            segStart = self.path[i]
            segEnd = self.path[i+1]

            p1 = (segStart[0] - x, segStart[1] - y)
            p2 = (segEnd[0] - x, segEnd[1] - y)

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]

            d = math.sqrt(dx * dx + dy * dy)
            D = p1[0] * p2[1] - p2[0] * p1[1]

            discriminant = r * r * d * d - D * D
            if discriminant < 0 or p1 == p2:
                continue

            sign = lambda x: (1, -1)[x < 0]

            x1 = (D * dy + sign(dy) * dx * math.sqrt(discriminant)) / (d * d)
            x2 = (D * dy - sign(dy) * dx * math.sqrt(discriminant)) / (d * d)

            y1 = (-D * dx + abs(dy) * math.sqrt(discriminant)) / (d * d)
            y2 = (-D * dx - abs(dy) * math.sqrt(discriminant)) / (d * d)

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

        if len(self.path) > 0:
            lastPoint = self.path[len(self.path) - 1]

            endX = lastPoint[0]
            endY = lastPoint[1]

            if math.sqrt((endX - x) * (endX - x) + (endY - y) * (endY - y)) <= r:
                return (endX, endY)

        return lookahead