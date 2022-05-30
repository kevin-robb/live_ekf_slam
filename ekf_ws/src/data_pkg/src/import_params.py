#!/usr/bin/env python3

"""
Script to read all the params from params.txt into a Python dictionary,
to be used by all the Python nodes.
Need to keep the txt file for the C++ nodes to access independently.
"""
import rospy
import rospkg, cv2
import numpy as np

class Config:
    @staticmethod
    def read_params():
        """
        Read params from config file and add to dict in correct data type.
        @param path to data_pkg.
        """
        Config.params = {}
        # find the filepath to the params file.
        rospack = rospkg.RosPack()
        Config.params["DATA_PKG_PATH"] = rospack.get_path('data_pkg')
        params_file = open(Config.params["DATA_PKG_PATH"]+"/config/params.txt", "r")
        lines = params_file.readlines()
        # set all params.
        for line in lines:
            if len(line) < 3 or line[0] == "#": # skip comments and blank lines.
                continue
            p_line = line.split("=")
            key = p_line[0].strip()
            arg = p_line[1].strip()
            try:
                Config.params[key] = int(arg)
            except:
                try:
                    Config.params[key] = float(arg)
                except:
                    # check if bool or str.
                    if arg.lower() in ["true", "false"]:
                        Config.params[key] = (arg.lower() == "true")
                    else:
                        Config.params[key] = arg
        # Compute any other params that are needed.
        # occ map <-> lm coords transform params.
        Config.params["SCALE"] = Config.params["MAP_BOUND"] / (Config.params["OCC_MAP_SIZE"] / 2)
        Config.params["SHIFT"] = Config.params["OCC_MAP_SIZE"] / 2
        # size of region plotter will display.
        Config.params["DISPLAY_REGION"] = [Config.params["MAP_BOUND"] * Config.params["DISPLAY_REGION_MULT"] * sign for sign in (-1, 1)]


    @staticmethod
    def read_map():
        """
        Read in the map from the image file and convert it to a 2D list occ grid.
        Save a color map for display, and save a binarized occupancy grid for path planning.
        """
        # read map image and account for possible white = transparency that cv2 will call black.
        # https://stackoverflow.com/questions/31656366/cv2-imread-and-cv2-imshow-return-all-zeros-and-black-image/62985765#62985765
        img = cv2.imread(Config.params["DATA_PKG_PATH"]+'/config/maps/'+Config.params["OCC_MAP_IMAGE"], cv2.IMREAD_UNCHANGED)
        if img.shape[2] == 4: # we have an alpha channel
            a1 = ~img[:,:,3] # extract and invert that alpha
            img = cv2.add(cv2.merge([a1,a1,a1,a1]), img) # add up values (with clipping)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB) # strip alpha channels
        # cv2.imshow('initial map', img); cv2.waitKey(0); cv2.destroyAllWindows()


        # save the color map for the plotter.
        # reduce from float64 (not supported by cv2) to float32.
        # convert from BGR to RGB for display.
        Config.color_map = img #cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2RGB)


        # lower the image resolution to the desired grid size.
        img = cv2.resize(img, (Config.params["OCC_MAP_SIZE"], Config.params["OCC_MAP_SIZE"]))


        # turn this into a grayscale img and then to a binary map.
        occ_map_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1]
        # normalize to range [0,1].
        occ_map_img = np.divide(occ_map_img, 255)
        # cv2.imshow("Thresholded Map", occ_map_img); cv2.waitKey(0); cv2.destroyAllWindows()

        # create matrix from map, converting cells to int 0 or 1.
        # anything not completely white (1) is considered occluded (0).
        occ_map = []
        for i in range(len(occ_map_img)):
            row = []
            for j in range(len(occ_map_img[0])):
                row.append(int(occ_map_img[i][j]))
            occ_map.append(row)
        # print("raw occupancy grid:\n",occ_map)
        # determine index pairs to select all neighbors when ballooning obstacles.
        nbrs = []
        for i in range(-Config.params["OCC_MAP_BALLOON_AMT"], Config.params["OCC_MAP_BALLOON_AMT"]+1):
            for j in range(-Config.params["OCC_MAP_BALLOON_AMT"], Config.params["OCC_MAP_BALLOON_AMT"]+1):
                nbrs.append((i, j))
        # remove 0,0 which is just the parent cell.
        nbrs.remove((0,0))
        # expand all occluded cells outwards.
        for i in range(len(occ_map)):
            for j in range(len(occ_map[0])):
                if occ_map_img[i][j] < 0.5: # occluded.
                    # mark all neighbors as occluded.
                    for chg in nbrs:
                        occ_map[max(0, min(i+chg[0], Config.params["OCC_MAP_SIZE"]-1))][max(0, min(j+chg[1], Config.params["OCC_MAP_SIZE"]-1))] = 0
        # print("inflated map:\n",occ_map)
        Config.occ_map = occ_map
        # show value distribution in occ_map.
        freqs = [0, 0]
        for i in range(len(occ_map)):
            for j in range(len(occ_map[0])):
                if occ_map[i][j] == 0:
                    freqs[0] += 1
                else:
                    freqs[1] += 1
        print("Occ map value frequencies: "+str(freqs[1])+" free, "+str(freqs[0])+" occluded.")



# automatically read in the params as soon as this is loaded.
Config.read_params()
# then load the map we've chosen.
Config.read_map()

# # only run once, since it's all static.
# if __name__ == '__main__':
#     try:
#         rospy.logwarn("defining params")
#         # automatically read in the params as soon as this is loaded.
#         Config.read_params()
#         # then load the map we've chosen.
#         Config.read_map()
#     except rospy.ROSInterruptException:
#         pass

