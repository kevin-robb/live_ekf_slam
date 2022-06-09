#!/usr/bin/env python3

"""
Script to read all the params from params.txt into a Python dictionary,
to be used by all the Python nodes.
Need to keep the txt file for the C++ nodes to access independently.
"""
import rospkg

class Config:
    @staticmethod
    def read_params():
        """
        Read params from config file and add to dict in correct data type.
        @param path to base_pkg.
        """
        Config.params = {}
        # find the filepath to the params file.
        rospack = rospkg.RosPack()
        Config.params["BASE_PKG_PATH"] = rospack.get_path('base_pkg')
        params_file = open(Config.params["BASE_PKG_PATH"]+"/config/params.txt", "r")
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
        

# automatically read in the params as soon as this is loaded.
Config.read_params()
