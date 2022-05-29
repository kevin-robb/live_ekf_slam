#!/usr/bin/env python3

"""
Node to read all the params from params.txt into a Python dictionary,
to be used by all the Python nodes.
"""

import rospkg

def read_params():
    """
    Read params from config file and add to dict in correct data type.
    @param path to data_pkg.
    """
    # find the filepath to the params file.
    rospack = rospkg.RosPack()
    params_file = open(rospack.get_path('data_pkg')+"/config/params.txt", "r")
    lines = params_file.readlines()
    # set all params.
    params = {}
    for line in lines:
        if len(line) < 3 or line[0] == "#": # skip comments and blank lines.
            continue
        p_line = line.split("=")
        key = p_line[0].strip()
        arg = p_line[1].strip()
        try:
            params[key] = int(arg)
        except:
            try:
                params[key] = float(arg)
            except:
                # check if bool or str.
                if arg.lower() in ["true", "false"]:
                    params[key] = (arg.lower() == "true")
                else:
                    params[key] = arg
    return params
