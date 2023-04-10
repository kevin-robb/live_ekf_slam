Header header
# total number of iterations (i.e., number of vehicle poses in full history).
int32 num_iterations
# vechicle pose components for all iterations.
float32[] x_v
float32[] y_v
float32[] yaw_v

# total number of detected landmarks.
int32 M
# landmark positions [x_1,y_1,...,x_M,y_M] (2M elements).
float32[] landmarks

# we know there is a connection between every pose and the pose at the immediate previous/next iterations.
# however, each vehicle pose may have any number of connections to landmarks (including none).
# list of connections takes the form [i_1,j_1,...,i_k,j_k] where i_t is an iteration number representing a vehicle pose, and j_m is a particular landmark index.
int32[] meas_connections