Header header
# timestep.
int32 timestep
# vechicle poses, past->current.
float32[] x_v
float32[] y_v
float32[] yaw_v
# number of landmarks.
int32 M
# landmark positions [x_1,y_1,...,x_M,y_M] (2M elements).
float32[] landmarks
# covariance matrix. (stored as one long row with (3+2M)^2 elements).
float32[] P