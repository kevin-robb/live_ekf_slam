Header header
# timestep.
int32 timestep
# vechicle pose.
float32 x_v
float32 y_v
float32 yaw_v
# number of landmarks.
int32 M
# landmark positions [x_1,y_1,...,x_M,y_M] (2M elements).
float32[] landmarks
# covariance matrix. (stored as one long row with (3+2M)^2 elements).
float32[] P
# to iterate through P matrix:
# int count=0;
# for(int i=0;i<size*size;i++){
#     if(count%input.size==0)
#         cout<<endl;
#     cout<<matrix.data[i]<<" ";
#     count++;
# }
# referenced https://answers.ros.org/question/62555/custom-message-for-matrix-or-2d-array/