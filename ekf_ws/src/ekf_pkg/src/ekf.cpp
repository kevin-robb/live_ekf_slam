#include "ekf_pkg/ekf.h"

// init the EKF.
EKF::EKF() {
    // set the noise covariance matrices.
    this->V.setIdentity(2,2);
    this->V(0,0) = 0.02 * 0.02;
    this->V(1,1) = (0.5*pi/180) * (0.5*pi/180);
    this->W.setIdentity(2,2);
    this->V(0,0) = 0.1 * 0.1;
    this->V(1,1) = (pi/180) * (pi/180);
    // initialize state distribution.
    this->x_t.resize(3);
    this->x_t << 0.0 , 0.0, 0.0;
    this->x_pred.resize(3);
    this->x_t << 0.0 , 0.0, 0.0;
    this->P_t.setIdentity(3,3);
    this->P_t(0,0) = 0.01 * 0.01;
    this->P_t(1,1) = 0.01 * 0.01;
    this->P_t(2,2) = 0.005 * 0.005;
    this->P_pred.setIdentity(3,3);
    this->P_pred(0,0) = 0.01 * 0.01;
    this->P_pred(1,1) = 0.01 * 0.01;
    this->P_pred(2,2) = 0.005 * 0.005;
    // set jacobians that are constant.
    this->H_w.setIdentity(2,2);
}

// perform a full iteration of the EKF for this timestep.
void EKF::update(geometry_msgs::Vector3::ConstPtr odomMsg, std_msgs::Float32MultiArray::ConstPtr lmMeasMsg) {
    ////////////// PREDICTION STAGE /////////////////
    // extract odom.
    float d_d = odomMsg->x;
    float d_th = odomMsg->y;

    // compute jacobians
    this->F_x.setIdentity(3+2*this->M, 3+2*this->M);
    this->F_x(0,2) = -1*d_d*sin(this->x_t(2));
    this->F_x(1,2) = d_d*cos(this->x_t(2));
    
    this->F_v.setZero(3+2*this->M, 2);
    this->F_v(0,0) = cos(this->x_t(2));
    this->F_v(1,0) = sin(this->x_t(2));
    this->F_v(2,1) = 1;
    // make predictions.
    this->x_pred = this->x_t;
    this->x_pred(0) = this->x_t(0) + (d_d+this->v_d)*cos(this->x_t(2));
    this->x_pred(1) = this->x_t(1) + (d_d+this->v_d)*sin(this->x_t(2));
    this->x_pred(2) = remainder(this->x_t(2) + d_th + this->v_th, 2*pi);
    // propagate covariance.
    this->P_pred = this->F_x * this->P_t * this->F_x.transpose() + this->F_v * this->V * this->F_v.transpose();

    ///////////////// UPDATE STAGE //////////////////
    std::vector<float> lm_meas = lmMeasMsg->data;
    int num_landmarks = (int) (lm_meas.size() / 3);
    // if there was no detection, skip the update stage.
    if (num_landmarks < 1) {
        this->x_t = this->x_pred;
        this->P_t = this->P_pred;
        return;
    }
    // there is at least one detection, so we must update each individually.
    for (int l=0; l<num_landmarks; ++l) {
        // extract the landmark details.
        int id = (int) lm_meas[l*3];
        float r = lm_meas[l*3+1];
        float b = lm_meas[l*3+2];
        // check if we have seen this landmark before.
        int i = -1;
        for (int j=0; j<M; ++j) {
            if (this->lm_IDs[j] == id) {
                i = j;
                break;
            }
        }
        if (i != -1) { // the landmark was found.
            //////////// LANDMARK UPDATE ///////////
            // get index of this landmark in the state.
            i = i*2+3;
            // compute estimated distance from veh to lm.
            float dist = std::sqrt(std::pow(this->x_t(i) - this->x_pred(0),2) + std::pow(this->x_t(i+1) - this->x_pred(1),2));
            // create jacobians.
            this->H_x.setZero(2,2*this->M+3);
            this->H_x(0,0) = -(this->x_t(i)-this->x_pred(0))/dist;
            this->H_x(0,1) = -(this->x_t(i+1)-this->x_pred(1))/dist;
            this->H_x(1,0) = (this->x_t(i+1)-this->x_pred(1))/(dist*dist);
            this->H_x(1,1) = -(this->x_t(i)-this->x_pred(0))/(dist*dist);
            this->H_x(1,2) = -1;
            this->H_x(0,i) = (this->x_t(i)-this->x_pred(0))/dist;
            this->H_x(0,i+1) = (this->x_t(i+1)-this->x_pred(1))/dist;
            this->H_x(1,i) = -(this->x_t(i+1)-this->x_pred(1))/(dist*dist);
            this->H_x(1,i+1) = (this->x_t(i)-this->x_pred(0))/(dist*dist);
            
            // compute the innovation.
            float ang = remainder(atan2(this->x_t(i+1)-this->x_pred(1), this->x_t(i)-this->x_pred(0)) - this->x_pred(2), 2*pi);
            this->nu(0) = r - dist - this->w_r;
            this->nu(1) = b - ang - this->w_b;
            // compute the innovation covariance.
            this->S = this->H_x * this->P_pred * this->H_x.transpose() + this->H_w * this->W * this->H_w.transpose();
            // compute the kalman gain.
            this->K = this->P_pred * this->H_x.transpose() * this->S.inverse();

            // update the state.
            this->x_pred = this->x_pred + this->K * this->nu;
            this->x_pred(2) = remainder(this->x_pred(2), 2*pi);
            this->P_pred = this->P_pred - this->K * this->H_x * this->P_pred;
        } else { // this is a new landmark.
            /////////// LANDMARK INSERTION /////////
            // increment number of landmarks.
            this->M += 1;
            // resize the state to fit the new landmark.
            this->x_pred.conservativeResize(3+2*this->M);
            this->x_pred(3+2*this->M-2) = this->x_pred(0) + r*cos(this->x_pred(2)+b);
            this->x_pred(3+2*this->M-1) = this->x_pred(1) + r*sin(this->x_pred(2)+b);
            // add landmark ID to the list.
            this->lm_IDs.push_back(id);

            // compute insertion jacobian.
            this->Y.setIdentity(3+2*this->M, 3+2*this->M);
            // G_z submatrix.
            this->Y(3+2*this->M-2,3+2*this->M-2) = cos(this->x_pred(2)+b);
            this->Y(3+2*this->M-2,3+2*this->M-1) = -r*sin(this->x_pred(2)+b);
            this->Y(3+2*this->M-1,3+2*this->M-2) = sin(this->x_pred(2)+b);
            this->Y(3+2*this->M-1,3+2*this->M-1) = r*cos(this->x_pred(2)+b);
            // G_x submatrix.
            this->Y(3+2*this->M-2,0) = 1;
            this->Y(3+2*this->M-2,1) = 0;
            this->Y(3+2*this->M-2,2) = -r*sin(this->x_pred(2)+b);
            this->Y(3+2*this->M-1,0) = 0;
            this->Y(3+2*this->M-1,1) = 1;
            this->Y(3+2*this->M-1,2) = r*cos(this->x_pred(2)+b);

            // update covariance.
            this->p1.resize(3+2*this->M-2, 3+2*this->M);
            this->z1.setZero(3+2*this->M-2, 2);
            this->p1 << this->P_pred, this->z1;

            this->p2.resize(2, 3+2*this->M);
            this->z2.setZero(2, 3+2*this->M-2);
            this->p2 << z2, this->W;

            this->p3.resize(3+2*this->M, 3+2*this->M);
            this->p3 << this->p1, this->p2;
            
            this->P_pred = this->Y * this->p3 * this->Y.transpose();
        }
    }
    // after all landmarks have been processed, update the state.
    this->x_t = this->x_pred;
    this->P_t = this->P_pred;
    /////////////// END OF EKF ITERATION ///////////////////
}

// return the state as a message.
ekf_pkg::EKFState EKF::getState() {
    ekf_pkg::EKFState stateMsg;
    // vehicle pose.
    stateMsg.x_v = this->x_t(0);
    stateMsg.y_v = this->x_t(1);
    stateMsg.yaw_v = this->x_t(2);
    // landmarks.
    stateMsg.M = this->M;
    std::vector<float> lm;
    for (int i=0; i<2*M; ++i) {
        lm.push_back(this->x_t(3+i));
    }
    stateMsg.landmarks = lm;
    // covariance. collapse all rows side by side into a vector.
    std::vector<float> p;
    for (int i=0; i<2*M+3; ++i) {
        for (int j=0; j<2*M+3; ++j) {
            p.push_back(this->P_t(i,j));
        }
    }
    stateMsg.P = p;
    return stateMsg;
}