#include "ukf_pkg/ukf.h"

// init the UKF.
UKF::UKF() {
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
    // set the sigma stuff to the right starting size.
    this->X.setZero(3,7);
    this->Wts = Eigen::VectorXd::Constant(7,(1-this->W_0)/6);
    this->Wts(0) = this->W_0;
    // set the expanding process noise.
    this->Q.setZero(3,3);
    this->Q(0,0) = this->V(0,0) * cos(this->x_t(2));
    this->Q(1,1) = this->V(0,0) * sin(this->x_t(2));
    this->Q(2,2) = this->V(1,1);
}

// perform a full iteration of the UKF for this timestep.
void UKF::update(geometry_msgs::Vector3::ConstPtr odomMsg, std_msgs::Float32MultiArray::ConstPtr lmMeasMsg) {
    // update timestep.
    this->timestep += 1;

    ////////////// PREDICTION STAGE /////////////////
    // extract odom.
    float u_d = odomMsg->x;
    float u_th = odomMsg->y;

    // get vector length.
    int n = this->M*2 + 3;
    // std::cout << "state length n=" << n << std::endl << std::flush;
    // see if a landmark was been added to the state last iteration.
    if (this->X.rows() != n) {
        // expand the size of X matrix.
        this->X.conservativeResize(n, 1+2*n);
        // recompute the weights vector.
        this->Wts.setOnes(1+2*n);
        this->Wts *= (1-this->W_0)/(2*n);
        this->Wts(0) = this->W_0;
        // expand the process noise.
        this->Q.setZero(n,n);
        this->Q(0,0) = this->V(0,0) * cos(this->x_t(2));
        this->Q(1,1) = this->V(0,0) * sin(this->x_t(2));
        this->Q(2,2) = this->V(1,1);
    }
    // get sqrt cov term.
    this->sqtP = this->P_t;
    std::cout << "TODO TAKE SQRT OF P_T" << std::endl << std::flush;
    // this->sqtP = this->P_t.sqrt() * std::sqrt(n/(1-this->W_0));
    // compute sigma points.
    this->X.col(0) = this->x_t;
    for (int i=1; i<=n; ++i) {
        this->X.col(i) = this->x_t + this->sqtP.col(i-1);
    }
    for (int i=1; i<=n; ++i) {
        this->X.col(i+n) = this->x_t - this->sqtP.col(i-1);
    }
    // std::cout << "Computed sigma pts:\n" << this->X << std::endl << std::flush;

    // propagate sigma vectors with motion model f.
    this->X_pred.setZero(n,2*n+1);
    for (int i=0; i<2*n+1; ++i) {
        this->X_pred.col(i) = motionModel(this->X.col(i), u_d, u_th);
    }
    // std::cout << "Propagated sigma pts:\n" << this->X_pred << std::endl << std::flush;
    // compute state mean prediction.
    this->x_pred.setZero(n);
    for (int i=0; i<2*n+1; ++i) {
        this->x_pred += this->Wts(i) * this->X_pred.col(i);
    }
    //compute state covariance prediction.
    this->P_pred.setZero(n, n);
    for (int i=0; i<2*n+1; ++i) {
        // std::cout << "i=" << i << std::endl << std::flush;
        // std::cout << "X_pred col=\n" << this->X_pred.col(i) << std::endl << std::flush;
        // std::cout << "Q=" << this->Q << std::endl << std::flush;
        this->P_pred += this->Wts(i) * (this->X_pred.col(i) - this->x_pred) * (this->X_pred.col(i) - this->x_pred).transpose() + this->Q;
    }
    // std::cout << "Predicted state x_pred:\n" << this->x_pred << "\nand P_pred:\n" << this->P_pred << std::endl << std::flush;

    ///////////////// UPDATE STAGE //////////////////
    std::vector<float> lm_meas = lmMeasMsg->data;
    int num_landmarks = (int) (lm_meas.size() / 3);
    // std::cout << "landmarks detected: " << num_landmarks << std::endl << std::flush;
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
        int lm_i = -1;
        for (int j=0; j<M; ++j) {
            if (this->lm_IDs[j] == id) {
                lm_i = j;
                break;
            }
        }
        if (lm_i != -1) { // the landmark was found.
            //////////// LANDMARK UPDATE ///////////
            // std::cout << "LM UPDATE" << std::endl << std::flush;
            // get index of this landmark in the state.
            lm_i = lm_i*2+3;

            // get meas estimate for all sigma points.
            this->X_zest.setZero(2,2*n+1);
            for (int i=0; i<2*n+1; ++i) {
                this->X_zest.col(i) = sensingModel(this->X_pred.col(i), lm_i);
            }
            // std::cout << "debug 1" << std::endl << std::flush;
            // compute overall measurement estimate.
            this->z_est.setZero(2);
            for (int i=0; i<2*n+1; ++i) {
                this->z_est += this->Wts(i) * this->X_zest.col(i);
            }
            // std::cout << "debug 2" << std::endl << std::flush;
            // compute innovation covariance.
            this->S.setZero(2, 2);
            for (int i=0; i<2*n+1; ++i) {
                // std::cout << "X_zest col=" << this->X_zest.col(i) << std::endl << std::flush;
                // std::cout << "z_est=" << this->z_est << std::endl << std::flush;
                // std::cout << "W=" << this->W << std::endl << std::flush;
                this->S += this->Wts(i) * (this->X_zest.col(i) - this->z_est) * (this->X_zest.col(i) - this->z_est).transpose() + this->W;
            }
            // std::cout << "debug 3" << std::endl << std::flush;
            // compute cross covariance b/w x_pred and z_est.
            this->C.setZero(n,2);
            for (int i=0; i<2*n+1; ++i) {
                this->C += this->Wts(i) * (this->X_pred.col(i) - this->x_pred) * (this->X_zest.col(i) - this->z_est).transpose();
            }
            // std::cout << "debug 4" << std::endl << std::flush;
            // compute kalman gain.
            this->K = this->C * this->S.inverse();

            // std::cout << "debug 5" << std::endl << std::flush;
            // compute the posterior distribution.
            this->z(0) = r; this->z(1) = b;
            this->x_pred = this->x_pred + this->K * (this->z - this->z_est);
            this->P_pred = this->P_pred - this->K * this->S * this->K.transpose();

            // std::cout << "lm update state:\n" << this->x_pred << "\nand cov:\n" << this->P_pred << std::endl << std::flush;

            // TODO UKF landmark insertion.
        } else { // this is a new landmark.
            // std::cout << "LM INSERTION" << std::endl << std::flush;
            /////////// LANDMARK INSERTION /////////
            // increment number of landmarks.
            // resize the state to fit the new landmark.
            this->x_pred.conservativeResize(n+2);
            this->x_pred(n) = this->x_pred(0) + r*cos(this->x_pred(2)+b);
            this->x_pred(n+1) = this->x_pred(1) + r*sin(this->x_pred(2)+b);
            // add landmark ID to the list.
            this->lm_IDs.push_back(id);

            // expand X_pred in case there's a re-detection also this timestep.
            this->X_pred.conservativeResize(n+2,2*n+5);
            this->P_pred.conservativeResize(n+2, n+2);

            // // compute insertion jacobian.
            // this->Y.setIdentity(3+2*this->M, 3+2*this->M);
            // // G_z submatrix.
            // this->Y(3+2*this->M-2,3+2*this->M-2) = cos(this->x_pred(2)+b);
            // this->Y(3+2*this->M-2,3+2*this->M-1) = -r*sin(this->x_pred(2)+b);
            // this->Y(3+2*this->M-1,3+2*this->M-2) = sin(this->x_pred(2)+b);
            // this->Y(3+2*this->M-1,3+2*this->M-1) = r*cos(this->x_pred(2)+b);
            // // G_x submatrix.
            // this->Y(3+2*this->M-2,0) = 1;
            // this->Y(3+2*this->M-2,1) = 0;
            // this->Y(3+2*this->M-2,2) = -r*sin(this->x_pred(2)+b);
            // this->Y(3+2*this->M-1,0) = 0;
            // this->Y(3+2*this->M-1,1) = 1;
            // this->Y(3+2*this->M-1,2) = r*cos(this->x_pred(2)+b);

            // // update covariance.
            // this->p_temp.setZero(3+2*this->M, 3+2*this->M);
            // this->p_temp.block(0,0,3+2*this->M-2,3+2*this->M-2) = this->P_pred;
            // this->p_temp.block(3+2*this->M-2,3+2*this->M-2,2,2) = this->W;
            
            // this->P_pred = this->Y * this->p_temp * this->Y.transpose();

            // update state size.
            this->M += 1; n += 2;
        }
    }
    // after all landmarks have been processed, update the state.
    this->x_t = this->x_pred;
    this->P_t = this->P_pred;
    /////////////// END OF UKF ITERATION ///////////////////
}

Eigen::VectorXd UKF::motionModel(Eigen::VectorXd x, float u_d, float u_th) {
    Eigen::VectorXd x_pred = x;
    x_pred(0) = x(0)+(u_d+this->v_d)*std::cos(x(2));
    x_pred(1) = x(1)+(u_d+this->v_d)*std::sin(x(2));
    x_pred(2) = remainder(x(2) + u_th + this->v_th, 2*pi);
    // x_pred.block(3,0,this->M*2,1) = x.block(3,0,this->M*2,1);
    return x_pred;
}

Eigen::VectorXd UKF::sensingModel(Eigen::VectorXd x, int lm_i) {
    Eigen::VectorXd z_est = Eigen::VectorXd::Zero(2);
    z_est(0) = std::sqrt(std::pow(x(lm_i)-x(0), 2) + std::pow(x(lm_i+1)-x(1), 2)) + this->w_r;
    // z_est(0) = 3; // TODO REMOVE THIS JUST FOR DEBUG.
    z_est(1) = std::atan2(x(lm_i+1)-x(1), x(lm_i)-x(0)) - x(2) + this->w_b;
    // std::cout << "sensingModel got z_est=\n" << z_est << std::endl << std::flush;
    return z_est;
}

// return the state as a message.
ukf_pkg::UKFState UKF::getState() {
    ukf_pkg::UKFState stateMsg;
    // timestep.
    stateMsg.timestep = this->timestep;
    // vehicle pose.
    stateMsg.x_v = this->x_t(0);
    stateMsg.y_v = this->x_t(1);
    stateMsg.yaw_v = this->x_t(2);
    // landmarks.
    stateMsg.M = this->M;
    std::vector<float> lm;
    for (int i=0; i<M; ++i) {
        lm.push_back((float) this->lm_IDs[i]);
        lm.push_back(this->x_t(3+i*2));
        lm.push_back(this->x_t(3+i*2+1));
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