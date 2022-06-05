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
    this->x_pred.setZero(3);
    this->P_t.setIdentity(3,3);
    this->P_t(0,0) = 0.01 * 0.01;
    this->P_t(1,1) = 0.01 * 0.01;
    this->P_t(2,2) = 0.005 * 0.005;
    this->P_pred.setIdentity(3,3);
    this->P_pred(0,0) = 0.01 * 0.01;
    this->P_pred(1,1) = 0.01 * 0.01;
    this->P_pred(2,2) = 0.005 * 0.005;
    // set the sigma stuff to the right starting size.
    this->X.setZero(3,7);
    // set the expanding process noise.
    this->Q.setZero(3,3);
    this->Q(0,0) = this->V(0,0) * cos(this->x_t(2));
    this->Q(1,1) = this->V(0,0) * sin(this->x_t(2));
    this->Q(2,2) = this->V(1,1);
}

void UKF::init(float x_0, float y_0, float yaw_0, float W_0) {
    // set starting vehicle pose.
    this->x_t << x_0, y_0, yaw_0;
    // set mean sigma pt weight.
    this->W_0 = W_0;
    this->Wts = Eigen::VectorXd::Constant(7,(1-this->W_0)/6);
    this->Wts(0) = this->W_0;
    // set initialized flag.
    this->isInit = true;
}

void UKF::setTrueMap(std_msgs::Float32MultiArray::ConstPtr trueMapMsg) {
    // set the true map for localization-only mode.
    this->map = trueMapMsg->data;    
}

data_pkg::UKFState UKF::getState() {
    // state length for convenience.
    int n = 2 * this->M + 3;
    // return the state as a message.
    data_pkg::UKFState stateMsg;
    // timestep.
    stateMsg.timestep = this->timestep;
    // vehicle pose.
    stateMsg.x_v = this->x_t(0);
    stateMsg.y_v = this->x_t(1);
    stateMsg.yaw_v = this->x_t(2);
    // landmarks.
    stateMsg.M = this->M;
    std::vector<float> lm;
    for (int i=0; i<this->M; ++i) {
        lm.push_back((float) this->lm_IDs[i]);
        lm.push_back(this->x_t(3+i*2));
        lm.push_back(this->x_t(3+i*2+1));
    }
    stateMsg.landmarks = lm;
    // covariance. collapse all rows side by side into a vector.
    std::vector<float> p;
    for (int i=0; i<n; ++i) {
        for (int j=0; j<n; ++j) {
            p.push_back(this->P_t(i,j));
        }
    }
    stateMsg.P = p;
    // sigma points. also collapse into a vector.
    std::vector<float> sigma_pts;
    std::vector<float> sigma_pts_propagated;
    for (int j=0; j<2*n+1; ++j) {
        for (int i=0; i<n; ++i) {
            sigma_pts.push_back(this->X(i,j));
            sigma_pts_propagated.push_back(this->X_pred(i,j));
        }
    }
    stateMsg.X = sigma_pts;
    stateMsg.X_pred = sigma_pts_propagated;

    return stateMsg;
}

Eigen::MatrixXd UKF::nearestSPD() {
    // find the nearest symmetric positive semidefinite matrix to P_t using Froebius Norm.
    // https://scicomp.stackexchange.com/questions/30631/how-to-find-the-nearest-a-near-positive-definite-from-a-given-matrix
    // first compute nearest symmetric matrix.
    this->Y = 0.5 * (this->P_t + this->P_t.transpose());
    // multiply by inner coefficient for UKF.
    this->Y *= (2*this->M+3)/(1-this->W_0);
    // compute eigen decomposition of Y.
    Eigen::EigenSolver<Eigen::MatrixXd> es(this->Y);
    this->D = es.eigenvalues().real().asDiagonal();
    this->Qv = es.eigenvectors().real();
    // cap values to be nonnegative.
    // this->P_lower_bound.setZero(this->M*2+3, this->M*2+3);
    this->P_lower_bound.setOnes(this->M*2+3, this->M*2+3);
    // this->P_lower_bound *= 0.0;
    // this->P_lower_bound *= 0.0001;
    this->P_lower_bound *= 0.00000001;
    this->D = (this->D.array().max(this->P_lower_bound.array())).matrix();
    // compute nearest positive semidefinite matrix.
    // return ((this->Qv * this->D * this->Qv.transpose()).array().max(this->P_lower_bound.array())).matrix();
    return this->Qv * this->D * this->Qv.transpose();
}

Eigen::VectorXd UKF::motionModel(Eigen::VectorXd x, float u_d, float u_th) {
    // predict to propagate state mean forward one timestep using commanded odometry.
    Eigen::VectorXd x_pred = x;
    x_pred(0) = x(0)+(u_d+this->v_d)*std::cos(x(2));
    x_pred(1) = x(1)+(u_d+this->v_d)*std::sin(x(2));
    x_pred(2) = remainder(x(2) + u_th + this->v_th, 2*pi);
    return x_pred;
}

Eigen::VectorXd UKF::localizationSensingModel(Eigen::VectorXd x, int lm_id) {
    // Generate the measurement we'd expect given x, a
    // current belief of the vehicle pose, and the known
    // location of the lm_id landmark on the true map.
    Eigen::VectorXd z_est = Eigen::VectorXd::Zero(2);
    z_est(0) = std::sqrt(std::pow(this->map[lm_id*3+1]-x(0), 2) + std::pow(this->map[lm_id*3+2]-x(1), 2)) + this->w_r;
    z_est(1) = std::atan2(this->map[lm_id*3+2]-x(1), this->map[lm_id*3+1]-x(0)) - x(2) + this->w_b;
    // cap bearing within (-pi, pi).
    z_est(1) = remainder(z_est(1), 2*pi);
    return z_est;
}

void UKF::localizationUpdate(data_pkg::Command::ConstPtr cmdMsg, std_msgs::Float32MultiArray::ConstPtr lmMeasMsg) {
    // perform a full iteration of UKF-Localization for this timestep.
    // update timestep.
    this->timestep += 1;

    // update process noise with new pos est.
    this->Q(0,0) = this->V(0,0) * cos(this->x_t(2));
    this->Q(1,1) = this->V(0,0) * sin(this->x_t(2));

    ////////////// PREDICTION STAGE /////////////////
    // extract odom.
    float u_d = cmdMsg->fwd;
    float u_th = cmdMsg->ang;

    // get vector length.
    int n = 3;

    // compute the sqrt cov term.
    std::cout << "\nnearestSPD:\n" << nearestSPD() << std::endl << std::flush;
    this->sqtP = nearestSPD().sqrt();
    std::cout << "sqtP:\n" << this->sqtP << std::endl << std::flush;

    // compute sigma points.
    this->X.col(0) = this->x_t;
    for (int i=1; i<=n; ++i) {
        this->X.col(i) = this->x_t + this->sqtP.col(i-1);
    }
    for (int i=1; i<=n; ++i) {
        this->X.col(i+n) = this->x_t - this->sqtP.col(i-1);
    }
    // cap all headings within (-pi, pi).
    for (int i=0; i<2*n+1; ++i) {
        this->X(2,i) = remainder(this->X(2,i), 2*pi);
    }

    // propagate sigma vectors with motion model f.
    this->X_pred.setZero(n,2*n+1);
    for (int i=0; i<2*n+1; ++i) {
        this->X_pred.col(i) = motionModel(this->X.col(i), u_d, u_th);
    }
    // compute state mean prediction.
    this->x_pred.setZero(n);
    this->x_yaw_components.setZero(2);
    for (int i=0; i<2*n+1; ++i) {
        this->x_pred += this->Wts(i) * this->X_pred.col(i);
        // convert angles to complex numbers to average them correctly (assume hypoteneuse = 1).
        this->x_yaw_components(0) += this->Wts(i) * cos(this->X_pred.col(i)(2)); // real component.
        this->x_yaw_components(1) += this->Wts(i) * sin(this->X_pred.col(i)(2)); // imaginary component.
    }
    // convert averaged heading back from complex to angle. also cap w/in (-pi, pi).
    this->x_pred(2) = remainder(atan2(this->x_yaw_components(1), this->x_yaw_components(0)), 2*pi);

    //compute state covariance prediction.
    this->P_pred.setZero(n, n);
    for (int i=0; i<2*n+1; ++i) {
        this->P_pred += this->Wts(i) * (this->X_pred.col(i) - this->x_pred) * (this->X_pred.col(i) - this->x_pred).transpose();
    }
    // add process noise cov.
    this->P_pred += this->Q;

    ///////////////// UPDATE STAGE //////////////////
    std::vector<float> lm_meas = lmMeasMsg->data;
    int num_landmarks = (int) (lm_meas.size() / 3);
    // we must update for each detection individually.
    // if there was no detection, this loop is skipped.
    for (int l=0; l<num_landmarks; ++l) {
        // extract the landmark details.
        int id = (int) lm_meas[l*3];
        float r = lm_meas[l*3+1];
        float b = lm_meas[l*3+2];
        // // get the true landmark position from the known map.
        // float x_l = this->map[id*3+1];
        // float y_l = this->map[id*3+2];

        // get meas estimate for all sigma points.
        this->X_zest.setZero(2,2*n+1);
        for (int i=0; i<2*n+1; ++i) {
            this->X_zest.col(i) = localizationSensingModel(this->X_pred.col(i), id);
        }
        // compute overall measurement estimate.
        this->z_est.setZero(2);
        for (int i=0; i<this->Wts.rows(); ++i) { //2*n+1
            this->z_est += this->Wts(i) * this->X_zest.col(i);
        }
        // cap bearing within (-pi, pi).
        this->z_est(1) = remainder(this->z_est(1), 2*pi);
        // compute innovation covariance.
        this->S.setZero(2, 2);
        for (int i=0; i<this->Wts.rows(); ++i) { //2*n+1
            this->S += this->Wts(i) * (this->X_zest.col(i) - this->z_est) * (this->X_zest.col(i) - this->z_est).transpose();
        }
        // add sensing noise cov.
        this->S += this->W;
        // compute cross covariance b/w x_pred and z_est.
        this->C.setZero(n,2);
        for (int i=0; i<this->Wts.rows(); ++i) { // 2*n+1
            this->C += this->Wts(i) * (this->X_pred.col(i) - this->x_pred) * (this->X_zest.col(i) - this->z_est).transpose();
        }
        // compute kalman gain.
        this->K = this->C * this->S.inverse();

        // compute the posterior distribution.
        this->z(0) = r; this->z(1) = b;
        this->x_pred = this->x_pred + this->K * (this->z - this->z_est);
        // cap heading within (-pi, pi).
        this->x_pred(2) = remainder(this->x_pred(2), 2*pi);
        this->P_pred = this->P_pred - this->K * this->S * this->K.transpose();

    }
    // after all landmarks have been processed, update the state.
    this->x_t = this->x_pred;
    this->P_t = this->P_pred;
    /////////////// END OF UKF ITERATION ///////////////////
}

Eigen::VectorXd UKF::slamSensingModel(Eigen::VectorXd x, int lm_i) {
    // Generate measurement we expect from current estimates
    // of veh pose and landmark position.
    Eigen::VectorXd z_est = Eigen::VectorXd::Zero(2);
    z_est(0) = std::sqrt(std::pow(x(lm_i)-x(0), 2) + std::pow(x(lm_i+1)-x(1), 2)) + this->w_r;
    z_est(1) = std::atan2(x(lm_i+1)-x(1), x(lm_i)-x(0)) - x(2) + this->w_b;
    return z_est;
}

void UKF::slamUpdate(data_pkg::Command::ConstPtr cmdMsg, std_msgs::Float32MultiArray::ConstPtr lmMeasMsg) {
    // perform a full iteration of UKF-SLAM for this timestep.
    // update timestep.
    this->timestep += 1;

    ////////////// PREDICTION STAGE /////////////////
    // extract odom.
    float u_d = cmdMsg->fwd;
    float u_th = cmdMsg->ang;

    // get vector length.
    int n = this->M*2 + 3;
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

    // compute the sqrt cov term.
    this->sqtP = nearestSPD().sqrt();

    // compute sigma points.
    this->X.col(0) = this->x_t;
    for (int i=1; i<=n; ++i) {
        this->X.col(i) = this->x_t + this->sqtP.col(i-1);
    }
    for (int i=1; i<=n; ++i) {
        this->X.col(i+n) = this->x_t - this->sqtP.col(i-1);
    }

    // propagate sigma vectors with motion model f.
    this->X_pred.setZero(n,2*n+1);
    for (int i=0; i<2*n+1; ++i) {
        this->X_pred.col(i) = motionModel(this->X.col(i), u_d, u_th);
    }
    // compute state mean prediction.
    this->x_pred.setZero(n);
    for (int i=0; i<2*n+1; ++i) {
        this->x_pred += this->Wts(i) * this->X_pred.col(i);
    }
    //compute state covariance prediction.
    this->P_pred.setZero(n, n);
    for (int i=0; i<2*n+1; ++i) {
        this->P_pred += this->Wts(i) * (this->X_pred.col(i) - this->x_pred) * (this->X_pred.col(i) - this->x_pred).transpose();
    }
    // add process noise cov.
    this->P_pred += this->Q;

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
        int lm_i = -1;
        for (int j=0; j<M; ++j) {
            if (this->lm_IDs[j] == id) {
                lm_i = j;
                break;
            }
        }
        if (lm_i != -1) { // the landmark was found.
            //////////// LANDMARK UPDATE ///////////
            // get index of this landmark in the state.
            lm_i = lm_i*2+3;

            // get meas estimate for all sigma points.
            this->X_zest.setZero(2,2*n+1);
            for (int i=0; i<2*n+1; ++i) {
                this->X_zest.col(i) = slamSensingModel(this->X_pred.col(i), lm_i);
            }
            // compute overall measurement estimate.
            this->z_est.setZero(2);
            for (int i=0; i<this->Wts.rows(); ++i) { //2*n+1
                this->z_est += this->Wts(i) * this->X_zest.col(i);
            }
            // compute innovation covariance.
            this->S.setZero(2, 2);
            for (int i=0; i<this->Wts.rows(); ++i) { //2*n+1
                this->S += this->Wts(i) * (this->X_zest.col(i) - this->z_est) * (this->X_zest.col(i) - this->z_est).transpose();
            }
            // add sensing noise cov.
            this->S += this->W;
            // compute cross covariance b/w x_pred and z_est.
            this->C.setZero(n,2);
            for (int i=0; i<this->Wts.rows(); ++i) { // 2*n+1
                this->C += this->Wts(i) * (this->X_pred.col(i) - this->x_pred) * (this->X_zest.col(i) - this->z_est).transpose();
            }
            // compute kalman gain.
            this->K = this->C * this->S.inverse();

            // compute the posterior distribution.
            this->z(0) = r; this->z(1) = b;
            this->x_pred = this->x_pred + this->K * (this->z - this->z_est);
            this->P_pred = this->P_pred - this->K * this->S * this->K.transpose();


            // TODO UKF landmark insertion.
        } else { // this is a new landmark.
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
            
            // expand state cov matrix.
            // fill new space with zeros and W at bottom right.
            this->p_temp.setIdentity(n+2,n+2);
            this->p_temp.block(0,0,n,n) = this->P_pred;
            this->p_temp.block(n,n,2,2) = this->W;
            this->P_pred = this->p_temp;

            // update state size.
            this->M += 1; n += 2;
        }
    }
    // after all landmarks have been processed, update the state.
    this->x_t = this->x_pred;
    this->P_t = this->P_pred;
    /////////////// END OF UKF ITERATION ///////////////////
}

