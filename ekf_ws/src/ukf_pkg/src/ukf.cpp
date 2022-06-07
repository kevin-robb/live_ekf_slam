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

void UKF::init(float x_0, float y_0, float yaw_0, float W_0, bool ukfSlamMode) {
    // set starting vehicle pose.
    this->x_t << x_0, y_0, yaw_0;
    // set mean sigma pt weight.
    this->W_0 = W_0;
    this->Wts = Eigen::VectorXd::Constant(7,(1-this->W_0)/6);
    this->Wts(0) = this->W_0;
    // set operating mode. (loc or slam)
    this->ukfSlamMode = ukfSlamMode;
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
    // std::vector<float> sigma_pts_propagated;
    for (int j=0; j<2*n+1; ++j) {
        for (int i=0; i<n; ++i) {
            sigma_pts.push_back(this->X(i,j));
            // sigma_pts_propagated.push_back(this->X_pred(i,j));
        }
    }
    stateMsg.X = sigma_pts;
    // stateMsg.X_pred = sigma_pts_propagated;

    return stateMsg;
}

Eigen::MatrixXd UKF::nearestSPD() {
    // find the nearest symmetric positive (semi)definite matrix to P_t using Froebius Norm.
    // https://scicomp.stackexchange.com/questions/30631/how-to-find-the-nearest-a-near-positive-definite-from-a-given-matrix
    // https://stackoverflow.com/questions/61639182/find-the-nearest-postive-definte-matrix-with-eigen

    // first compute nearest symmetric matrix.
    this->Y = 0.5 * (this->P_t + this->P_t.transpose());
    // multiply by inner coefficient for UKF.
    this->Y *= (2*this->M+3)/(1-this->W_0);
    // compute eigen decomposition of Y.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(this->Y);
    this->D = solver.eigenvalues();
    this->Qv = solver.eigenvectors();
    // cap eigenvalues to be strictly positive.
    this->Dplus = this->D.cwiseMax(0.00000001);
    // compute nearest positive semidefinite matrix.
    return this->Qv * this->Dplus.asDiagonal() * this->Qv.transpose();
}

Eigen::VectorXd UKF::motionModel(Eigen::VectorXd x, float u_d, float u_th) {
    // predict to propagate state mean forward one timestep using commanded odometry.
    Eigen::VectorXd x_pred = x;
    x_pred(0) = x(0)+(u_d+this->v_d)*std::cos(x(2));
    x_pred(1) = x(1)+(u_d+this->v_d)*std::sin(x(2));
    x_pred(2) = remainder(x(2) + u_th + this->v_th, 2*pi);
    return x_pred;
}

Eigen::VectorXd UKF::sensingModel(Eigen::VectorXd x, int lm_i) {
    Eigen::VectorXd z_est = Eigen::VectorXd::Zero(2);
    if (this->ukfSlamMode) {
        // SLAM mode. lm_i is the index of this landmark in our state.
        // Generate measurement we expect from current estimates
        // of veh pose and landmark position.
        z_est(0) = std::sqrt(std::pow(x(lm_i)-x(0), 2) + std::pow(x(lm_i+1)-x(1), 2)) + this->w_r;
        z_est(1) = std::atan2(x(lm_i+1)-x(1), x(lm_i)-x(0)) - x(2) + this->w_b;
    } else {
        // localization-only, so we have access to the true map.
        // lm_i is the known ID of the landmark.
        // Generate the measurement we'd expect given x, a
        // current belief of the vehicle pose, and the known
        // location of the lm_i landmark on the true map.
        z_est(0) = std::sqrt(std::pow(this->map[lm_i*3+1]-x(0), 2) + std::pow(this->map[lm_i*3+2]-x(1), 2)) + this->w_r;
        z_est(1) = std::atan2(this->map[lm_i*3+2]-x(1), this->map[lm_i*3+1]-x(0)) - x(2) + this->w_b;
    }
    // cap bearing within (-pi, pi).
    // z_est(1) = remainder(z_est(1), 2*pi);
    return z_est;
}


void UKF::ukfIterate(data_pkg::Command::ConstPtr cmdMsg, std_msgs::Float32MultiArray::ConstPtr lmMeasMsg) {
    // perform a full iteration of UKF-SLAM for this timestep.
    // update timestep count.
    this->timestep += 1;

    // get vector length.
    int n = this->M*2 + 3;
    // expand the sizes of everything if the state length has grown.
    if (n != this->X.rows()) {
        // expand the size of sigma pts matrices.
        this->X.conservativeResize(n,1+2*n);
        this->X_pred.conservativeResize(n,1+2*n);
        // recompute the weights vector at new size.
        this->Wts.setOnes(1+2*n);
        this->Wts *= (1-this->W_0)/(2*n);
        this->Wts(0) = this->W_0;
        // expand the process noise with zeros.
        // the elements are set at the start of ukfIterate; only need to update its size.
        this->Q.setZero(n,n);
        this->Q(2,2) = this->V(1,1);
    }
    
    // update the process noise components for current yaw.
    this->Q(0,0) = this->V(0,0) * cos(this->x_t(2));
    this->Q(1,1) = this->V(0,0) * sin(this->x_t(2));

    ////////////// PREDICTION STAGE /////////////////
    predictionStage(cmdMsg);

    ///////////////// UPDATE STAGE //////////////////
    updateStage(lmMeasMsg);

    /////////////// END OF UKF ITERATION ///////////////////
}


void UKF::predictionStage(data_pkg::Command::ConstPtr cmdMsg) {
    ////////////// PREDICTION STAGE /////////////////
    // extract odom.
    float u_d = cmdMsg->fwd;
    float u_th = cmdMsg->ang;

    // get vector length.
    int n = this->M*2 + 3;

    // compute the sqrt cov term.
    try {
        this->sqtP = nearestSPD().sqrt();
    } catch (...) {
        std::cout << "\nsqtP failed with nearestSPD:\n" << nearestSPD() << std::endl << std::flush;
    }
    
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
    this->complex_angle.setZero(2);
    for (int i=0; i<2*n+1; ++i) {
        this->x_pred += this->Wts(i) * this->X_pred.col(i);
        // convert angles to vectors in complex plane to average them correctly (assume unit circle).
        this->complex_angle(0) += this->Wts(i) * cos(this->X_pred.col(i)(2)); // real component.
        this->complex_angle(1) += this->Wts(i) * sin(this->X_pred.col(i)(2)); // imaginary component.
    }
    // convert averaged heading back from complex to angle. also cap w/in (-pi, pi).
    this->x_pred(2) = remainder(atan2(this->complex_angle(1), this->complex_angle(0)), 2*pi);

    //compute state covariance prediction.
    this->P_pred.setZero(n, n);
    for (int i=0; i<2*n+1; ++i) {
        this->P_pred += this->Wts(i) * (this->X_pred.col(i) - this->x_pred) * (this->X_pred.col(i) - this->x_pred).transpose();
    }
    // add process noise cov.
    this->P_pred += this->Q;
}


void UKF::updateStage(std_msgs::Float32MultiArray::ConstPtr lmMeasMsg) {
    ///////////////// UPDATE STAGE //////////////////
    // get vector length.
    int n = this->M*2 + 3;

    // extract landmark measurements.
    std::vector<float> lm_meas = lmMeasMsg->data;
    int num_landmarks = (int) (lm_meas.size() / 3);
    // some extra logic to make sure we perform all landmark updates FIRST,
    // and all landmark insertions LAST. This makes all the code simpler without changing functionality.
    std::vector<int> new_landmark_indexes;
    // we must update for each detection individually.
    // if there was no detection, this loop is skipped.
    for (int l=0; l<num_landmarks; ++l) {
        // extract the landmark details.
        int id = (int) lm_meas[l*3];
        float r = lm_meas[l*3+1];
        float b = lm_meas[l*3+2];
        int lm_i = -1;
        if (this->ukfSlamMode) {
            // check if we have seen this landmark before.
            for (int j=0; j<M; ++j) {
                if (this->lm_IDs[j] == id) {
                    lm_i = j;
                    break;
                }
            }
        }
        // if it's a new landmark, wait to handle it later.
        if (lm_i == -1) {
            new_landmark_indexes.push_back(l);
        } else {
            landmarkUpdate(lm_i, id, r, b);
        }
    }
    // now do all the landmark insertions.
    for (int i=0; i<new_landmark_indexes.size(); ++i) {
        int l = new_landmark_indexes[i];
        // extract the landmark details.
        int id = (int) lm_meas[l*3];
        float r = lm_meas[l*3+1];
        float b = lm_meas[l*3+2];
        // insert it into the state.
        landmarkInsertion(id, r, b);
    }
    // after all landmarks have been processed, update the state.
    this->x_t = this->x_pred;
    this->P_t = this->P_pred;
}

void UKF::landmarkUpdate(int lm_i, int id, float r, float b) {
    // localization mode, or the landmark was found in the state.
    //////////// LANDMARK UPDATE ///////////
    if (this->ukfSlamMode) {
        // get index of this landmark in the state.
        lm_i = lm_i*2+3;
    } else {
        // get landmark's unique identifier.
        lm_i = id;
    }
    
    // get meas estimate for all sigma points.
    this->X_zest.setZero(2,2*this->Wts.rows()+1);
    for (int i=0; i<2*this->Wts.rows()+1; ++i) {
        this->X_zest.col(i) = sensingModel(this->X_pred.col(i), lm_i);
    }
    // compute overall measurement estimate.
    this->z_est.setZero(2);
    this->complex_angle.setZero(2);
    for (int i=0; i<this->Wts.rows(); ++i) { //2*n+1
        this->z_est(0) += this->Wts(i) * this->X_zest.col(i)(0);
        // convert angles to vectors in complex plane to average them correctly (assume unit circle).
        this->complex_angle(0) += this->Wts(i) * cos(this->X_zest.col(i)(1)); // real component.
        this->complex_angle(1) += this->Wts(i) * sin(this->X_zest.col(i)(1)); // imaginary component.
    }
    // convert averaged heading back from complex to angle. also cap w/in (-pi, pi).
    this->z_est(1) = remainder(atan2(this->complex_angle(1), this->complex_angle(0)), 2*pi);
    
    // compute innovation covariance.
    this->S.setZero(2, 2);
    for (int i=0; i<this->Wts.rows(); ++i) { //2*n+1
        this->diff = (this->X_zest.col(i) - this->z_est);
        // keep angle in range.
        this->diff(1) = remainder(this->diff(1), 2*pi);
        // add innovation covariance contribution.
        this->S += this->Wts(i) * this->diff * this->diff.transpose();
    }
    // add sensing noise cov.
    this->S += this->W;
    // compute cross covariance b/w x_pred and z_est.
    this->C.setZero(this->Wts.rows(),2);
    for (int i=0; i<this->Wts.rows(); ++i) { //2*n+1
        this->diff = (this->X_pred.col(i) - this->x_pred);
        // keep angles in range.
        this->diff(2) = remainder(this->diff(2), 2*pi);
        // std::cout << "diff: " << this->diff << "\n" << std::flush;
        this->diff2 = (this->X_zest.col(i) - this->z_est);
        this->diff2(1) = remainder(this->diff2(1), 2*pi);
        // std::cout << "diff2: " << this->diff2 << "\n" << std::flush;
        // add cross covariance contribution.
        this->C += this->Wts(i) * this->diff * this->diff2.transpose();
    }
    // compute kalman gain.
    this->K = this->C * this->S.inverse();

    // compute the posterior distribution.
    this->z(0) = r; this->z(1) = b;
    this->innovation = (this->z - this->z_est);
    this->innovation(1) = remainder(this->innovation(1), 2*pi);
    this->x_pred = this->x_pred + this->K * this->innovation;
    // cap heading within (-pi, pi).
    this->x_pred(2) = remainder(this->x_pred(2), 2*pi);
    this->P_pred = this->P_pred - this->K * this->S * this->K.transpose();
}

void UKF::landmarkInsertion(int id, float r, float b) {
    // slam mode, and this is a new landmark.
    /////////// LANDMARK INSERTION /////////
    int n = this->M*2+3;
    // resize the state to fit the new landmark.
    this->x_pred.conservativeResize(n+2);
    this->x_pred(n) = this->x_pred(0) + r*cos(this->x_pred(2)+b);
    this->x_pred(n+1) = this->x_pred(1) + r*sin(this->x_pred(2)+b);
    // add landmark ID to the list.
    this->lm_IDs.push_back(id);

    // expand state cov matrix.
    // fill new space with zeros and W at bottom right.
    this->p_temp.setIdentity(n+2,n+2);
    this->p_temp.block(0,0,n,n) = this->P_pred;
    this->p_temp.block(n,n,2,2) = this->W;
    this->P_pred = this->p_temp;

    // increment number of landmarks and update state size.
    this->M += 1; //n += 2;
}