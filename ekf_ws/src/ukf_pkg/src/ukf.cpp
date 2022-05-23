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
    this->Wts = Eigen::VectorXd::Constant(7,(1-this->W_0)/6);
    this->Wts(0) = this->W_0;
    // set the expanding process noise.
    this->Q.setZero(3,3);
    this->Q(0,0) = this->V(0,0) * cos(this->x_t(2));
    this->Q(1,1) = this->V(0,0) * sin(this->x_t(2));
    this->Q(2,2) = this->V(1,1);
}

void UKF::init(float x_0, float y_0, float yaw_0) {
    // set starting vehicle pose.
    this->x_t << x_0, y_0, yaw_0;
}

void UKF::setTrueMap(std_msgs::Float32MultiArray::ConstPtr trueMapMsg) {
    // set the true map for localization-only mode.
    this->map = trueMapMsg->data;    
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
    int n = 3;

    // compute the sqrt cov term.
    this->sqtP = nearestSPD().sqrt() * std::sqrt(n/(1-this->W_0));

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
        // // get the true landmark position from the known map.
        // float x_l = this->map[id*3+1];
        // float y_l = this->map[id*3+2];

        // get meas estimate for all sigma points.
        this->X_zest.setZero(2,2*n+1);
        for (int i=0; i<2*n+1; ++i) {
            this->X_zest.col(i) = sensingModel(this->X_pred.col(i), id);
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

    }
    // after all landmarks have been processed, update the state.
    this->x_t = this->x_pred;
    this->P_t = this->P_pred;
    /////////////// END OF UKF ITERATION ///////////////////
}

Eigen::MatrixXd UKF::nearestSPD() {
    // find the nearest symmetric positive semidefinite matrix to P_t using Froebius Norm.
    // https://scicomp.stackexchange.com/questions/30631/how-to-find-the-nearest-a-near-positive-definite-from-a-given-matrix
    // first compute nearest symmetric matrix.
    this->Y = 0.5 * (this->P_t + this->P_t.transpose());
    // compute eigen decomposition of Y.
    Eigen::EigenSolver<Eigen::MatrixXd> es(this->Y);
    this->D = es.eigenvalues().real().asDiagonal();
    this->Qv = es.eigenvectors().real();
    // cap values to be nonnegative.
    // this->P_lower_bound.setZero(this->M*2+3, this->M*2+3);
    this->P_lower_bound.setOnes(3, 3);
    this->P_lower_bound *= 0.00000001;
    this->D = (this->D.array().max(this->P_lower_bound.array())).matrix();
    // compute nearest positive semidefinite matrix.
    return this->Qv * this->D * this->Qv.transpose();
}

Eigen::VectorXd UKF::motionModel(Eigen::VectorXd x, float u_d, float u_th) {
    Eigen::VectorXd x_pred = x;
    x_pred(0) = x(0)+(u_d+this->v_d)*std::cos(x(2));
    x_pred(1) = x(1)+(u_d+this->v_d)*std::sin(x(2));
    x_pred(2) = remainder(x(2) + u_th + this->v_th, 2*pi);
    return x_pred;
}

Eigen::VectorXd UKF::sensingModel(Eigen::VectorXd x, int lm_id) {
    // Generate the measurement we'd expect given x, a
    // current belief of the vehicle pose, and the known
    // location of the lm_id landmark on the true map.
    Eigen::VectorXd z_est = Eigen::VectorXd::Zero(2);
    z_est(0) = std::sqrt(std::pow(this->map[lm_id*3+1]-x(0), 2) + std::pow(this->map[lm_id*3+2]-x(1), 2)) + this->w_r;
    z_est(1) = std::atan2(this->map[lm_id*3+2]-x(1), this->map[lm_id*3+1]-x(0)) - x(2) + this->w_b;
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