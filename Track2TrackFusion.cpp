#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <iomanip>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

using namespace Eigen;
using namespace std;

class MultiSensorFusion {
private:
    // Simulation parameters
    double dt;
    double T_sim;
    int N_steps;
    
    // Random number generator
    std::random_device rd;
    std::mt19937 gen;
    std::normal_distribution<double> normal_dist;
    
    // Sensor models
    struct SensorModel {
        Matrix3d R; // Measurement noise covariance
        Vector3d bias;
    };
    
    SensorModel radar, video;
    
    // EKF matrices
    Matrix<double, 6, 6> F; // State transition matrix
    Matrix<double, 3, 6> H; // Measurement matrix
    Matrix<double, 6, 6> Q; // Process noise covariance
    
public:
    MultiSensorFusion(double dt_val = 0.1, double T_sim_val = 20.0) 
        : dt(dt_val), T_sim(T_sim_val), gen(rd()), normal_dist(0.0, 1.0) {
        
        N_steps = static_cast<int>(T_sim / dt);
        
        // Initialize sensor models
        initializeSensorModels();
        
        // Initialize system matrices
        initializeSystemMatrices();
    }
    
    void initializeSensorModels() {
        // Radar sensor (good in range, poor in angles)
        radar.R = Matrix3d::Zero();
        radar.R(0, 0) = 5.0 * 5.0;      // Range variance
        radar.R(1, 1) = 0.05 * 0.05;    // Azimuth variance
        radar.R(2, 2) = 0.05 * 0.05;    // Elevation variance
        radar.bias = Vector3d::Zero();
        
        // Video sensor (poor in range, good in angles)
        video.R = Matrix3d::Zero();
        video.R(0, 0) = 50.0 * 50.0;    // Range variance
        video.R(1, 1) = 0.005 * 0.005;  // Azimuth variance
        video.R(2, 2) = 0.005 * 0.005;  // Elevation variance
        video.bias = Vector3d::Zero();
    }
    
    void initializeSystemMatrices() {
        // State transition matrix (constant velocity model)
        F = Matrix<double, 6, 6>::Identity();
        F(0, 1) = dt; // range - range_rate
        F(2, 3) = dt; // azimuth - azimuth_rate
        F(4, 5) = dt; // elevation - elevation_rate
        
        // Measurement matrix (observe position only)
        H = Matrix<double, 3, 6>::Zero();
        H(0, 0) = 1; // range
        H(1, 2) = 1; // azimuth
        H(2, 4) = 1; // elevation
        
        // Process noise covariance
        double q_pos = 1.0;
        double q_vel = 0.1;
        
        Q = Matrix<double, 6, 6>::Zero();
        // Range block
        Q(0, 0) = q_pos * pow(dt, 4) / 4;
        Q(0, 1) = q_pos * pow(dt, 3) / 2;
        Q(1, 0) = q_pos * pow(dt, 3) / 2;
        Q(1, 1) = q_pos * dt * dt;
        
        // Azimuth block
        Q(2, 2) = q_vel * pow(dt, 4) / 4;
        Q(2, 3) = q_vel * pow(dt, 3) / 2;
        Q(3, 2) = q_vel * pow(dt, 3) / 2;
        Q(3, 3) = q_vel * dt * dt;
        
        // Elevation block
        Q(4, 4) = q_vel * pow(dt, 4) / 4;
        Q(4, 5) = q_vel * pow(dt, 3) / 2;
        Q(5, 4) = q_vel * pow(dt, 3) / 2;
        Q(5, 5) = q_vel * dt * dt;
    }
    
    Vector3d generateNoise(const Matrix3d& covariance) {
        // Generate multivariate normal noise using Cholesky decomposition
        LLT<Matrix3d> chol(covariance);
        Matrix3d L = chol.matrixL();
        
        Vector3d noise;
        for (int i = 0; i < 3; i++) {
            noise(i) = normal_dist(gen);
        }
        
        return L * noise;
    }
    
    pair<Vector<double, 6>, Matrix<double, 6, 6>> ekfUpdate(const Vector<double, 6>& x, const Matrix<double, 6, 6>& P, const Vector3d& z, const Matrix3d& R) {
        
        // Prediction step
        Vector<double, 6> x_pred = F * x;
        Matrix<double, 6, 6> P_pred = F * P * F.transpose() + Q;
        
        // Update step
        Vector3d y = z - H * x_pred; // Innovation
        Matrix3d S = H * P_pred * H.transpose() + R; // Innovation covariance
        Matrix<double, 6, 3> K = P_pred * H.transpose() * S.inverse(); // Kalman gain
        
        Vector<double, 6> x_new = x_pred + K * y;
        Matrix<double, 6, 6> P_new = (Matrix<double, 6, 6>::Identity() - K * H) * P_pred;
        
        return make_pair(x_new, P_new);
    }
    
    pair<Vector<double, 6>, Matrix<double, 6, 6>> covarianceIntersection(const Vector<double, 6>& x1, const Matrix<double, 6, 6>& P1, const Vector<double, 6>& x2, const Matrix<double, 6, 6>& P2) {
        
        // Calculate optimal weight
        double tr_P1_inv = P1.inverse().trace();
        double tr_P2_inv = P2.inverse().trace();
        double omega = tr_P2_inv / (tr_P1_inv + tr_P2_inv);
        
        // Fused covariance
        Matrix<double, 6, 6> P_fused_inv = omega * P1.inverse() + (1.0 - omega) * P2.inverse();
        Matrix<double, 6, 6> P_fused = P_fused_inv.inverse();
        
        // Fused state
        Vector<double, 6> x_fused = P_fused * (omega * P1.inverse() * x1 + (1.0 - omega) * P2.inverse() * x2);
        
        return make_pair(x_fused, P_fused);
    }
    
    Vector3d calculateRMSE(const vector<Vector3d>& estimates, const vector<Vector3d>& truth) {
        Vector3d rmse = Vector3d::Zero();
        int n = estimates.size();
        
        for (int i = 0; i < n; i++) {
            Vector3d error = estimates[i] - truth[i];
            rmse += error.cwiseProduct(error);
        }
        
        rmse /= n;
        return rmse.cwiseSqrt();
    }
    
    void saveResults(const string& filename, const vector<double>& time,
                    const vector<Vector3d>& true_traj,
                    const vector<Vector3d>& radar_est,
                    const vector<Vector3d>& video_est,
                    const vector<Vector3d>& fused_est) {
        
        ofstream file(filename);
        file << "Time,True_Range,True_Az,True_El,Radar_Range,Radar_Az,Radar_El,"
             << "Video_Range,Video_Az,Video_El,Fused_Range,Fused_Az,Fused_El\n";
        
        for (size_t i = 0; i < time.size(); i++) {
            file << fixed << setprecision(6) << time[i] << ","
                 << true_traj[i](0) << "," << true_traj[i](1) << "," << true_traj[i](2) << ","
                 << radar_est[i](0) << "," << radar_est[i](1) << "," << radar_est[i](2) << ","
                 << video_est[i](0) << "," << video_est[i](1) << "," << video_est[i](2) << ","
                 << fused_est[i](0) << "," << fused_est[i](1) << "," << fused_est[i](2) << "\n";
        }
        
        file.close();
    }
    
    void runSimulation() {
        cout << "Running multi-sensor fusion simulation..." << endl;
        
        // Generate true trajectory
        vector<double> time;
        vector<Vector3d> true_trajectory;
        
        for (int k = 0; k < N_steps; k++) {
            double t = k * dt;
            time.push_back(t);
            
            double true_range = 1000 + 50 * sin(0.2 * t) + 10 * t;
            double true_azimuth = 0.5 * sin(0.1 * t);
            double true_elevation = 0.2 * cos(0.15 * t);
            
            true_trajectory.push_back(Vector3d(true_range, true_azimuth, true_elevation));
        }
        
        // Initialize states and covariances
        Vector<double, 6> x_radar, x_video, x_fused;
        x_radar << true_trajectory[0](0), 0, true_trajectory[0](1), 0, true_trajectory[0](2), 0;
        x_video = x_radar;
        x_fused = x_radar;
        
        Matrix<double, 6, 6> P_radar, P_video, P_fused;
        P_radar = Matrix<double, 6, 6>::Identity();
        P_radar.diagonal() << 100, 10, 0.1, 0.01, 0.1, 0.01;
        
        P_video = Matrix<double, 6, 6>::Identity();
        P_video.diagonal() << 200, 20, 0.05, 0.005, 0.05, 0.005;
        
        P_fused = Matrix<double, 6, 6>::Identity();
        P_fused.diagonal() << 100, 10, 0.05, 0.005, 0.05, 0.005;
        
        // Storage for results
        vector<Vector3d> radar_estimates, video_estimates, fused_estimates;
        vector<Vector3d> radar_measurements, video_measurements;
        
        // Main simulation loop
        for (int k = 0; k < N_steps; k++) {
            // Generate sensor measurements
            Vector3d z_radar = true_trajectory[k] + generateNoise(radar.R);
            Vector3d z_video = true_trajectory[k] + generateNoise(video.R);
            
            radar_measurements.push_back(z_radar);
            video_measurements.push_back(z_video);
            
            // Individual sensor tracking
            auto radar_result = ekfUpdate(x_radar, P_radar, z_radar, radar.R);
            x_radar = radar_result.first;
            P_radar = radar_result.second;
            
            auto video_result = ekfUpdate(x_video, P_video, z_video, video.R);
            x_video = video_result.first;
            P_video = video_result.second;
            
            // Track-to-track fusion
            auto fusion_result = covarianceIntersection(x_radar, P_radar, x_video, P_video);
            x_fused = fusion_result.first;
            P_fused = fusion_result.second;
            
            // Store results (position only)
            radar_estimates.push_back(Vector3d(x_radar(0), x_radar(2), x_radar(4)));
            video_estimates.push_back(Vector3d(x_video(0), x_video(2), x_video(4)));
            fused_estimates.push_back(Vector3d(x_fused(0), x_fused(2), x_fused(4)));
        }
        
        // Performance analysis
        cout << "\nComputing performance metrics..." << endl;
        
        Vector3d radar_rmse = calculateRMSE(radar_estimates, true_trajectory);
        Vector3d video_rmse = calculateRMSE(video_estimates, true_trajectory);
        Vector3d fused_rmse = calculateRMSE(fused_estimates, true_trajectory);
        
        cout << fixed << setprecision(4);
        cout << "\nRMS Errors:" << endl;
        cout << "Radar - Range: " << radar_rmse(0) << " m, Azimuth: " << radar_rmse(1) 
             << " rad, Elevation: " << radar_rmse(2) << " rad" << endl;
        cout << "Video - Range: " << video_rmse(0) << " m, Azimuth: " << video_rmse(1) 
             << " rad, Elevation: " << video_rmse(2) << " rad" << endl;
        cout << "Fused - Range: " << fused_rmse(0) << " m, Azimuth: " << fused_rmse(1) 
             << " rad, Elevation: " << fused_rmse(2) << " rad" << endl;
        
        // Save results to CSV
        saveResults("tracking_results.csv", time, true_trajectory, 
                   radar_estimates, video_estimates, fused_estimates);
        
        cout << "\nResults saved to 'tracking_results.csv'" << endl;
        cout << "Simulation completed successfully!" << endl;
    }
};

int main() {
    try {
        MultiSensorFusion fusion(0.1, 20.0); // dt = 0.1s, T_sim = 20s
        fusion.runSimulation();
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return -1;
    }
    
    return 0;
}
