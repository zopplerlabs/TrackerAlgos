import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv
from scipy.stats import chi2
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

@dataclass
class Track:
    """Represents a radar track with state and covariance"""
    id: int
    sensor_id: int
    timestamp: float
    state: np.ndarray  # [x, y, vx, vy]
    covariance: np.ndarray  # 4x4 covariance matrix
    confidence: float

class TrackToTrackFusion:
    """
    Track-to-Track Fusion system for defense radar
    Implements multiple fusion algorithms including:
    - Covariance Intersection (CI)
    - Simple Convex Combination (SCC)
    - Kalman Filter based fusion
    """
    
    def __init__(self, gate_threshold: float = 9.21):  # Chi-square threshold for 95% confidence
        self.gate_threshold = gate_threshold
        self.fusion_history = []
        
    def mahalanobis_distance(self, track1: Track, track2: Track) -> float:
        """
        Calculate Mahalanobis distance between two tracks
        Used for gating and association
        """
        state_diff = track1.state - track2.state
        combined_cov = track1.covariance + track2.covariance
        
        try:
            inv_cov = inv(combined_cov)
            distance = np.sqrt(state_diff.T @ inv_cov @ state_diff)
            return distance
        except np.linalg.LinAlgError:
            return float('inf')
    
    def chi_square_gate(self, track1: Track, track2: Track) -> bool:
        """
        Chi-square gating test for track association
        Returns True if tracks should be associated
        """
        state_diff = track1.state - track2.state
        combined_cov = track1.covariance + track2.covariance
        
        try:
            inv_cov = inv(combined_cov)
            chi_square_stat = state_diff.T @ inv_cov @ state_diff
            return chi_square_stat <= self.gate_threshold
        except np.linalg.LinAlgError:
            return False
    
    def covariance_intersection(self, track1: Track, track2: Track, omega: float = None) -> Track:
        """
        Covariance Intersection fusion algorithm
        Handles unknown correlations between tracks
        """
        if omega is None:
            # Optimal omega calculation
            P1_inv = inv(track1.covariance)
            P2_inv = inv(track2.covariance)
            
            # Simplified omega calculation (can be optimized further)
            trace1 = np.trace(P1_inv)
            trace2 = np.trace(P2_inv)
            omega = trace1 / (trace1 + trace2)
        
        # Fused covariance
        P1_inv = inv(track1.covariance)
        P2_inv = inv(track2.covariance)
        P_fused_inv = omega * P1_inv + (1 - omega) * P2_inv
        P_fused = inv(P_fused_inv)
        
        # Fused state
        x_fused = P_fused @ (omega * P1_inv @ track1.state + 
                            (1 - omega) * P2_inv @ track2.state)
        
        # Create fused track
        fused_track = Track(
            id=max(track1.id, track2.id),
            sensor_id=-1,  # Indicates fused track
            timestamp=max(track1.timestamp, track2.timestamp),
            state=x_fused,
            covariance=P_fused,
            confidence=min(track1.confidence + track2.confidence, 1.0)
        )
        
        return fused_track
    
    def simple_convex_combination(self, track1: Track, track2: Track) -> Track:
        """
        Simple Convex Combination fusion
        Weight based on inverse covariance determinant
        """
        det1 = np.linalg.det(track1.covariance)
        det2 = np.linalg.det(track2.covariance)
        
        # Weights based on inverse determinant (smaller covariance gets higher weight)
        w1 = (1/det1) / (1/det1 + 1/det2)
        w2 = 1 - w1
        
        # Fused state
        x_fused = w1 * track1.state + w2 * track2.state
        
        # Fused covariance (simplified)
        P_fused = w1**2 * track1.covariance + w2**2 * track2.covariance
        
        fused_track = Track(
            id=max(track1.id, track2.id),
            sensor_id=-1,
            timestamp=max(track1.timestamp, track2.timestamp),
            state=x_fused,
            covariance=P_fused,
            confidence=w1 * track1.confidence + w2 * track2.confidence
        )
        
        return fused_track
    
    def kalman_fusion(self, track1: Track, track2: Track) -> Track:
        """
        Kalman Filter based fusion assuming independent tracks
        """
        # Measurement model (identity matrix for direct state measurement)
        H = np.eye(4)
        
        # Track 1 as prediction, Track 2 as measurement
        x_pred = track1.state
        P_pred = track1.covariance
        
        z = track2.state
        R = track2.covariance
        
        # Innovation
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        
        # Kalman gain
        K = P_pred @ H.T @ inv(S)
        
        # Update
        x_fused = x_pred + K @ y
        P_fused = (np.eye(4) - K @ H) @ P_pred
        
        fused_track = Track(
            id=max(track1.id, track2.id),
            sensor_id=-1,
            timestamp=max(track1.timestamp, track2.timestamp),
            state=x_fused,
            covariance=P_fused,
            confidence=min(track1.confidence + track2.confidence, 1.0)
        )
        
        return fused_track
    
    def fuse_tracks(self, tracks: List[Track], method: str = 'CI') -> List[Track]:
        """
        Main fusion function that processes multiple tracks
        """
        if len(tracks) < 2:
            return tracks
        
        fused_tracks = []
        used_indices = set()
        
        for i in range(len(tracks)):
            if i in used_indices:
                continue
                
            current_track = tracks[i]
            associated_tracks = [current_track]
            
            for j in range(i + 1, len(tracks)):
                if j in used_indices:
                    continue
                    
                if self.chi_square_gate(current_track, tracks[j]):
                    associated_tracks.append(tracks[j])
                    used_indices.add(j)
            
            used_indices.add(i)
            
            # Fuse associated tracks
            if len(associated_tracks) > 1:
                fused = associated_tracks[0]
                for track in associated_tracks[1:]:
                    if method == 'CI':
                        fused = self.covariance_intersection(fused, track)
                    elif method == 'SCC':
                        fused = self.simple_convex_combination(fused, track)
                    elif method == 'Kalman':
                        fused = self.kalman_fusion(fused, track)
                
                fused_tracks.append(fused)
                
                # Store fusion history
                self.fusion_history.append({
                    'timestamp': max(t.timestamp for t in associated_tracks),
                    'method': method,
                    'input_tracks': len(associated_tracks),
                    'mahalanobis_distance': self.mahalanobis_distance(associated_tracks[0], associated_tracks[1]) if len(associated_tracks) == 2 else 0
                })
            else:
                fused_tracks.append(current_track)
        
        return fused_tracks

def generate_sample_dataset(n_sensors: int = 3, n_targets: int = 5, n_timesteps: int = 20) -> List[Track]:
    """
    Generate sample radar track dataset with realistic parameters
    """
    np.random.seed(42)  # For reproducibility
    tracks = []
    track_id = 0
    
    # True target positions and velocities
    true_targets = []
    for i in range(n_targets):
        true_targets.append({
            'initial_pos': np.random.uniform(-10000, 10000, 2),  # meters
            'velocity': np.random.uniform(-100, 100, 2)  # m/s
        })
    
    for timestep in range(n_timesteps):
        timestamp = timestep * 1.0  # 1 second intervals
        
        for target_idx, target in enumerate(true_targets):
            # True position at this timestep
            true_pos = target['initial_pos'] + target['velocity'] * timestamp
            true_state = np.concatenate([true_pos, target['velocity']])
            
            for sensor_id in range(n_sensors):
                # Add sensor-specific noise and bias
                pos_noise_std = 50 + sensor_id * 20  # Different accuracy per sensor
                vel_noise_std = 5 + sensor_id * 2
                
                # Position bias per sensor
                pos_bias = np.array([sensor_id * 30, sensor_id * 20])
                
                # Measured state with noise and bias
                pos_noise = np.random.normal(0, pos_noise_std, 2)
                vel_noise = np.random.normal(0, vel_noise_std, 2)
                
                measured_pos = true_pos + pos_noise + pos_bias
                measured_vel = target['velocity'] + vel_noise
                measured_state = np.concatenate([measured_pos, measured_vel])
                
                # Covariance matrix (sensor dependent)
                pos_var = pos_noise_std**2
                vel_var = vel_noise_std**2
                covariance = np.diag([pos_var, pos_var, vel_var, vel_var])
                
                # Add some correlation between position and velocity
                covariance[0, 2] = covariance[2, 0] = pos_var * vel_var * 0.1
                covariance[1, 3] = covariance[3, 1] = pos_var * vel_var * 0.1
                
                track = Track(
                    id=track_id,
                    sensor_id=sensor_id,
                    timestamp=timestamp,
                    state=measured_state,
                    covariance=covariance,
                    confidence=0.7 + 0.1 * (3 - sensor_id)  # Better sensors have higher confidence
                )
                
                tracks.append(track)
                track_id += 1
    
    return tracks

def analyze_fusion_performance(original_tracks: List[Track], fused_tracks: List[Track]) -> dict:
    """
    Analyze the performance of track fusion
    """
    results = {
        'original_count': len(original_tracks),
        'fused_count': len(fused_tracks),
        'reduction_ratio': (len(original_tracks) - len(fused_tracks)) / len(original_tracks),
        'avg_confidence_original': np.mean([t.confidence for t in original_tracks]),
        'avg_confidence_fused': np.mean([t.confidence for t in fused_tracks]),
        'avg_position_uncertainty_original': np.mean([np.sqrt(np.trace(t.covariance[:2, :2])) for t in original_tracks]),
        'avg_position_uncertainty_fused': np.mean([np.sqrt(np.trace(t.covariance[:2, :2])) for t in fused_tracks])
    }
    
    return results

def plot_results(tracks: List[Track], fused_tracks: List[Track], fusion_system: TrackToTrackFusion):
    """
    Create comprehensive plots for track fusion analysis
    """
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Track positions before fusion
    plt.subplot(2, 3, 1)
    sensors = {}
    for track in tracks:
        if track.sensor_id not in sensors:
            sensors[track.sensor_id] = {'x': [], 'y': []}
        sensors[track.sensor_id]['x'].append(track.state[0])
        sensors[track.sensor_id]['y'].append(track.state[1])
    
    for sensor_id, data in sensors.items():
        plt.scatter(data['x'], data['y'], label=f'Sensor {sensor_id}', alpha=0.6, s=30)
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Original Tracks from Multiple Sensors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Fused track positions
    plt.subplot(2, 3, 2)
    fused_x = [track.state[0] for track in fused_tracks]
    fused_y = [track.state[1] for track in fused_tracks]
    
    plt.scatter(fused_x, fused_y, c='red', s=100, alpha=0.8, marker='*', label='Fused Tracks')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Fused Tracks')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Uncertainty ellipses comparison
    plt.subplot(2, 3, 3)
    
    # Plot original track uncertainties
    for i, track in enumerate(tracks[::10]):  # Sample every 10th track to avoid clutter
        pos = track.state[:2]
        cov = track.covariance[:2, :2]
        
        # Eigenvalues and eigenvectors for ellipse
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        angle = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])
        
        # 95% confidence ellipse
        width, height = 2 * np.sqrt(5.991 * eigenvals)
        
        ellipse = plt.matplotlib.patches.Ellipse(pos, width, height, 
                                               angle=np.degrees(angle), 
                                               alpha=0.3, color='blue')
        plt.gca().add_patch(ellipse)
        plt.plot(pos[0], pos[1], 'bo', markersize=3)
    
    # Plot fused track uncertainties
    for track in fused_tracks:
        pos = track.state[:2]
        cov = track.covariance[:2, :2]
        
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        angle = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])
        width, height = 2 * np.sqrt(5.991 * eigenvals)
        
        ellipse = plt.matplotlib.patches.Ellipse(pos, width, height, 
                                               angle=np.degrees(angle), 
                                               alpha=0.7, color='red')
        plt.gca().add_patch(ellipse)
        plt.plot(pos[0], pos[1], 'r*', markersize=8)
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Uncertainty Ellipses (Blue: Original, Red: Fused)')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Confidence comparison
    plt.subplot(2, 3, 4)
    original_conf = [track.confidence for track in tracks]
    fused_conf = [track.confidence for track in fused_tracks]
    
    plt.hist(original_conf, bins=20, alpha=0.6, label='Original Tracks', color='blue')
    plt.hist(fused_conf, bins=20, alpha=0.6, label='Fused Tracks', color='red')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.title('Track Confidence Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Fusion statistics over time
    plt.subplot(2, 3, 5)
    if fusion_system.fusion_history:
        timestamps = [h['timestamp'] for h in fusion_system.fusion_history]
        input_tracks = [h['input_tracks'] for h in fusion_system.fusion_history]
        
        plt.plot(timestamps, input_tracks, 'o-', linewidth=2, markersize=6)
        plt.xlabel('Time (s)')
        plt.ylabel('Number of Input Tracks per Fusion')
        plt.title('Fusion Activity Over Time')
        plt.grid(True, alpha=0.3)
    
    # Plot 6: Position uncertainty reduction
    plt.subplot(2, 3, 6)
    original_uncertainties = [np.sqrt(np.trace(t.covariance[:2, :2])) for t in tracks]
    fused_uncertainties = [np.sqrt(np.trace(t.covariance[:2, :2])) for t in fused_tracks]
    
    plt.boxplot([original_uncertainties, fused_uncertainties], 
                labels=['Original', 'Fused'])
    plt.ylabel('Position Uncertainty (m)')
    plt.title('Position Uncertainty Comparison')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def compare_fusion_methods(tracks: List[Track]) -> dict:
    """
    Compare different fusion methods
    """
    methods = ['CI', 'SCC', 'Kalman']
    results = {}
    
    for method in methods:
        fusion_system = TrackToTrackFusion()
        fused = fusion_system.fuse_tracks(tracks.copy(), method=method)
        
        performance = analyze_fusion_performance(tracks, fused)
        performance['method'] = method
        results[method] = performance
    
    return results

# Main execution
if __name__ == "__main__":
    print("Track-to-Track Fusion for Defense Radar Systems")
    print("=" * 50)
    
    # Generate sample dataset
    print("Generating sample radar track dataset...")
    tracks = generate_sample_dataset(n_sensors=3, n_targets=5, n_timesteps=20)
    print(f"Generated {len(tracks)} tracks from {3} sensors tracking {5} targets over {20} timesteps")
    
    # Initialize fusion system
    fusion_system = TrackToTrackFusion(gate_threshold=9.21)  # 95% confidence
    
    # Perform fusion using Covariance Intersection
    print("\nPerforming track fusion using Covariance Intersection...")
    fused_tracks = fusion_system.fuse_tracks(tracks, method='CI')
    
    # Analyze performance
    performance = analyze_fusion_performance(tracks, fused_tracks)
    
    print(f"\nFusion Results:")
    print(f"Original tracks: {performance['original_count']}")
    print(f"Fused tracks: {performance['fused_count']}")
    print(f"Reduction ratio: {performance['reduction_ratio']:.2%}")
    print(f"Average confidence improvement: {performance['avg_confidence_fused'] - performance['avg_confidence_original']:.3f}")
    print(f"Position uncertainty reduction: {(performance['avg_position_uncertainty_original'] - performance['avg_position_uncertainty_fused'])/performance['avg_position_uncertainty_original']:.2%}")
    
    # Compare fusion methods
    print("\nComparing fusion methods...")
    method_comparison = compare_fusion_methods(tracks)
    
    comparison_df = pd.DataFrame(method_comparison).T
    print("\nMethod Comparison:")
    print(comparison_df[['method', 'fused_count', 'reduction_ratio', 'avg_confidence_fused', 'avg_position_uncertainty_fused']])
    
    # Create visualizations
    print("\nGenerating plots...")
    plot_results(tracks, fused_tracks, fusion_system)
    
    # Additional analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Method comparison plot
    methods = list(method_comparison.keys())
    reduction_ratios = [method_comparison[m]['reduction_ratio'] for m in methods]
    uncertainty_reductions = [(method_comparison[m]['avg_position_uncertainty_original'] - 
                              method_comparison[m]['avg_position_uncertainty_fused']) / 
                             method_comparison[m]['avg_position_uncertainty_original'] for m in methods]
    
    axes[0, 0].bar(methods, reduction_ratios, color=['blue', 'green', 'orange'])
    axes[0, 0].set_ylabel('Track Reduction Ratio')
    axes[0, 0].set_title('Track Reduction by Fusion Method')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].bar(methods, uncertainty_reductions, color=['blue', 'green', 'orange'])
    axes[0, 1].set_ylabel('Uncertainty Reduction Ratio')
    axes[0, 1].set_title('Uncertainty Reduction by Fusion Method')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Track trajectory visualization
    axes[1, 0].set_title('Sample Track Trajectories')
    unique_targets = {}
    for track in tracks[:60]:  # Sample first 60 tracks
        target_key = (round(track.state[0]/1000) * 1000, round(track.state[1]/1000) * 1000)
        if target_key not in unique_targets:
            unique_targets[target_key] = {'x': [], 'y': [], 't': []}
        unique_targets[target_key]['x'].append(track.state[0])
        unique_targets[target_key]['y'].append(track.state[1])
        unique_targets[target_key]['t'].append(track.timestamp)
    
    for i, (key, traj) in enumerate(list(unique_targets.items())[:5]):
        axes[1, 0].plot(traj['x'], traj['y'], 'o-', label=f'Target {i+1}', alpha=0.7)
    
    axes[1, 0].set_xlabel('X Position (m)')
    axes[1, 0].set_ylabel('Y Position (m)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Mahalanobis distance distribution
    distances = []
    for i in range(len(tracks)):
        for j in range(i+1, len(tracks)):
            if tracks[i].timestamp == tracks[j].timestamp:
                dist = fusion_system.mahalanobis_distance(tracks[i], tracks[j])
                if dist < 100:  # Filter out extreme values
                    distances.append(dist)
    
    axes[1, 1].hist(distances, bins=30, alpha=0.7, color='purple')
    axes[1, 1].axvline(x=np.sqrt(fusion_system.gate_threshold), color='red', 
                      linestyle='--', label='Gating Threshold')
    axes[1, 1].set_xlabel('Mahalanobis Distance')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Mahalanobis Distance Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nTrack-to-Track Fusion Analysis Complete!")
    print("The system successfully demonstrated:")
    print("- Multi-sensor track fusion algorithms")
    print("- Mathematical gating and association")
    print("- Performance analysis and visualization")
    print("- Comparison of different fusion methods")
