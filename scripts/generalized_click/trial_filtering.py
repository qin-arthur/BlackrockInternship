"""Trial filtering for MINT compatibility.

Filters trials to ensure compatibility with MINT's trajectory assumptions.
Discards non-movement epochs and trials without behavioral data.
Applies behavioral clustering to targetless movement trials.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings


def extract_behavioral_features(trial_data: List[Dict], verbose: bool = True) -> Dict:
    """Extract behavioral features from trial data for clustering.
    
    Args:
        trial_data: List of trial dictionaries
        verbose: Whether to print progress
        
    Returns:
        Dictionary with features and metadata for clustering
    """
    features = []
    trial_indices = []
    
    for i, trial in enumerate(trial_data):
        if trial['target'] is not None:
            continue  # Skip trials with explicit targets
        
        position = trial['position']  # (T, 2)
        velocity = trial['velocity']  # (T, 2)
        
        # Skip trials with no or very little behavioral data
        if position is None or velocity is None:
            continue
        if position.shape[0] < 5 or velocity.shape[0] < 5:  # Skip very short trials
            continue
        if np.all(position == 0) or np.all(velocity == 0):  # Skip trials with no movement
            continue
            
        try:
            # Extract kinematic features
            trial_features = compute_kinematic_features(position, velocity)
            features.append(trial_features)
            trial_indices.append(i)
            
        except Exception as e:
            if verbose:
                print(f"    Warning: Failed to extract features for trial {i}: {e}")
            continue
    
    if len(features) == 0:
        return {
            'features': np.array([]),
            'trial_indices': [],
            'feature_names': []
        }
    
    features_array = np.array(features)
    feature_names = [
        'start_x', 'start_y', 'end_x', 'end_y',
        'total_distance', 'straight_distance', 'path_efficiency',
        'mean_speed', 'max_speed', 'speed_variance',
        'movement_angle', 'angle_variance',
        'duration_normalized', 'acceleration_peaks'
    ]
    
    if verbose:
        print(f"    Extracted features from {len(trial_indices)} target-less trials")
        print(f"    Feature dimensions: {features_array.shape}")
    
    return {
        'features': features_array,
        'trial_indices': trial_indices,
        'feature_names': feature_names
    }


def compute_kinematic_features(position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """Compute kinematic features for a single trial.
    
    Args:
        position: Position data (T, 2)
        velocity: Velocity data (T, 2)
        
    Returns:
        Array of kinematic features
    """
    # Basic position features
    start_pos = position[0]
    end_pos = position[-1]
    
    # Distance metrics
    displacement = end_pos - start_pos
    straight_distance = np.linalg.norm(displacement)
    
    # Path length
    position_diffs = np.diff(position, axis=0)
    step_distances = np.linalg.norm(position_diffs, axis=1)
    total_distance = np.sum(step_distances)
    
    # Path efficiency (straightness)
    path_efficiency = straight_distance / (total_distance + 1e-6)
    
    # Speed metrics
    speed = np.linalg.norm(velocity, axis=1)
    mean_speed = np.mean(speed)
    max_speed = np.max(speed)
    speed_variance = np.var(speed)
    
    # Movement direction
    if straight_distance > 1e-6:
        movement_angle = np.arctan2(displacement[1], displacement[0])
    else:
        movement_angle = 0.0
    
    # Direction consistency (how much direction varies)
    if len(velocity) > 1:
        velocity_angles = np.arctan2(velocity[:, 1], velocity[:, 0])
        # Handle angle wrapping
        angle_diffs = np.diff(velocity_angles)
        angle_diffs = np.mod(angle_diffs + np.pi, 2*np.pi) - np.pi
        angle_variance = np.var(angle_diffs)
    else:
        angle_variance = 0.0
    
    # Temporal features
    duration_normalized = len(position) / 100.0  # Normalize by expected duration
    
    # Acceleration features (number of significant acceleration peaks)
    if len(speed) > 2:
        acceleration = np.diff(speed)
        accel_threshold = np.std(acceleration) * 2
        acceleration_peaks = np.sum(np.abs(acceleration) > accel_threshold)
    else:
        acceleration_peaks = 0
    
    features = np.array([
        start_pos[0], start_pos[1], end_pos[0], end_pos[1],
        total_distance, straight_distance, path_efficiency,
        mean_speed, max_speed, speed_variance,
        movement_angle, angle_variance,
        duration_normalized, acceleration_peaks
    ])
    
    return features


def cluster_behavioral_features(features: np.ndarray, 
                               n_clusters: Optional[int] = None,
                               verbose: bool = True) -> Tuple[np.ndarray, Dict]:
    """Cluster behavioral features to create pseudo-target conditions.
    
    Args:
        features: Feature matrix (n_trials, n_features)
        n_clusters: Number of clusters (auto-determined if None)
        verbose: Whether to print progress
        
    Returns:
        Tuple of (cluster_labels, clustering_info)
    """
    if len(features) == 0:
        return np.array([]), {}
    
    if len(features) < 3:
        # Too few trials to cluster meaningfully
        return np.zeros(len(features)), {'method': 'single_cluster', 'n_clusters': 1}
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Remove any features with zero variance
    feature_variance = np.var(features_scaled, axis=0)
    valid_features = feature_variance > 1e-6
    features_scaled = features_scaled[:, valid_features]
    
    if features_scaled.shape[1] == 0:
        # No informative features
        return np.zeros(len(features)), {'method': 'no_variance', 'n_clusters': 1}
    
    # Dimensionality reduction if needed
    if features_scaled.shape[1] > 6:
        pca = PCA(n_components=min(6, len(features) - 1))
        features_reduced = pca.fit_transform(features_scaled)
        explained_variance = np.sum(pca.explained_variance_ratio_)
    else:
        features_reduced = features_scaled
        explained_variance = 1.0
    
    # Determine number of clusters if not specified
    if n_clusters is None:
        # Use elbow method or reasonable heuristic
        max_clusters = min(8, len(features) // 3)  # At least 3 trials per cluster
        if max_clusters < 2:
            n_clusters = 1
        else:
            # Simple heuristic: sqrt(n_trials / 2)
            n_clusters = max(2, min(max_clusters, int(np.sqrt(len(features) / 2))))
    
    if n_clusters == 1 or len(features) < n_clusters:
        return np.zeros(len(features)), {'method': 'single_cluster', 'n_clusters': 1}
    
    # Perform K-means clustering
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_reduced)
        
        # Validate clustering quality
        cluster_counts = np.bincount(cluster_labels)
        min_cluster_size = np.min(cluster_counts)
        
        clustering_info = {
            'method': 'kmeans',
            'n_clusters': n_clusters,
            'cluster_counts': cluster_counts,
            'min_cluster_size': min_cluster_size,
            'explained_variance': explained_variance,
            'inertia': kmeans.inertia_,
            'scaler': scaler,
            'valid_features': valid_features
        }
        
        if verbose:
            print(f"    Behavioral clustering results:")
            print(f"      Method: K-means with {n_clusters} clusters")
            print(f"      Cluster sizes: {cluster_counts}")
            print(f"      Min cluster size: {min_cluster_size}")
            print(f"      Explained variance: {explained_variance:.3f}")
        
        return cluster_labels, clustering_info
        
    except Exception as e:
        if verbose:
            print(f"    Warning: Clustering failed: {e}")
        return np.zeros(len(features)), {'method': 'failed', 'n_clusters': 1, 'error': str(e)}


def create_pseudo_targets_from_clusters(cluster_labels: np.ndarray, 
                                       features: np.ndarray,
                                       clustering_info: Dict,
                                       verbose: bool = True) -> List[Tuple[float, float]]:
    """Create pseudo-target coordinates from cluster centroids.
    
    Args:
        cluster_labels: Cluster assignment for each trial
        features: Original feature matrix
        clustering_info: Information from clustering
        verbose: Whether to print progress
        
    Returns:
        List of pseudo-target coordinates for each trial
    """
    if len(cluster_labels) == 0:
        return []
    
    pseudo_targets = []
    
    # Use end position as the basis for pseudo-targets
    end_positions = features[:, 2:4]  # end_x, end_y from feature extraction
    
    n_clusters = len(np.unique(cluster_labels))
    
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        if np.sum(cluster_mask) == 0:
            continue
            
        # Compute centroid of end positions for this cluster
        cluster_end_positions = end_positions[cluster_mask]
        centroid = np.mean(cluster_end_positions, axis=0)
        
        # Round to reasonable precision (similar to adaptive precision)
        centroid_rounded = np.round(centroid, decimals=2)
        
        if verbose and n_clusters <= 10:  # Don't spam for too many clusters
            n_trials = np.sum(cluster_mask)
            print(f"    Cluster {cluster_id}: {n_trials} trials -> pseudo-target {tuple(centroid_rounded)}")
    
    # Assign pseudo-targets to each trial
    for label in cluster_labels:
        cluster_mask = cluster_labels == label
        cluster_end_positions = end_positions[cluster_mask]
        centroid = np.mean(cluster_end_positions, axis=0)
        centroid_rounded = np.round(centroid, decimals=2)
        pseudo_targets.append(tuple(centroid_rounded))
    
    return pseudo_targets


def should_discard_trial(trial: Dict) -> Tuple[bool, str]:
    """Determine if a trial should be discarded from MINT training.
    
    Discard trials that break MINT's core assumptions:
    - No behavioral goal
    - No consistent temporal structure  
    - No meaningful movement data
    
    Args:
        trial: Trial dictionary
        
    Returns:
        Tuple of (should_discard, reason)
    """
    # Check epoch type first - these are fundamentally incompatible with MINT
    epoch_type = trial.get('epoch_type', '')
    
    # Always discard these epoch types regardless of data content
    non_movement_epochs = ['InterTrial', 'FailSafe', 'Calibration', 'OrthoCalibration', 'Idle']
    for epoch in non_movement_epochs:
        if epoch in epoch_type:
            return True, f"Non-movement epoch: {epoch_type}"
    
    # Check for behavioral data
    position = trial.get('position')
    velocity = trial.get('velocity')
    target = trial.get('target')
    
    # If has target and behavior, always keep
    if target is not None:
        return False, "Has explicit target"
    
    # No target - check behavioral content
    has_behavioral_data = (
        position is not None and velocity is not None and 
        position.shape[0] > 5 and velocity.shape[0] > 5 and
        not (np.all(position == 0) or np.all(velocity == 0))
    )
    
    if not has_behavioral_data:
        return True, "No meaningful behavioral data"
    
    # Check if epoch type suggests movement-related activity
    movement_epochs = ['Click', 'Drag', 'Reach', 'Move', 'Observation']
    is_movement_epoch = any(epoch in epoch_type for epoch in movement_epochs)
    
    if not is_movement_epoch:
        return True, f"Non-movement epoch type: {epoch_type}"
    
    # Keep trials with movement data and movement-related epoch types
    return False, "Has behavioral data and movement epoch"


def filter_trials_for_mint(all_trial_data: List[Dict], verbose: bool = True) -> List[Dict]:
    """Filter trials for MINT compatibility and assign pseudo-targets where appropriate.
    
    Strategy:
    1. Discard trials incompatible with MINT (InterTrial, FailSafe, Calibration, etc.)
    2. Keep trials with targets and behavioral data
    3. For targetless trials with behavioral data: use behavioral clustering
    
    Args:
        all_trial_data: List of all trial dictionaries
        verbose: Whether to print progress
        
    Returns:
        Filtered trial data with pseudo-targets assigned where appropriate
    """
    if verbose:
        print(f"\n    Filtering trials for MINT compatibility:")
    
    # First pass: identify which trials to keep vs discard
    keep_trials = []
    discard_trials = []
    discard_reasons = {}
    
    for i, trial in enumerate(all_trial_data):
        should_discard, reason = should_discard_trial(trial)
        if should_discard:
            discard_trials.append(i)
            discard_reasons[i] = reason
        else:
            keep_trials.append(i)
    
    if verbose:
        print(f"    Trial filtering results:")
        print(f"      Keeping: {len(keep_trials)} trials")
        print(f"      Discarding: {len(discard_trials)} trials")
        
        # Show discard reasons
        reason_counts = {}
        for reason in discard_reasons.values():
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        print(f"    Discard reasons:")
        for reason, count in reason_counts.items():
            print(f"      {reason}: {count} trials")
    
    # Create filtered trial data with only kept trials
    filtered_trial_data = [all_trial_data[i] for i in keep_trials]
    
    if len(filtered_trial_data) == 0:
        if verbose:
            print(f"    WARNING: No trials remaining after filtering!")
        return []
    
    # Now handle pseudo-target assignment for remaining targetless trials
    targetless_with_behavior = []
    
    for i, trial in enumerate(filtered_trial_data):
        if trial['target'] is None:
            targetless_with_behavior.append(i)
    
    if verbose:
        n_with_targets = len(filtered_trial_data) - len(targetless_with_behavior)
        print(f"    After filtering: {n_with_targets} trials with targets, {len(targetless_with_behavior)} targetless with behavior")
    
    # Apply behavioral clustering to targetless trials with behavioral data
    if len(targetless_with_behavior) > 0:
        if verbose:
            print(f"    Applying behavioral clustering to {len(targetless_with_behavior)} targetless trials...")
        
        targetless_trial_data = [filtered_trial_data[i] for i in targetless_with_behavior]
        feature_data = extract_behavioral_features(targetless_trial_data, verbose=verbose)
        
        if len(feature_data['trial_indices']) > 0:
            cluster_labels, clustering_info = cluster_behavioral_features(
                feature_data['features'], 
                verbose=verbose
            )
            
            pseudo_targets = create_pseudo_targets_from_clusters(
                cluster_labels, 
                feature_data['features'],
                clustering_info,
                verbose=verbose
            )
            
            # Assign pseudo-targets to targetless trials
            for i, local_idx in enumerate(feature_data['trial_indices']):
                global_idx = targetless_with_behavior[local_idx]
                if i < len(pseudo_targets):
                    filtered_trial_data[global_idx]['target'] = pseudo_targets[i]
                    filtered_trial_data[global_idx]['is_pseudo_target'] = True
                    filtered_trial_data[global_idx]['pseudo_target_type'] = 'behavioral_clustering'
    
    # Mark all trials with target assignment status
    for trial in filtered_trial_data:
        if 'is_pseudo_target' not in trial:
            trial['is_pseudo_target'] = False
    
    if verbose:
        n_real = sum(1 for trial in filtered_trial_data if trial['target'] is not None and not trial.get('is_pseudo_target', False))
        n_pseudo = sum(1 for trial in filtered_trial_data if trial.get('is_pseudo_target', False))
        n_none = sum(1 for trial in filtered_trial_data if trial['target'] is None)
        
        print(f"    Final results:")
        print(f"      Real targets: {n_real} trials")
        print(f"      Pseudo-targets (behavioral): {n_pseudo} trials") 
        print(f"      No targets: {n_none} trials")
        print(f"      Total kept for MINT: {len(filtered_trial_data)} trials")
    
    return filtered_trial_data