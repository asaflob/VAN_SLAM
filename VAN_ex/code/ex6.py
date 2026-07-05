from ex5 import *
import gtsam
import numpy as np
import matplotlib.pyplot as plt
from gtsam.utils import plot
"""
def build_bundle_graph(db, poses, window_frames, K, P, Q):

    Constructs a GTSAM nonlinear factor graph for a specific window of frames.

    This function initializes the optimization problem by setting the first frame
    of the window as the local origin (anchor). It adds a PriorFactor to the anchor
    and GenericStereoFactor3D projection factors for all valid tracked 3D landmarks.

    Args:
        db (TrackingDB): The tracking database containing features and stereo matches.
        poses (list of np.ndarray): List of 4x4 Extrinsic matrices (World-to-Camera) from PnP.
        window_frames (list of int): Indices of the frames included in the current bundle window.
        K (np.ndarray): 3x3 Intrinsic camera matrix.
        P (np.ndarray): 3x4 Left camera projection matrix.
        Q (np.ndarray): 3x4 Right camera projection matrix.

    Returns:
        tuple:
            - graph (gtsam.NonlinearFactorGraph): The constructed factor graph.
            - initial_estimate (gtsam.Values): The initial poses and 3D points for optimization.
            - stereo_calib (gtsam.Cal3_S2Stereo): The stereo calibration object used.


def run_window_bundle_adjustment(db, poses, start_frame, end_frame, K, P, Q):
    Builds and optimizes a bundle graph for a specific window of frames.

    Args:
        db, poses, K, P, Q: Same as before.
        start_frame (int): The index of the first frame in the window (c0).
        end_frame (int): The index of the last frame in the window (ck).

    Returns:
        tuple: (graph, result)
        
    window_frames = list(range(start_frame, end_frame + 1))

    graph, initial_estimate, stereo_calib = build_bundle_graph(db, poses, window_frames, K, P, Q)

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
    result = optimizer.optimize()

    return graph, result

"""


def q6_1(graph, result, c0_key, ck_key, plot_fig=False):
    """
    Extracts the relative pose and conditional covariance between two keyframes from a BA optimization.

    Args:
        graph (gtsam.NonlinearFactorGraph): The optimized factor graph from the Bundle Adjustment.
        result (gtsam.Values): The optimized values (poses and landmarks) from the BA.
        c0_key (int): The GTSAM key for the first keyframe (c0).
        ck_key (int): The GTSAM key for the second keyframe (ck).
        plot_fig (bool): Flag to plot the 3D trajectory with covariances.

    Returns:
        tuple:
            - relative_pose (gtsam.Pose3): The estimated relative motion between c0 and ck.
            - conditional_cov (np.ndarray): The 6x6 conditional covariance matrix P(ck|c0).
    """

    #Extract the Pose3 objects for c0 and ck from 'result' using their keys.
    pose_c0 = result.atPose3(c0_key)
    pose_ck = result.atPose3(ck_key)

    #Calculate the relative pose between the two frames:
    relative_pose = pose_c0.between(pose_ck)

    #Calculate the marginal covariances from the optimization result:
    marginals = gtsam.Marginals(graph, result)

    #Extract the joint marginal covariance matrix for c0 and ck:
    keys = gtsam.KeyVector()
    keys.append(c0_key)
    keys.append(ck_key)
    joint_cov = marginals.jointMarginalCovariance(keys).fullMatrix() #

    #Extract the 6x6 sub-blocks from the 12x12 joint covariance matrix:
    Sigma_00 = joint_cov[0:6, 0:6]  # Marginal covariance of c0
    Sigma_0k = joint_cov[0:6, 6:12]  # Cross-covariance
    Sigma_k0 = joint_cov[6:12, 0:6]  # Cross-covariance (which is Sigma_0k.T)
    Sigma_kk = joint_cov[6:12, 6:12]  # Marginal covariance of ck


    #Apply Conditioning to find the covariance associated with P(ck|c0):
    #    Calculate the conditional covariance using the Schur complement formula:
    conditional_cov = Sigma_kk - (Sigma_k0 @ np.linalg.inv(Sigma_00) @ Sigma_0k)

    orig_c0 = gtsam.symbolIndex(c0_key)
    orig_ck = gtsam.symbolIndex(ck_key)

    #Print the resulting relative pose and the computed conditional_cov.
    #todo
    # print(f"--- Relative Pose between Keyframe {c0_key} and Keyframe {ck_key} ---")
    # print(relative_pose)
    # print(f"--- Conditional Covariance P(c{ck_key} | c{c0_key}) ---")
    # print(conditional_cov)
    # print("-" * 50)

    #Handle Plotting (if plot_fig is True):
    if plot_fig:
        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection='3d')

        # Plot the trajectory with marginals as requested in the assignment[cite: 1]
        plot.plot_trajectory(1, result, marginals=marginals, scale=1)

        ax.set_title(f"3D Trajectory & Covariances (Frames {orig_c0} to {orig_ck})")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

    return relative_pose, conditional_cov


def process_all_ba_windows(all_ba_results, plot_first_only=True):
    """
    Iterates over all Bundle Adjustment optimizations to estimate the relative motion
    between every two consecutive keyframes and their relative covariance[cite: 1].

    Args:
        all_ba_results (list of dict): A list where each element contains the data
                                       from one BA window {'graph': graph, 'result': result,
                                       'c0_key': int, 'ck_key': int}.
        plot_first_only (bool): If True, only plots the 3D graph for the first window.

    Returns:
        list of dict: The constraints ready to be added to the Pose Graph.
    """
    pose_graph_constraints = []

    for idx, window_data in enumerate(all_ba_results):
        graph = window_data['graph']
        result = window_data['result']
        c0_key = window_data['c0_key']
        ck_key = window_data['ck_key']

        # Plot the 3D graph only for the first BA result (or as specified)
        should_plot = plot_first_only and (idx == 0)

        print(f"Processing BA Window {idx + 1}...")
        rel_pose, cond_cov = q6_1(graph, result, c0_key, ck_key, plot_fig=should_plot)

        pose_graph_constraints.append({
            'c0_key': c0_key,
            'ck_key': ck_key,
            'relative_pose': rel_pose,
            'noise_model': gtsam.noiseModel.Gaussian.Covariance(cond_cov)  # Ready for BetweenFactorPose3
        })

    return pose_graph_constraints


def generate_all_ba_results(db, poses, K, P, Q):
    """
    Runs Bundle Adjustment on sliding windows over the entire trajectory to generate
    results for Pose Graph optimization (Exercise 6).
    """
    print("\n--- Running Bundle Adjustments to generate data for Ex 6 ---")
    current_start = 0
    all_ba_results = []

    while current_start < len(poses) - 5:
        end = get_keyframe(poses, current_start)
        window = list(range(current_start, end + 1))

        graph, initial, calib = build_bundle_graph(db, poses, window, K, P, Q)

        if graph.size() == 0:
            current_start = end
            continue

        print(f"Optimizing window: Frame {current_start} to Frame {end}")
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
        result = optimizer.optimize()

        c0_key = gtsam.symbol('x', current_start)
        ck_key = gtsam.symbol('x', end)

        all_ba_results.append({
            'graph': graph,
            'result': result,
            'c0_key': c0_key,
            'ck_key': ck_key
        })

        current_start = end

    return all_ba_results

######## 6.2 ########

def plot_pose_graph_locations(values, title="Pose Graph Locations", color='b'):
    """
    Helper function to plot the 3D locations of the keyframes from a GTSAM Values object.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = [], [], []

    for key in values.keys():
        pose = values.atPose3(key)
        x.append(pose.x())
        y.append(pose.y())
        z.append(pose.z())

    ax.plot(x, y, z, marker='o', linestyle='-', color=color, markersize=4)

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=0, azim=-90)
    plt.show()

def q6_2(pose_graph_constraints, initial_poses_dict, ground_truth_path):
    """
    Builds and optimizes a Pose Graph based on the relative motions estimated in Section 6.1.

    Args:
        pose_graph_constraints (list of dict): Extracted relative poses and covariances.
            Each dict contains: 'c0_key', 'ck_key', 'relative_pose', and 'noise_model'.
        initial_poses_dict (dict): A mapping from GTSAM keys to global gtsam.Pose3 objects
            to serve as the initial guess.
    """

    #Initialize Graph and Values:
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    #Add Anchor (PriorFactor):
    first_key = pose_graph_constraints[0]['c0_key']
    anchor_pose = initial_poses_dict[first_key]
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6] * 6))
    graph.add(gtsam.PriorFactorPose3(first_key, anchor_pose, prior_noise))


    #Add Relative Pose Constraints:
    for constraint in pose_graph_constraints:
        c0 = constraint['c0_key']
        ck = constraint['ck_key']
        rel_pose = constraint['relative_pose']
        noise_model = constraint['noise_model']
        graph.add(gtsam.BetweenFactorPose3(c0, ck, rel_pose, noise_model))


    #Construct Initial Guess:
    for key, pose in initial_poses_dict.items():
        if not initial_estimate.exists(key):
            initial_estimate.insert(key, pose)

    #Plot Initial Poses:
    print("Plotting Initial Poses (Before Optimization)...")
    plot_pose_graph_locations(initial_estimate, title="Initial Guess - Before Optimization", color='red')


    # Initialize Optimizer & Print Initial Error:
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
    initial_error = optimizer.error()
    print(f"Initial Error (Before Optimization): {initial_error}")

    # Optimize:
    print("Running Optimization...")
    result = optimizer.optimize()

    # Print Final Error:
    final_error = optimizer.error()
    print(f"Final Error (After Optimization): {final_error}")

    #Plot Optimized Locations (Without Covariances):
    print("Plotting Optimized Poses (Without Covariances)...")
    plot_pose_graph_locations(result, title="Optimized Trajectory - Without Covariances", color='green')

    # Extract Marginals:
    print("Calculating Marginals...")
    marginals = gtsam.Marginals(graph, result)

    # Plot Optimized Locations (With Final Marginal Covariances):
    print("Plotting Optimized Poses (With Covariances)...")
    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    plot.plot_trajectory(2, result, marginals=marginals, scale=1)

    ax.set_title("Optimized 3D Trajectory & Final Covariances")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

    compare_with_ground_truth(result, ground_truth_path)

    return graph, result


def compare_with_ground_truth(result, ground_truth_path):
    """
    Compares the optimized GTSAM trajectory with the Ground Truth trajectory,
    plotting the 2D top-down view and the Euclidean error over time.
    """
    # שימוש בפונקציה מהתרגילים הקודמים לטעינת הפוזות
    # ודא שיש לך ייבוא מתאים, למשל: from ex4 import load_kitti_poses
    gt_poses = load_kitti_poses(ground_truth_path)

    est_positions = []
    gt_positions = []
    frame_indices = []

    # איסוף ומיון המפתחות מהגרף כדי שהגרף ישורטט לפי סדר התקדמות הרכב
    keys = sorted([key for key in result.keys() if gtsam.symbolChr(key) == ord('x')])

    for key in keys:
        frame_idx = gtsam.symbolIndex(key)
        frame_indices.append(frame_idx)

        # 1. שליפת המיקום המוערך מתוך התוצאה הסופית של האופטימיזציה
        pose_est = result.atPose3(key)
        est_positions.append([pose_est.x(), pose_est.z()])

        # 2. שליפת המיקום האמיתי מתוך ה-Ground Truth
        T_gt = gt_poses[frame_idx]

        # חילוץ מיקום המצלמה העולמי מתוך הטרנספורמציה (בדומה לתרגיל 5)
        C_gt = -T_gt[:3, :3].T @ T_gt[:3, 3]
        gt_positions.append([C_gt[0], C_gt[2]])

    est_positions = np.array(est_positions)
    gt_positions = np.array(gt_positions)

    # --- שרטוט מסלול מוערך מול מסלול אמיתי ממבט על (X-Z Plane) ---
    plt.figure(figsize=(10, 8))
    plt.plot(gt_positions[:, 0], gt_positions[:, 1], 'k--', label='Ground Truth', linewidth=2)
    plt.plot(est_positions[:, 0], est_positions[:, 1], 'b-', label='Pose Graph Trajectory', linewidth=2)

    # סימון נקודת ההתחלה בראשית הצירים
    plt.scatter(est_positions[0, 0], est_positions[0, 1], c='red', marker='*', s=200, label='Start (0,0)', zorder=5)

    plt.title("Trajectory: Pose Graph vs Ground Truth (Top-Down View)")
    plt.xlabel("X (Right/Left) [meters]")
    plt.ylabel("Z (Forward Depth) [meters]")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()

    # --- חישוב ושרטוט שגיאה אוקלידית ---
    euclidean_errors = np.linalg.norm(est_positions - gt_positions, axis=1)

    plt.figure(figsize=(10, 5))
    plt.plot(frame_indices, euclidean_errors, 'm-o', markersize=4, color='purple')
    plt.title("Pose Graph Localization Error (Euclidean distance)")
    plt.xlabel("Keyframe ID")
    plt.ylabel("Euclidean Translation Error [meters]")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    project_root = r"C:\university\SHANA 5\semester B\67604-slam\VAN_SLAM\VAN_ex"
    sequence_dir = os.path.join(project_root, 'dataset', 'dataset_2026', 'sequences', '00')
    poses_dir = os.path.join(project_root, 'dataset', 'dataset_2026', 'poses')
    ground_truth_path = os.path.join(poses_dir, '00.txt')

    db = TrackingDB()
    db_filename = 'my_tracking_data_ex4'
    poses_npy_filename = 'my_estimated_poses_ex4.npy'
    K, M1, M2 = read_cameras()
    P, Q = K @ M1, K @ M2  # multiply by intrinsic camera matrix
    db.load(db_filename)
    estimated_poses = np.load(poses_npy_filename, allow_pickle=True)
    estimated_poses = list(estimated_poses)
    print("Data and estimated poses loaded successfully from file.")

    all_ba_results = generate_all_ba_results(db, estimated_poses, K, P, Q)

    print("\n ----- 6.1 ------")

    #pose_graph_constraints for 6.2
    pose_graph_constraints = process_all_ba_windows(all_ba_results, plot_first_only=True)

    print(f"\nExtracted {len(pose_graph_constraints)} constraints.")

    print("\n ----- 6.2 ------")

    initial_poses_dict = {}

    first_key = pose_graph_constraints[0]['c0_key']
    current_pose = gtsam.Pose3()
    initial_poses_dict[first_key] = current_pose

    for constraint in pose_graph_constraints:
        c0_key = constraint['c0_key']
        ck_key = constraint['ck_key']
        rel_pose = constraint['relative_pose']

        if c0_key in initial_poses_dict:
            # הפעולה compose מבצעת את הכפל הגיאומטרי: T_global_k = T_global_0 * T_relative
            next_pose = initial_poses_dict[c0_key].compose(rel_pose)
            initial_poses_dict[ck_key] = next_pose

    q6_2(pose_graph_constraints, initial_poses_dict, ground_truth_path)
