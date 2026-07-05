import gtsam
from gtsam.utils import plot
import numpy as np
import matplotlib.pyplot as plt
import cv2
from ex4 import *


def invert_se3(T):
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def q5_1(db, poses, K, P, Q):
    # 1. Find a random track of length >= 10
    target_track = None
    for track_id, frames in db.trackId_to_frames.items():
        if len(frames) >= 10:
            target_track = track_id
            break

    if target_track is None:
        print("No track with length >= 10 found.")
        return

    track_frames = db.frames(target_track)
    print(f"Selected Track #{target_track} with {len(track_frames)} frames.")

    # Load global poses (Assuming these are Camera-to-World based on ex4 code)
    # If your poses are World-to-Camera, you'll need to invert them using np.linalg.inv()
    # poses = load_kitti_poses(poses_path)

    # 2. Extract intrinsics and baseline for GTSAM
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    # The baseline 'b' can be derived from the projection matrices P, Q
    # In KITTI, typically Q (right camera) has a negative offset in the X axis: Q[0,3] = -fx * baseline
    baseline = -Q[0, 3] / fx

    # Create gtsam Calibration
    stereo_calib = gtsam.Cal3_S2Stereo(fx, fy, 0.0, cx, cy, baseline)

    stereo_cameras = {}

    # Define cameras for all frames in the track
    for frame_id in track_frames:
        T_wc = poses[frame_id]  # Assuming Camera-to-World

        # Extract R, t to create gtsam.Pose3
        R_cw = T_wc[:3, :3].T
        t_cw = -R_cw @ T_wc[:3, 3]
        pose3 = gtsam.Pose3(gtsam.Rot3(R_cw), gtsam.Point3(t_cw))
        stereo_cameras[frame_id] = gtsam.StereoCamera(pose3, stereo_calib)

    # 3. Triangulate the 3D point from the last frame
    last_frame = track_frames[-1]
    last_camera = stereo_cameras[last_frame]
    last_link = db.link(last_frame, target_track)

    # Create GTSAM StereoPoint2: (uL, uR, v)
    stereo_pt_last = gtsam.StereoPoint2(last_link.x_left, last_link.x_right, last_link.y)

    # backproject returns the 3D point in global coordinates!
    point3d_global = last_camera.backproject(stereo_pt_last)

    # 4. Define noise model for Factor
    sigma = 2.0
    noise_model = gtsam.noiseModel.Isotropic.Sigma(3, sigma)

    # Symbols for dummy Values object (required to evaluate factor error)
    pose_key = gtsam.symbol('x', 1)
    point_key = gtsam.symbol('l', 1)

    reproj_errors = []
    factor_errors = []
    distances = []  # frames from last frame (like in ex4)

    # 5. Project to all frames and calculate errors
    # Reversing to match your Ex4 plot style (distance from reference)
    for frame_id in reversed(track_frames):
        dist = last_frame - frame_id
        distances.append(dist)

        camera = stereo_cameras[frame_id]
        link = db.link(frame_id, target_track)
        measured_pt = gtsam.StereoPoint2(link.x_left, link.x_right, link.y)

        # --- L2 Reprojection Error ---
        projected_pt = camera.project(point3d_global)

        # difference in uL, uR, v
        du_l = measured_pt.uL() - projected_pt.uL()
        du_r = measured_pt.uR() - projected_pt.uR()
        dv = measured_pt.v() - projected_pt.v()

        l2_err = np.sqrt(du_l ** 2 + du_r ** 2 + dv ** 2)
        reproj_errors.append(l2_err)

        # --- Factor Error ---
        factor = gtsam.GenericStereoFactor3D(measured_pt, noise_model, pose_key, point_key, stereo_calib)

        # To get the factor error, we must supply the state (pose and point) using gtsam.Values
        values = gtsam.Values()
        values.insert(pose_key, camera.pose())
        values.insert(point_key, point3d_global)

        f_err = factor.error(values)
        factor_errors.append(f_err)

    # 6. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(distances, reproj_errors, marker='o', color='tab:blue')
    ax1.set_title(f'Reprojection Error (L2 Norm) - Track #{target_track}')
    ax1.set_xlabel('Distance from reference (frames)')
    ax1.set_ylabel('Error (pixels)')
    ax1.grid(True, alpha=0.3)

    ax2.plot(distances, factor_errors, marker='s', color='tab:red')
    ax2.set_title(f'Factor Error - Track #{target_track}')
    ax2.set_xlabel('Distance from reference (frames)')
    ax2.set_ylabel('Factor Error (Mahalanobis)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def get_keyframe(poses, start_frame, min_frames=10, max_frames=20, min_dist=4.0):
    dist_accumulated = 0.0
    for i in range(start_frame, min(start_frame + max_frames, len(poses) - 1)):
        # Calculate distance between consecutive frames
        T_curr = poses[i]
        T_next = poses[i + 1]
        step_dist = np.linalg.norm(T_curr[:3, 3] - T_next[:3, 3])
        dist_accumulated += step_dist

        frames_passed = i - start_frame + 1
        if frames_passed >= min_frames and (dist_accumulated >= min_dist or frames_passed >= max_frames):
            return i + 1
    return min(start_frame + max_frames, len(poses) - 1)


def build_bundle_graph(db, poses, window_frames, K, P, Q):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    baseline = -Q[0, 3] / fx
    stereo_calib = gtsam.Cal3_S2Stereo(fx, fy, 0.0, cx, cy, baseline)

    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    start_frame = window_frames[0]
    T_ref = np.eye(4)
    T_ref[:3, :] = poses[start_frame][:3, :]

    pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.001] * 6))
    base_measurement_noise = gtsam.noiseModel.Isotropic.Sigma(3, 2.0)
    robust_noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.345),
                                                  base_measurement_noise)

    temp_poses = {}
    for f in window_frames:
        T_f_inv = invert_se3(poses[f])
        T_local = T_ref @ T_f_inv
        temp_poses[f] = gtsam.Pose3(gtsam.Rot3(T_local[:3, :3]), gtsam.Point3(T_local[:3, 3]))

    # חייבים להכניס את הפריים הראשון ואת עוגן ה-Prior שלו
    sym_start = gtsam.symbol('x', start_frame)
    initial_estimate.insert(sym_start, temp_poses[start_frame])
    graph.add(gtsam.PriorFactorPose3(sym_start, temp_poses[start_frame], pose_noise))

    tracks_in_window = set()
    for f in window_frames:
        tracks_in_window.update(db.tracks(f))

    for t_id in tracks_in_window:
        track_frms = [f for f in window_frames if f in db.frames(t_id)]
        if len(track_frms) < 2:
            continue

        first_f = track_frms[0]
        link_first = db.link(first_f, t_id)

        cam_first = gtsam.StereoCamera(temp_poses[first_f], stereo_calib)
        pt3d = cam_first.backproject(gtsam.StereoPoint2(link_first.x_left, link_first.x_right, link_first.y))

        if pt3d[2] < 0.5 or pt3d[2] > 100.0:
            continue

        sym_l = gtsam.symbol('l', t_id)
        if not initial_estimate.exists(sym_l):
            initial_estimate.insert(sym_l, pt3d)

        for f in track_frms:
            sym_x = gtsam.symbol('x', f)

            # --- התיקון המרכזי ---
            # נוסיף את המצלמה ל-Initial Estimate רק אם היא קיימת בטראק תקין
            if not initial_estimate.exists(sym_x):
                initial_estimate.insert(sym_x, temp_poses[f])

            link = db.link(f, t_id)
            meas = gtsam.StereoPoint2(link.x_left, link.x_right, link.y)
            graph.add(gtsam.GenericStereoFactor3D(meas, robust_noise, sym_x, sym_l, stereo_calib))

    return graph, initial_estimate, stereo_calib


def find_worst_projection_factor(graph, values):
    max_err = -1
    worst_factor = None

    for i in range(graph.size()):
        factor = graph.at(i)
        if type(factor) is gtsam.GenericStereoFactor3D:
            err = factor.error(values)
            if err > max_err:
                max_err = err
                worst_factor = factor

    sym_x_worst = worst_factor.keys()[0]
    sym_l_worst = worst_factor.keys()[1]
    frame_c = gtsam.symbolIndex(sym_x_worst)
    point_q = gtsam.symbolIndex(sym_l_worst)

    return worst_factor, max_err, sym_x_worst, sym_l_worst, frame_c, point_q


def plot_factor_projections(values, stereo_calib, frame_c, point_q, sym_x, sym_l, db, left_dir, right_dir,
                            title_prefix):
    pose_c = values.atPose3(sym_x)
    pt_q = values.atPoint3(sym_l)
    cam = gtsam.StereoCamera(pose_c, stereo_calib)

    link = db.link(frame_c, point_q)
    measured_uL, measured_uR, measured_v = link.x_left, link.x_right, link.y

    projected = cam.project(pt_q)
    proj_uL, proj_uR, proj_v = projected.uL(), projected.uR(), projected.v()

    dist_L = np.sqrt((measured_uL - proj_uL) ** 2 + (measured_v - proj_v) ** 2)
    dist_R = np.sqrt((measured_uR - proj_uR) ** 2 + (measured_v - proj_v) ** 2)

    print(f"[{title_prefix}] Distances from measurement: Left = {dist_L:.2f}px, Right = {dist_R:.2f}px")

    img_L = cv2.imread(f"{left_dir}/{frame_c:06d}.png", cv2.IMREAD_GRAYSCALE)
    img_R = cv2.imread(f"{right_dir}/{frame_c:06d}.png", cv2.IMREAD_GRAYSCALE)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    patch_size = 30

    def plot_patch(ax, img, m_u, m_v, p_u, p_v, title):
        y_min, y_max = max(0, int(m_v - patch_size)), min(img.shape[0], int(m_v + patch_size))
        x_min, x_max = max(0, int(m_u - patch_size)), min(img.shape[1], int(m_u + patch_size))

        ax.imshow(img[y_min:y_max, x_min:x_max], cmap='gray')
        ax.plot(m_u - x_min, m_v - y_min, 'go', label='Measured', markersize=8, fillstyle='none', markeredgewidth=2)
        ax.plot(p_u - x_min, p_v - y_min, 'rx', label='Projected', markersize=8, markeredgewidth=2)
        ax.set_title(title)
        ax.legend()
        ax.axis('off')

    plot_patch(axes[0], img_L, measured_uL, measured_v, proj_uL, proj_v, f"Left Image (Frame {frame_c})")
    plot_patch(axes[1], img_R, measured_uR, measured_v, proj_uR, proj_v, f"Right Image (Frame {frame_c})")
    plt.suptitle(f"{title_prefix} - Frame {frame_c}, Track {point_q}\nErrors: L={dist_L:.2f}px, R={dist_R:.2f}px")
    plt.show()

def plot_trajectory_and_landmarks(result, window_frames):
    # import gtsam
    fig = plt.figure(figsize=(15, 6))

    ax2d = fig.add_subplot(121)
    points_2d_x, points_2d_z = [], []
    for val_key in result.keys():
        if gtsam.symbolChr(val_key) == ord('l'):
            pt = result.atPoint3(val_key)
            points_2d_x.append(pt[0])
            points_2d_z.append(pt[2])

    poses_2d_x, poses_2d_z = [], []
    for f in window_frames:
        pose = result.atPose3(gtsam.symbol('x', f))
        poses_2d_x.append(pose.x())
        poses_2d_z.append(pose.z())

    ax2d.scatter(points_2d_x, points_2d_z, s=1, c='gray', alpha=0.5, label='Landmarks')
    ax2d.plot(poses_2d_x, poses_2d_z, 'r-o', label='Trajectory', linewidth=2)
    ax2d.set_title("2D View from Above (X-Z Plane)")
    ax2d.set_xlabel("X (meters)")
    ax2d.set_ylabel("Z (meters)")
    ax2d.axis('equal')
    ax2d.legend()

    ax3d = fig.add_subplot(122, projection='3d')

    poses_3d = np.array([result.atPose3(gtsam.symbol('x', f)).translation() for f in window_frames])
    ax3d.plot(poses_3d[:, 0], poses_3d[:, 1], poses_3d[:, 2], 'r-o', label='Trajectory', linewidth=2)

    points_3d = np.array([[result.atPoint3(k)[0], result.atPoint3(k)[1], result.atPoint3(k)[2]]
                          for k in result.keys() if gtsam.symbolChr(k) == ord('l')])
    if len(points_3d) > 0:
        ax3d.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], s=1, c='gray', alpha=0.5, label='Landmarks')

    ax3d.set_title("3D Trajectory & Landmarks")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")

    plt.tight_layout()
    plt.show()


def q5_3(db, poses, left_images_dir, right_images_dir, K, P, Q):
    # poses = load_kitti_poses(poses_path)
    start_frame = 0
    end_frame = get_keyframe(poses, start_frame)
    window_frames = list(range(start_frame, end_frame + 1))
    print(f"Bundle Window: Frames {start_frame} to {end_frame} (Total: {len(window_frames)})")

    graph, initial_estimate, stereo_calib = build_bundle_graph(db, poses, window_frames, K, P, Q)

    initial_error = graph.error(initial_estimate)
    print(f"\n--- Before Optimization ---")
    print(f"Total Factors: {graph.size()}")
    print(f"Total Initial Graph Error: {initial_error:.4f}")
    print(f"Average Factor Error: {initial_error / graph.size():.4f}")

    worst_factor, max_err, sym_x, sym_l, frame_c, point_q = find_worst_projection_factor(graph, initial_estimate)
    print(f"\nWorst Factor -> Frame: {frame_c}, Point ID: {point_q}")
    print(f"Initial Error of worst factor: {max_err:.4f}")

    plot_factor_projections(initial_estimate, stereo_calib, frame_c, point_q, sym_x, sym_l,
                            db, left_images_dir, right_images_dir, "INITIAL ESTIMATE")

    print("\nRunning Levenberg-Marquardt Optimizer...")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
    result = optimizer.optimize()

    final_error = graph.error(result)
    print(f"\n--- After Optimization ---")
    print(f"Total Final Graph Error: {final_error:.4f}")
    print(f"Average Factor Error: {final_error / graph.size():.4f}")
    print(f"Final Error of previously worst factor: {worst_factor.error(result):.4f}")

    plot_factor_projections(result, stereo_calib, frame_c, point_q, sym_x, sym_l,
                            db, left_images_dir, right_images_dir, "OPTIMIZED ESTIMATE")

    plot_trajectory_and_landmarks(result, window_frames)


def run_window_bundle_adjustment(db, poses, start_frame, end_frame, K, P, Q):
    """
    Builds and optimizes a bundle graph for a specific window of frames.

    Args:
        db, poses, K, P, Q: Same as before.
        start_frame (int): The index of the first frame in the window (c0).
        end_frame (int): The index of the last frame in the window (ck).

    Returns:
        tuple: (graph, result)
    """
    window_frames = list(range(start_frame, end_frame + 1))

    graph, initial_estimate, stereo_calib = build_bundle_graph(db, poses, window_frames, K, P, Q)

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
    result = optimizer.optimize()

    return graph, result

def q5_4(db, poses, ground_truth_path, K, P, Q):
    # poses = load_kitti_poses(poses_path)
    gt_poses = load_kitti_poses(ground_truth_path)

    current_start = 0
    all_optimized_poses = {0: poses[0]}  # מילון ששומר פוזה גלובלית לכל פריים
    keyframes_indices = [0]

    print("--- Starting Global Bundle Adjustment ---")

    while current_start < len(poses) - 5:
        end = get_keyframe(poses, current_start)
        window = list(range(current_start, end + 1))

        # בונים גרף
        graph, initial, calib = build_bundle_graph(db, poses, window, K, P, Q)

        if graph.size() == 0:
            # ---> הגיבוי הקריטי הראשון <---
            for f in window:
                if f not in all_optimized_poses:
                    all_optimized_poses[f] = poses[f]
            current_start = end
            keyframes_indices.append(current_start)
            continue

        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
        result = optimizer.optimize()

        # הדפסת שגיאת פקטור העוגן (עבור הדו"ח)
        prior_error = 0.0
        for i in range(graph.size()):
            if isinstance(graph.at(i), gtsam.PriorFactorPose3):
                prior_error = graph.at(i).error(result)
                break

        # נדפיס רק עבור חלונות אחרונים או לפי צורך
        if current_start > len(poses) - 50:
            print(f"Window {current_start}-{end}: Anchoring factor error = {prior_error}")

        T_start_global = np.eye(4)
        T_start_global[:3, :] = all_optimized_poses[current_start][:3, :]

        for f in window:
            sym = gtsam.symbol('x', f)
            if result.exists(sym):
                pose_local = result.atPose3(sym).matrix()
                pose_local_inv = invert_se3(pose_local)
                T_f_global = pose_local_inv @ T_start_global

                all_optimized_poses[f] = T_f_global
                # poses[f] = T_f_global
            else:
                if f not in all_optimized_poses:
                    T_local_pnp = poses[current_start] @ invert_se3(poses[f])
                    all_optimized_poses[f] = invert_se3(T_local_pnp) @ T_start_global

        current_start = end
        keyframes_indices.append(current_start)

    # --- הדרישה לדו"ח: מיקום הפריים הראשון של החלון האחרון ---
    last_start_frame = keyframes_indices[-2] if len(keyframes_indices) > 1 else keyframes_indices[0]
    T_last_window_start = all_optimized_poses[last_start_frame]
    R_last = T_last_window_start[:3, :3]
    t_last = T_last_window_start[:3, 3]
    position_last = -R_last.T @ t_last
    print(f"\nFinal position of the first frame of the last bundle (Frame {last_start_frame}):")
    print(f"[X: {position_last[0]:.6f}, Y: {position_last[1]:.6f}, Z: {position_last[2]:.6f}]")

    # --- יצירת גרפים וניתוח ---
    est_positions = []
    gt_positions = []

    for i in keyframes_indices:
        T_est = all_optimized_poses[i]
        C_est = -T_est[:3, :3].T @ T_est[:3, 3]
        est_positions.append(C_est)

        T_gt = gt_poses[i]
        C_gt = -T_gt[:3, :3].T @ T_gt[:3, 3]
        gt_positions.append(C_gt)

    est_positions = np.array(est_positions)
    gt_positions = np.array(gt_positions)

    # --- מחיקת השורה של align_trajectories! ---

    # --- הצגת מפה גלובלית 2D ---
    plt.figure(figsize=(10, 8))
    # מציירים ישירות את הקואורדינטות של X ו-Z מה-Ground Truth
    plt.plot(gt_positions[:, 0], gt_positions[:, 2], 'k--', label='Ground Truth', linewidth=2)
    # מציירים ישירות את האומדן שלנו
    plt.plot(est_positions[:, 0], est_positions[:, 2], 'b-', label='Estimated Trajectory', linewidth=2)

    # סימון נקודת ההתחלה
    plt.scatter(est_positions[0, 0], est_positions[0, 2], c='red', marker='*', s=200, label='Start (0,0)', zorder=5)

    plt.title("Trajectory: Estimated vs Ground Truth (Top-Down View)")
    plt.xlabel("X (Right/Left) [meters]")
    plt.ylabel("Z (Forward Depth) [meters]")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()

    # --- הצגת שגיאת לוקליזציה אוקלידית לאורך זמן ---
    euclidean_errors = np.linalg.norm(est_positions[:, [0, 2]] - gt_positions[:, [0, 2]], axis=1)

    plt.figure(figsize=(10, 5))
    plt.plot(keyframes_indices, euclidean_errors, 'm-o', markersize=4)
    plt.title("Keyframe Localization Error (Euclidean distance)")
    plt.xlabel("Keyframe ID")
    plt.ylabel("Euclidean Translation Error [meters]")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    project_root = r"C:\university\SHANA 5\semester B\67604-slam\VAN_SLAM\VAN_ex"
    sequence_dir = os.path.join(project_root, 'dataset', 'dataset_2026', 'sequences', '00')
    poses_dir = os.path.join(project_root, 'dataset', 'dataset_2026', 'poses')
    ground_truth_path = os.path.join(poses_dir, '00.txt')

    left_images_dir = os.path.join(sequence_dir, 'image_0')
    right_images_dir = os.path.join(sequence_dir, 'image_1')

    db = TrackingDB()
    db_filename = 'my_tracking_data_ex4'
    poses_npy_filename = 'my_estimated_poses_ex4.npy'
    K, M1, M2 = read_cameras()
    P, Q = K @ M1, K @ M2  # multiply by intrinsic camera matrix
    try:
        db.load(db_filename)
        # טוענים גם את הפוזות שהערכנו מתרגיל 4
        estimated_poses = np.load(poses_npy_filename, allow_pickle=True)
        # הפיכת מערך numpy לרשימה (כדי שיתאים לקוד שלך)
        estimated_poses = list(estimated_poses)
        print("Data and estimated poses loaded successfully from file.")

    except FileNotFoundError:
        print("Data files not found. Running tracking sequence to generate data...")

        all_T, _ = track_full_sequence_ex4(
            sequence_dir=sequence_dir,
            K=K, P=P, Q=Q,
            db=db,
            max_frames=None
        )
        db.serialize(db_filename)

        # מחשבים את הפוזות הגלובליות מהטרנספורמציות היחסיות
        estimated_poses = calculate_global_poses(all_T)

        np.save(poses_npy_filename, estimated_poses)


        ###### 5.1 #######
    print("###### 5.1 #######")
    q5_1(db, estimated_poses, K, P, Q)

    ###### 5.3 #######
    print("###### 5.3 #######")
    q5_3(db, estimated_poses, left_images_dir, right_images_dir, K, P, Q)

    ###### 5.4 #######
    print("###### 5.4 #######")
    q5_4(db, estimated_poses, ground_truth_path, K, P, Q)