import gtsam
from gtsam.utils import plot
import numpy as np
import matplotlib.pyplot as plt
import cv2
from ex4 import *

def q5_1(db, poses_path, K, P, Q):
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
    poses = load_kitti_poses(poses_path)

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
        R_wc = T_wc[:3, :3]
        t_wc = T_wc[:3, 3]

        pose3 = gtsam.Pose3(gtsam.Rot3(R_wc), gtsam.Point3(t_wc))
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
    sigma = 1.0
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


def get_keyframe(poses, start_frame, min_frames=5, max_frames=20, min_dist=2.0):
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


# def build_bundle_graph(db, poses, window_frames, K, P, Q):
#     # Calibration
#     fx, fy = K[0, 0], K[1, 1]
#     cx, cy = K[0, 2], K[1, 2]
#     baseline = -Q[0, 3] / fx
#     stereo_calib = gtsam.Cal3_S2Stereo(fx, fy, 0.0, cx, cy, baseline)
#
#     graph = gtsam.NonlinearFactorGraph()
#     initial_estimate = gtsam.Values()
#
#     pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]))
#     measurement_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
#
#     start_frame = window_frames[0]
#     T_ref_inv = np.linalg.inv(poses[start_frame])
#     stereo_cameras = {}
#
#     # Initialize Poses
#     for f in window_frames:
#         T_local = T_ref_inv @ poses[f]
#         R_local, t_local = T_local[:3, :3], T_local[:3, 3]
#
#         pose3 = gtsam.Pose3(gtsam.Rot3(R_local), gtsam.Point3(t_local))
#         sym_x = gtsam.symbol('x', f)
#         initial_estimate.insert(sym_x, pose3)
#         stereo_cameras[f] = gtsam.StereoCamera(pose3, stereo_calib)
#
#         if f == start_frame:
#             graph.add(gtsam.PriorFactorPose3(sym_x, pose3, pose_noise))
#
#     # Initialize Landmarks and add Projection Factors
#     tracks_in_window = set()
#     for f in window_frames:
#         tracks_in_window.update(db.tracks(f))
#
#     for t_id in tracks_in_window:
#         track_frms = [f for f in window_frames if f in db.frames(t_id)]
#         if len(track_frms) < 2:
#             continue
#
#         sym_l = gtsam.symbol('l', t_id)
#
#         # Triangulate
#         if not initial_estimate.exists(sym_l):
#             first_f = track_frms[0]
#             link = db.link(first_f, t_id)
#             stereo_pt = gtsam.StereoPoint2(link.x_left, link.x_right, link.y)
#             point3d = stereo_cameras[first_f].backproject(stereo_pt)
#             initial_estimate.insert(sym_l, point3d)
#
#         # Add factors
#         for f in track_frms:
#             link = db.link(f, t_id)
#             measured_pt = gtsam.StereoPoint2(link.x_left, link.x_right, link.y)
#             sym_x = gtsam.symbol('x', f)
#             factor = gtsam.GenericStereoFactor3D(measured_pt, measurement_noise, sym_x, sym_l, stereo_calib)
#             graph.add(factor)
#
#     return graph, initial_estimate, stereo_calib

def build_bundle_graph(db, poses, window_frames, K, P, Q):
    # Calibration
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    baseline = -Q[0, 3] / fx
    stereo_calib = gtsam.Cal3_S2Stereo(fx, fy, 0.0, cx, cy, baseline)

    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    start_frame = window_frames[0]
    T_ref_inv = np.linalg.inv(poses[start_frame])

    pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01] * 6))
    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)

    # הכנת המצלמות
    temp_poses = {}
    for f in window_frames:
        T_local = T_ref_inv @ poses[f]
        temp_poses[f] = gtsam.Pose3(gtsam.Rot3(T_local[:3, :3]), gtsam.Point3(T_local[:3, 3]))

    # הוספת Prior למצלמה הראשונה
    sym_start = gtsam.symbol('x', start_frame)
    initial_estimate.insert(sym_start, temp_poses[start_frame])
    graph.add(gtsam.PriorFactorPose3(sym_start, temp_poses[start_frame], pose_noise))

    # הוספת פקטורים ונקודות
    tracks_in_window = set()
    for f in window_frames:
        tracks_in_window.update(db.tracks(f))

    for t_id in tracks_in_window:
        track_frms = [f for f in window_frames if f in db.frames(t_id)]
        if len(track_frms) < 2: continue

        first_f = track_frms[0]
        link_first = db.link(first_f, t_id)
        disparity = link_first.x_left - link_first.x_right

        # סינון 1: נקודות קרובות לאינסוף (מה שזיהית קודם!)
        if disparity < 2.0:
            continue

        # טריאנגולציה
        cam_first = gtsam.StereoCamera(temp_poses[first_f], stereo_calib)
        pt3d = cam_first.backproject(gtsam.StereoPoint2(link_first.x_left, link_first.x_right, link_first.y))

        # --- התיקון החדש: Sanity Check (ה"וייב של RANSAC") ---
        # נבדוק אם הנקודה הזו מייצרת אומדן מטורף באחד הפריימים האחרים
        is_outlier = False
        for f in track_frms:
            cam_f = gtsam.StereoCamera(temp_poses[f], stereo_calib)
            try:
                # מטילים את הנקודה התלת-מימדית למצלמה בפריים f
                proj = cam_f.project(pt3d)
                link_f = db.link(f, t_id)

                # מחשבים שגיאת פיקסלים (אוקלידית) בין ההטלה למדידה
                dist = np.sqrt((proj.uL() - link_f.x_left) ** 2 +
                               (proj.uR() - link_f.x_right) ** 2 +
                               (proj.v() - link_f.y) ** 2)

                # אם השגיאה הראשונית גדולה מ-25 פיקסלים, זה אאוטלייר מסוכן!
                if dist > 25.0:
                    is_outlier = True
                    break
            except Exception:
                # מתרחש אם הנקודה נופלת מאחורי המצלמה (Cheirality Exception)
                is_outlier = True
                break

        if is_outlier:
            continue  # מסננים וזורקים את ה-Track הרעיל!
        # ----------------------------------------------------

        sym_l = gtsam.symbol('l', t_id)

        # הכנסה ל-Values
        if not initial_estimate.exists(sym_l):
            initial_estimate.insert(sym_l, pt3d)

        # הוספת פקטורים
        for f in track_frms:
            sym_x = gtsam.symbol('x', f)
            if not initial_estimate.exists(sym_x):
                initial_estimate.insert(sym_x, temp_poses[f])

            link = db.link(f, t_id)
            meas = gtsam.StereoPoint2(link.x_left, link.x_right, link.y)
            graph.add(gtsam.GenericStereoFactor3D(meas, measurement_noise, sym_x, sym_l, stereo_calib))

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
    import gtsam
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

def q5_3(db, poses_path, left_images_dir, right_images_dir, K, P, Q):
    poses = load_kitti_poses(poses_path)
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


def align_trajectories(est, gt):
    est_mean = est.mean(axis=0)
    gt_mean = gt.mean(axis=0)

    est_centered = est - est_mean
    gt_centered = gt - gt_mean

    H = est_centered.T @ gt_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt_modified = Vt.copy()
        Vt_modified[2, :] *= -1
        R = Vt_modified.T @ U.T

    return (est_centered @ R.T) + gt_mean

def q5_4(db, poses_path, ground_truth_path, K, P, Q):
    poses = load_kitti_poses(poses_path)
    gt_poses = load_kitti_poses(ground_truth_path)

    current_start = 0
    global_trajectory = [poses[0]]
    keyframes_indices = [0]
    all_optimized_poses = {0: poses[0]}

    print("--- Starting Global Bundle Adjustment ---")

    while current_start < len(poses) - 5:
        end = get_keyframe(poses, current_start)
        window = list(range(current_start, end + 1))

        graph, initial, calib = build_bundle_graph(db, poses, window, K, P, Q)

        # בדיקה שהגרף לא ריק
        if graph.size() == 0:
            current_start = end
            continue

        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
        result = optimizer.optimize()

        # תיקון: מציאת ה-Prior האמיתי בגרף במקום להניח שהוא תמיד ב-0
        prior_error = 0.0
        for i in range(graph.size()):
            if isinstance(graph.at(i), gtsam.PriorFactorPose3):
                prior_error = graph.at(i).error(result)
                break
        print(f"Window {current_start}-{end}: Anchoring factor error = {prior_error:.6f}")

        # תיקון: גישה בטוחה ל-Result
        sym_end = gtsam.symbol('x', end)
        if result.exists(sym_end):
            T_next_local = result.atPose3(sym_end).matrix()
            T_prev_global = global_trajectory[-1]
            T_next_global = T_prev_global @ T_next_local
            global_trajectory.append(T_next_global)
            all_optimized_poses[end] = T_next_global
        else:
            # אם הפריים לא באופטימיזציה, נשתמש במיקום המקורי כגיבוי
            print(f"Warning: Frame {end} not in optimization result, using PnP estimate.")
            all_optimized_poses[end] = global_trajectory[-1] @ np.linalg.inv(poses[current_start]) @ poses[end]
            global_trajectory.append(all_optimized_poses[end])

        current_start = end
        keyframes_indices.append(current_start)

    # 4. הדפסת המיקום של הפריים הראשון בחלון האחרון
    print(f"\nLast bundle window start frame ({keyframes_indices[-2]}) global position:\n",
          all_optimized_poses[keyframes_indices[-2]][:3, 3])

    errors = []
    times = keyframes_indices
    est_positions = np.array([all_optimized_poses[i][:3, 3] for i in keyframes_indices])
    gt_positions = np.array([gt_poses[i][:3, 3] for i in keyframes_indices])

    est_positions_aligned = align_trajectories(est_positions, gt_positions)

    errors = np.linalg.norm(est_positions - gt_positions, axis=1)

    plt.figure()
    plt.plot(times, errors, marker='o')
    plt.title("Keyframe Localization Error")
    plt.xlabel("Frame Index")
    plt.ylabel("Euclidean Error (m)")
    plt.show()

    plt.figure()
    plt.plot(est_positions_aligned[:, 0], est_positions_aligned[:, 2], 'r-o', label='Estimated')
    plt.plot(gt_positions[:, 0], gt_positions[:, 2], 'b-o', label='Ground Truth')
    plt.title("Trajectory: Estimated vs Ground Truth (X-Z Plane)")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    project_root = r"C:\university\SHANA 5\semester B\67604-slam\VAN_SLAM\VAN_ex"
    sequence_dir = os.path.join(project_root, 'dataset', 'dataset_2026', 'sequences', '00')

    left_images_dir = os.path.join(sequence_dir, 'image_0')
    db = TrackingDB()
    db_filename = 'my_tracking_data'
    K, M1, M2 = read_cameras()
    P, Q = K @ M1, K @ M2  # multiply by intrinsic camera matrix
    try:
        db.load(db_filename)
        print("Data loaded successfully from file.")
    except FileNotFoundError:
        print("Data file not found. Running tracking sequence to generate data...")

        track_full_sequence_ex4(
            sequence_dir=sequence_dir,
            K=K, P=P, Q=Q,
            db=db
        )#, max_frames=10
        db.serialize(db_filename)

    project_root = r"C:\university\SHANA 5\semester B\67604-slam\VAN_SLAM\VAN_ex"
    sequence_dir = os.path.join(project_root, 'dataset', 'dataset_2026', 'poses')
    poses_path = os.path.join(sequence_dir,'00.txt')

    ###### 5.1 #######
    # print("###### 5.1 #######")
    # q5_1(db, poses_path, K, P, Q)
    ###### end of 5.1 #######

    right_images_dir = os.path.join(sequence_dir, 'image_1')

    ###### 5.3 #######
    # print("###### 5.3 #######")
    # sequence_dir = os.path.join(project_root, 'dataset', 'dataset_2026', 'sequences', '00')
    # left_images_dir = os.path.join(sequence_dir, 'image_0')
    # right_images_dir = os.path.join(sequence_dir, 'image_1')
    #
    # q5_3(db, poses_path, left_images_dir, right_images_dir, K, P, Q)
    ###### end of 5.3 #######

    ground_truth_path = r"C:\university\SHANA 5\semester B\67604-slam\VAN_SLAM\VAN_ex\dataset\dataset_2026\poses\00.txt"
    print("###### 5.4 #######")
    q5_4(db, poses_path, ground_truth_path, K, P, Q)
    print("###### End of 5.4 #######")