from tracking_database import *
from PnP_RANSAC_ex2 import *
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from matplotlib.patches import Rectangle

#############
def track_full_sequence_ex4(sequence_dir, K, P, Q, db, log_filename="movement_log.csv", max_frames=None):
    print("\nStarting Full Sequence Tracking...")
    start_time = time.time()

    img_0_dir = os.path.join(sequence_dir, 'image_0')
    num_frames = len([name for name in os.listdir(img_0_dir) if name.endswith('.png')])

    if max_frames is not None:
        num_frames = min(num_frames, max_frames)

    print(f"Found {num_frames} frames to process.")
    all_T = []

    with open(log_filename, 'w') as log_file:
        log_file.write("Frame,Distance_Meters,Status\n")

        # --- Frame 0 ---
        prev_left_img, prev_left_kp, prev_left_desc, prev_inliers, prev_points_3d, prev_right_kp, prev_stereo_matches = process_single_point_cloud(
            0)

        prev_is_valid = [False] * len(prev_left_kp)
        prev_stereo_inliers_bool = [False] * len(prev_stereo_matches)
        prev_inlier_set = {(m.queryIdx, m.trainIdx) for m in prev_inliers}

        for i, m in enumerate(prev_stereo_matches):
            if (m.queryIdx, m.trainIdx) in prev_inlier_set:
                prev_stereo_inliers_bool[i] = True
                prev_is_valid[m.queryIdx] = True

        slam_to_db_prev = {}
        idx = 0
        for i, valid in enumerate(prev_is_valid):
            if valid:
                slam_to_db_prev[i] = idx
                idx += 1

        valid_features_prev, links_prev = TrackingDB.create_links(
            features=prev_left_desc,
            kp_left=prev_left_kp,
            kp_right=prev_right_kp,
            matches=prev_stereo_matches,
            inliers=prev_stereo_inliers_bool
        )
        db.add_frame(links=links_prev, left_features=valid_features_prev)

        for i in tqdm(range(1, num_frames), desc="Tracking Progress", unit="frame"):
            curr_left_img, curr_left_kp, curr_left_desc, curr_inliers, curr_points_3d, curr_right_kp, curr_stereo_matches = process_single_point_cloud(
                i)

            curr_is_valid = [False] * len(curr_left_kp)
            curr_stereo_inliers_bool = [False] * len(curr_stereo_matches)
            curr_inlier_set = {(m.queryIdx, m.trainIdx) for m in curr_inliers}

            for j, m in enumerate(curr_stereo_matches):
                if (m.queryIdx, m.trainIdx) in curr_inlier_set:
                    curr_stereo_inliers_bool[j] = True
                    curr_is_valid[m.queryIdx] = True

            slam_to_db_curr = {}
            idx = 0
            for j, valid in enumerate(curr_is_valid):
                if valid:
                    slam_to_db_curr[j] = idx
                    idx += 1

            temporal_matches = left_matches(prev_left_desc, curr_left_desc)
            obj_points_3d, img_points_2d = get_pnp_data(prev_inliers, temporal_matches, prev_points_3d, curr_left_kp)

            T_refined = np.hstack((np.eye(3), np.zeros((3, 1))))
            success_flag = False
            step_distance = 0.0
            frame_status = "Failed PnP/RANSAC"

            temporal_inliers_bool = [False] * len(temporal_matches)

            if len(obj_points_3d) >= 4:
                best_R, best_t, best_supporters = RANSAC(
                    prev_inliers, temporal_matches, curr_inliers, prev_points_3d,
                    prev_left_kp, prev_right_kp, curr_left_kp, curr_right_kp,
                    P, Q, K, obj_points_3d, img_points_2d, p=0.99, threshold=2.0
                )

                if best_R is not None and len(best_supporters) >= 4:
                    R_refined, t_refined, T_calculated = calc_the_transformations_to_RANSAC_inliers(
                        best_supporters, prev_inliers, prev_points_3d, curr_left_kp, K
                    )

                    if T_calculated is not None:
                        step_distance = np.linalg.norm(T_calculated[:, 3])
                        if step_distance < 5.0:
                            T_refined = T_calculated
                            success_flag = True
                            frame_status = "Success"

                            match_to_idx = {(m.queryIdx, m.trainIdx): idx for idx, m in enumerate(temporal_matches)}
                            for supp_match in best_supporters:
                                key = (supp_match.queryIdx, supp_match.trainIdx)
                                if key in match_to_idx:
                                    temporal_inliers_bool[match_to_idx[key]] = True
                        else:
                            frame_status = "Rejected Refinement"

            all_T.append(T_refined)
            log_file.write(f"{i},{step_distance:.6f},{frame_status}\n")

            prev_db_size = valid_features_prev.shape[0]
            db_matches = [cv2.DMatch(j, 0, float('inf')) for j in range(prev_db_size)]
            db_inliers = [False] * prev_db_size

            for k, m in enumerate(temporal_matches):
                if temporal_inliers_bool[k]:  # אם RANSAC אישר שזה עקיבה טובה
                    orig_prev = m.queryIdx
                    orig_curr = m.trainIdx

                    if orig_prev in slam_to_db_prev and orig_curr in slam_to_db_curr:
                        db_prev = slam_to_db_prev[orig_prev]
                        db_curr = slam_to_db_curr[orig_curr]
                        db_matches[db_prev] = cv2.DMatch(db_prev, db_curr, m.distance)
                        db_inliers[db_prev] = True

            valid_features_curr, links_curr = TrackingDB.create_links(
                features=curr_left_desc,
                kp_left=curr_left_kp,
                kp_right=curr_right_kp,
                matches=curr_stereo_matches,
                inliers=curr_stereo_inliers_bool
            )

            db.add_frame(
                links=links_curr,
                left_features=valid_features_curr,
                matches_to_previous_left=db_matches,
                inliers=db_inliers
            )

            prev_left_img = curr_left_img
            prev_left_kp = curr_left_kp
            prev_left_desc = curr_left_desc
            prev_right_kp = curr_right_kp
            prev_inliers = curr_inliers
            prev_points_3d = curr_points_3d
            prev_stereo_matches = curr_stereo_matches

            slam_to_db_prev = slam_to_db_curr
            valid_features_prev = valid_features_curr

    tracking_time = time.time() - start_time
    print(f"\nTracking completed! Total time: {tracking_time:.2f} seconds.")
    return all_T, tracking_time

############3
def q4_3(db, images_dir_path):
    target_track_id = None
    for track_id, frames in db.trackId_to_frames.items():
        if len(frames) >= 6:
            target_track_id = track_id
            break

    if target_track_id is None:
        return

    track_frames = db.frames(target_track_id)
    n_frames = len(track_frames)

    fig, axes = plt.subplots(nrows=n_frames, ncols=2, figsize=(15, 3 * n_frames))

    for i, frame_id in enumerate(track_frames):
        img_path = os.path.join(images_dir_path, f'{frame_id:06d}.png')
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Error loading image: {img_path}")
            continue

        link = db.link(frame_id, target_track_id)
        x, y = int(link.x_left), int(link.y)

        half_size = 10
        y_min = max(0, y - half_size)
        y_max = min(img.shape[0], y + half_size)
        x_min = max(0, x - half_size)
        x_max = min(img.shape[1], x + half_size)

        patch = img[y_min:y_max, x_min:x_max]

        ax_full = axes[i, 0]
        ax_full.imshow(img, cmap='gray')
        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                         linewidth=1, edgecolor='r', facecolor='none')
        ax_full.add_patch(rect)
        ax_full.set_title(f'Frame {frame_id} - Full Image')
        ax_full.axis('off')

        ax_patch = axes[i, 1]
        ax_patch.imshow(patch, cmap='gray')
        rel_x = x - x_min
        rel_y = y - y_min
        ax_patch.plot(rel_x, rel_y, 'rx')
        ax_patch.set_title(f'Frame {frame_id} - Patch')
        ax_patch.axis('off')

    plt.suptitle(f'Track #{target_track_id}', fontsize=16)
    plt.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.show()


def q4_4(db):
    outgoing_tracks_counts = []
    frames = list(db.all_frames())

    if len(frames) < 2:
        return

    for i in range(len(frames) - 1):
        current_frame = frames[i]
        next_frame = frames[i + 1]

        tracks_current = set(db.tracks(current_frame))
        tracks_next = set(db.tracks(next_frame))

        outgoing = len(tracks_current.intersection(tracks_next))
        outgoing_tracks_counts.append(outgoing)

    x_frames = frames[:-1]

    mean_outgoing = np.mean(outgoing_tracks_counts)

    plt.figure(figsize=(15, 5))
    plt.plot(x_frames, outgoing_tracks_counts, label='Outgoing Tracks', color='tab:blue')
    plt.axhline(y=mean_outgoing, color='darkgreen', linestyle='-', linewidth=1.5,
                label=f'Mean: {mean_outgoing:.1f}')

    plt.title('Connectivity')
    plt.xlabel('frame')
    plt.ylabel('outgoing tracks')
    plt.xlim(0, x_frames[-1])
    plt.ylim(bottom=0)

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def q4_5(db):
    inlier_percentages = []
    frames = list(db.all_frames())

    if len(frames) < 2:
        return

    for i in range(len(frames) - 1):
        current_frame = frames[i]
        next_frame = frames[i + 1]

        tracks_current = set(db.tracks(current_frame))
        tracks_next = set(db.tracks(next_frame))
        inliers_count = len(tracks_current.intersection(tracks_next))

        total_features = db.frameId_to_lfeature[current_frame].shape[0]

        if total_features > 0:
            percent = (inliers_count / total_features) * 100.0
        else:
            percent = 0.0

        inlier_percentages.append(percent)

    x_frames = frames[1:]

    mean_percent = np.mean(inlier_percentages)

    plt.figure(figsize=(15, 5))
    plt.plot(x_frames, inlier_percentages, label='Inliers %', color='tab:orange')
    plt.axhline(y=mean_percent, color='red', linestyle='--', linewidth=1.5,
                label=f'Mean: {mean_percent:.1f}%')

    plt.title('Percentage of Inliers per Frame')
    plt.xlabel('frame')
    plt.ylabel('inliers (%)')

    plt.xlim(0, x_frames[-1])
    plt.ylim(0, 100)

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def q4_6(db):
    if not db.trackId_to_frames:
        return

    lengths = [len(frames) for frames in db.trackId_to_frames.values()]

    max_len = max(lengths)

    bins = np.arange(2, max_len + 2) - 0.5

    plt.figure(figsize=(10, 6))

    plt.hist(lengths, bins=bins, log=True, color='tab:blue')

    plt.title('Track length histogram')
    plt.xlabel('Track length')
    plt.ylabel('Track #')

    plt.grid(axis='y', alpha=0.3)
    plt.xlim(left=1)
    plt.tight_layout()
    plt.show()

#########4.7
def load_kitti_poses(filepath):
    poses = []
    with open(filepath, 'r') as f:
        for line in f:
            data = np.fromstring(line, dtype=float, sep=' ')
            pose = np.vstack((data.reshape(3, 4), [0, 0, 0, 1]))
            poses.append(pose)
    return poses


def q4_7_part1(db, poses_path, P, Q):
    target_track = None
    for track_id, frames in db.trackId_to_frames.items():
        if len(frames) >= 10:
            target_track = track_id
            break

    if target_track is None:
        return

    track_frames = db.frames(target_track)
    last_frame = track_frames[-1]

    poses = load_kitti_poses(poses_path)

    last_link = db.link(last_frame, target_track)
    pt_left = np.array([[last_link.x_left], [last_link.y]], dtype=float)
    pt_right = np.array([[last_link.x_right], [last_link.y]], dtype=float)

    pt4d_local = cv2.triangulatePoints(P, Q, pt_left, pt_right)
    pt3d_local = pt4d_local[:3] / pt4d_local[3]
    pt4d_local_hom = np.vstack((pt3d_local, [1.0]))

    T_last = poses[last_frame]
    pt4d_world = T_last @ pt4d_local_hom

    errors_left = []
    errors_right = []
    distances = []

    for frame in reversed(track_frames):
        dist = last_frame - frame
        distances.append(dist)

        T_curr = poses[frame]
        T_curr_inv = np.linalg.inv(T_curr)
        pt4d_curr_local = T_curr_inv @ pt4d_world

        proj_left = P @ pt4d_curr_local
        proj_right = Q @ pt4d_curr_local

        pl_x, pl_y = float(proj_left[0] / proj_left[2]), float(proj_left[1] / proj_left[2])
        pr_x, pr_y = float(proj_right[0] / proj_right[2]), float(proj_right[1] / proj_right[2])

        link = db.link(frame, target_track)

        err_left = np.sqrt((pl_x - link.x_left) ** 2 + (pl_y - link.y) ** 2)
        err_right = np.sqrt((pr_x - link.x_right) ** 2 + (pr_y - link.y) ** 2)

        errors_left.append(err_left)
        errors_right.append(err_right)

    plt.figure(figsize=(10, 6))
    plt.plot(distances, errors_left, label='Left', color='tab:blue')
    plt.plot(distances, errors_right, label='Right', color='tab:orange')

    plt.title(f'PnP - projection error vs track length (Track #{target_track})')
    plt.xlabel('distance from reference (frames)')
    plt.ylabel('projection error (pixels)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    plt.tight_layout()
    plt.show()


#########

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

    ###### 4.2 #######3
    print("###### 4.2 #######3")
    print(f"Total Tracks: {db.get_total_tracks()}")
    print(f"Total Frames: {db.get_total_frames()}")
    mean_len, max_len, min_len = db.get_mean_track_length()
    print(f"Track Lengths - Mean: {mean_len:.2f}, Max: {max_len}, Min: {min_len}")
    print(f"Mean links per frame: {db.get_mean_frame_links():.2f}")

    ###### end of 4.2 #######3

    ###### 4.3 #######3
    print("###### 4.3 #######3")
    q4_3(db, left_images_dir)
    ####### end 4.3 #######

    ###### 4.4 #######3
    print("###### 4.4 #######3")
    q4_4(db)
    ####### end 4.4 #######

    ###### 4.5 #######
    print("###### 4.5 #######")
    q4_5(db)
    ###### end of 4.5 #######

    ###### 4.6 #######
    print("###### 4.6 #######")
    q4_6(db)
    ###### end of 4.6 #######

    ###### 4.7 #######
    print("###### 4.7 #######")
    project_root = r"C:\university\SHANA 5\semester B\67604-slam\VAN_SLAM\VAN_ex"
    sequence_dir = os.path.join(project_root, 'dataset', 'dataset_2026', 'poses')

    poses_path = os.path.join(sequence_dir,'00.txt')
    q4_7_part1(db, poses_path, P, Q)
    ###### end of 4.7 #######

