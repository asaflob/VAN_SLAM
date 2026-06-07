from geometric_outline_rejection_ex1 import *
import math
import time
from tqdm import tqdm
#def read_and_extract_matches(index=0)
#return left_img, right_img, left_kp, right_kp, left_desc, desc_right,matches

#def filter_rectified_matches(img_left, img_right, matches, kp_left, kp_right,
#                                     title="Rectified Stereo Pattern Rejection"):
#return inliers, outliers

#def cv_triangulate_matched_points(inliers, kp_left, kp_right, P, Q):
#return X

################# debug functions ####################
def debug_plot_matches(img1, kp1, img2, kp2, matches, title="Matches Debug", num_to_show=15):
    """
    Draws lines connecting a subset of matched keypoints between two images.
    """
    # בדיקה האם יש יותר התאמות ממה שביקשנו להציג, ואם כן - לוקחים מדגם אקראי
    if len(matches) > num_to_show:
        matches_to_draw = random.sample(matches, num_to_show)
    else:
        matches_to_draw = matches

    # יצירת תמונה שמחברת את שתי התמונות ומציירת קווים
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches_to_draw, None,
                                  matchColor=(0, 255, 0),  # קווים ירוקים
                                  singlePointColor=None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(20, 8))
    plt.imshow(img_matches, cmap='gray')
    # עדכנו את הכותרת שתראה כמה מתוך כמה אנחנו מציגים
    plt.title(f"{title} - Showing {len(matches_to_draw)} out of {len(matches)} matches")
    plt.axis('off')
    plt.show()
#########################################################################

##### 3_1 #####

def process_single_point_cloud(index):
    """
        Processes a single stereo pair: extracts features, filters outliers, and triangulates.
        Returns the data needed for future frames.
    """

    #extract feature points
    left_img, right_img, left_kp, right_kp, left_desc, desc_right, matches = read_and_extract_matches(index)

    #the inliers and outliers after filter
    inliers, outliers = filter_rectified_matches(matches, left_kp, right_kp)

    #
    triangulate_points = cv_triangulate_matched_points(inliers, left_kp, right_kp, P, Q)

    return left_img, left_kp, left_desc, inliers, triangulate_points, right_kp, matches

def main_3_1():
    print("doing 3_1")

    #for the pair (left_0, right_0)
    left_img_0, left_kp_0, left_desc_0, inliers_0, triangulate_points_0, right_kp_0 = process_single_point_cloud(0)

    #for the pair (left_1, right_1)
    left_img_1, left_kp_1, left_desc_1, inliers_1, triangulate_points_1, right_kp_1 = process_single_point_cloud(1)

    print(f"Generated 3D point cloud for pair 0 with {len(triangulate_points_0)} points.")
    print(f"Generated 3D point cloud for pair 1 with {len(triangulate_points_1)} points.")

    return left_img_0, left_kp_0, left_desc_0, triangulate_points_0, inliers_0, right_kp_0,\
        left_img_1, left_kp_1, left_desc_1, triangulate_points_1, inliers_1, right_kp_1


##### 3_2 #####

def left_matches(left_desc_0,left_desc_1,ratio=0.95):
    """
        Matches descriptors between two images that are not a rectified stereo pair
        using KNN matching and Lowe's ratio test.
    """
    raw_matches = MATCHER.knnMatch(left_desc_0, left_desc_1, k=2)
    good_matches = []
    for m, n in raw_matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    print(f"Found {len(good_matches)} good matches between left_0 and left_1")
    return good_matches

def main_3_2():
    print("doing 3_2")

    left_img_0, left_kp_0, left_desc_0, triangulate_points_0, inliers_0, right_kp_0, \
        left_img_1, left_kp_1, left_desc_1, triangulate_points_1, inliers_1, right_kp_1 = main_3_1()

    temporal_matches = left_matches(left_desc_0, left_desc_1)
    # debug_plot_matches(left_img_0, left_kp_0, left_img_1, left_kp_1, temporal_matches,
    #                    "Temporal Matches (Left0 to Left1)")
    return temporal_matches


##### 3_3 #####
def get_pnp_data(inliers_0, temporal_matches, points_3d_0, kp_left_1):
    """
    Finds points that have both a 3D coordinate (from pair 0)
    and a 2D pixel location in left_1.
    """
    obj_points_3d = []
    img_points_2d = []
    max_depth = 150.0


    dict_3d = {}
    for i, match in enumerate(inliers_0):
        idx_left_0 = match.queryIdx
        dict_3d[idx_left_0] = points_3d_0[i]

    for match in temporal_matches:
        idx_left_0 = match.queryIdx
        idx_left_1 = match.trainIdx

        if idx_left_0 in dict_3d:
            point_3d = dict_3d[idx_left_0]
            z_depth = point_3d[2]  # Extract Z coordinate (depth)


            # Keep only points in front of the camera and within realistic bounds
            if 0 < z_depth < max_depth: #todo check if need more than 0.1
                obj_points_3d.append(point_3d)
                img_points_2d.append(kp_left_1[idx_left_1].pt)


    obj_points_3d = np.array(obj_points_3d, dtype=np.float32)
    img_points_2d = np.array(img_points_2d, dtype=np.float32)
    print(f"Found {len(img_points_2d)} 3D points.")
    print(f"Found {len(obj_points_3d)} points matched across all 4 images.")
    return obj_points_3d, img_points_2d


def calculate_pnp(obj_points_3d, img_points_2d, K, quiet=True):
    """
    Applies PnP on 4 matched points to calculate the extrinsic matrix [R|t].
    """
    subset_3d = obj_points_3d[:4]
    subset_2d = img_points_2d[:4]

    # todo this is 4 points, need to change
    success, rvec, tvec = cv2.solvePnP(subset_3d, subset_2d, K, distCoeffs=None, flags=cv2.SOLVEPNP_AP3P)

    if not success:
        if not quiet:
            print("PnP failed to find a solution.")
        return None, None, None

    R, _ = cv2.Rodrigues(rvec)

    Rt_matrix = np.hstack((R, tvec))

    if not quiet:
        print("Extrinsic Camera Matrix [R|t] for left_1:")
        print(Rt_matrix)

    return R, tvec, Rt_matrix


def plot_cameras_top_down(R, tvec, baseline=-0.54):
    """
    Plots the relative positions of the 4 cameras from a top-down view (X-Z plane).
    Inverts the X-axis for visualization so 'Right' is visually on the right.
    """
    # המיקומים נשארים בדיוק לפי המתמטיקה המקורית של PnP
    C_left_0 = np.array([0.0, 0.0, 0.0])
    C_right_0 = np.array([baseline, 0.0, 0.0])

    #  C = -R^T * t
    R_transpose = R.T
    C_left_1 = -np.dot(R_transpose, tvec).flatten()

    C_right_1 = C_left_1 + np.array([baseline, 0.0, 0.0])

    plt.figure(figsize=(10, 8))

    cameras = {
        "left_0": (C_left_0, 'blue', '^'),
        "right_0": (C_right_0, 'lightblue', '^'),
        "left_1": (C_left_1, 'red', '^'),
        "right_1": (C_right_1, 'lightcoral', '^')
    }

    for name, (pos, color, marker) in cameras.items():
        plt.scatter(pos[0], pos[2], color=color, marker=marker, s=200, label=name)
        plt.text(pos[0] + 0.02, pos[2] + 0.02, name, fontsize=12)

    plt.plot([C_left_0[0], C_right_0[0]], [C_left_0[2], C_right_0[2]], 'k--', alpha=0.5)
    plt.plot([C_left_1[0], C_right_1[0]], [C_left_1[2], C_right_1[2]], 'k--', alpha=0.5)

    plt.plot([C_left_0[0], C_left_1[0]], [C_left_0[2], C_left_1[2]], 'g-', linewidth=2, label='Trajectory')

    # --- התיקון הקריטי: הפיכת ציר ה-X תצוגתית ---
    plt.gca().invert_xaxis()
    # ---------------------------------------------

    plt.title('Top-Down View of Cameras (X-Z Plane) [cite: 1]')
    plt.xlabel('X (Reversed: Left -> Right) [meters]')
    plt.ylabel('Z (Forward Depth) [meters]')

    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.show()
##### 3_4 #####
def project_3d_to_2d(X_3d, proj_matrix):
    """
    Helper function: Projects a 3D point back to 2D pixel coordinates
    using a projection matrix (P or Q).
    """
    X_homog = np.append(X_3d, 1.0)

    uvw = proj_matrix @ X_homog

    u = uvw[0] / uvw[2]
    v = uvw[1] / uvw[2]
    return np.array([u, v])


def matches_and_supports(inliers_0, temporal_matches, inliers_1, points_3d_0,
                         kp_left_0, kp_right_0, kp_left_1, kp_right_1,
                         P, Q, R, t, threshold=2.0, quiet=True):
    """
    Finds supporters for the transformation [R|t].
    A supporter is a point whose reprojection error is < 2 pixels in all 4 images.
    """
    supporters = []
    all_four_matches = []
    dict_temporal = {m.queryIdx: m.trainIdx for m in temporal_matches}
    dict_inliers_1 = {m.queryIdx: m.trainIdx for m in inliers_1}

    for i, match_0 in enumerate(inliers_0):
        idx_L0 = match_0.queryIdx
        idx_R0 = match_0.trainIdx

        if idx_L0 in dict_temporal:
            idx_L1 = dict_temporal[idx_L0]

            if idx_L1 in dict_inliers_1:
                idx_R1 = dict_inliers_1[idx_L1]

                all_four_matches.append((idx_L0, idx_R0, idx_L1, idx_R1))

                X_3d = points_3d_0[i]  # הנקודה התלת-ממדית המקורית

                pt_actual_L0 = np.array(kp_left_0[idx_L0].pt)
                pt_actual_R0 = np.array(kp_right_0[idx_R0].pt)
                pt_actual_L1 = np.array(kp_left_1[idx_L1].pt)
                pt_actual_R1 = np.array(kp_right_1[idx_R1].pt)

                pt_proj_L0 = project_3d_to_2d(X_3d, P)
                pt_proj_R0 = project_3d_to_2d(X_3d, Q)

                # X_new = R * X + t
                X_new_3d = (R @ X_3d) + t.flatten()

                pt_proj_L1 = project_3d_to_2d(X_new_3d, P)
                pt_proj_R1 = project_3d_to_2d(X_new_3d, Q)

                err_L0 = np.linalg.norm(pt_proj_L0 - pt_actual_L0)
                err_R0 = np.linalg.norm(pt_proj_R0 - pt_actual_R0)
                err_L1 = np.linalg.norm(pt_proj_L1 - pt_actual_L1)
                err_R1 = np.linalg.norm(pt_proj_R1 - pt_actual_R1)

                if err_L0 < threshold and err_R0 < threshold and \
                        err_L1 < threshold and err_R1 < threshold:
                    supporter_match = cv2.DMatch(_queryIdx=idx_L0, _trainIdx=idx_L1, _distance=0)
                    supporters.append(supporter_match)
    if not quiet:
        print(f"Total points tracked across all 4 images: {len(all_four_matches)}")
        print(f"Total SUPPORTERS for current [R|t]: {len(supporters)}")

    return supporters, all_four_matches

def plot_supporters(img_left_0, kp_left_0, img_left_1, kp_left_1, all_matches, supporters, title="Matches vs Supporters"):
    """
    Plots all matches in red, and supporters in green over them.
    """
    img_all = cv2.drawMatches(img_left_0, kp_left_0, img_left_1, kp_left_1, all_matches, None,
                              matchColor=(0, 0, 255),
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    img_final = cv2.drawMatches(img_left_0, kp_left_0, img_left_1, kp_left_1, supporters, img_all,
                                matchColor=(0, 255, 0),
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS | cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG)

    img_final_rgb = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(20, 10))
    plt.imshow(img_final_rgb)
    plt.title(f"{title} | Total Matches: {len(all_matches)} (Red) | Supporters: {len(supporters)} (Green)")
    plt.axis('off')
    plt.show()


##### 3_5 #####

def prepare_ransac_data(inliers_0, temporal_matches, inliers_1, points_3d_0,
                        left_kp_0, right_kp_0, left_kp_1, right_kp_1, P, Q, threshold):
    """
    Pre-processes matches and 3D points before the RANSAC loop.
    Filters out points that don't match the original left_0 and right_0 projections.
    Returns vectorized numpy arrays for fast RANSAC iterations.
    """
    dict_temporal = {m.queryIdx: m.trainIdx for m in temporal_matches}
    dict_inliers_1 = {m.queryIdx: m.trainIdx for m in inliers_1}

    X_3d_list, pt_L1_list, pt_R1_list, original_matches = [], [], [], []

    for i, match_0 in enumerate(inliers_0):
        idx_L0 = match_0.queryIdx
        idx_R0 = match_0.trainIdx

        if idx_L0 in dict_temporal:
            idx_L1 = dict_temporal[idx_L0]
            if idx_L1 in dict_inliers_1:
                idx_R1 = dict_inliers_1[idx_L1]

                X_3d = points_3d_0[i]
                pt_L0 = np.array(left_kp_0[idx_L0].pt)
                pt_R0 = np.array(right_kp_0[idx_R0].pt)
                pt_L1 = np.array(left_kp_1[idx_L1].pt)
                pt_R1 = np.array(right_kp_1[idx_R1].pt)

                X_homog = np.append(X_3d, 1.0)

                uvw_L0 = P @ X_homog
                proj_L0 = np.array([uvw_L0[0] / uvw_L0[2], uvw_L0[1] / uvw_L0[2]])
                err_L0 = np.linalg.norm(proj_L0 - pt_L0)

                uvw_R0 = Q @ X_homog
                proj_R0 = np.array([uvw_R0[0] / uvw_R0[2], uvw_R0[1] / uvw_R0[2]])
                err_R0 = np.linalg.norm(proj_R0 - pt_R0)

                if err_L0 < threshold and err_R0 < threshold:
                    X_3d_list.append(X_3d)
                    pt_L1_list.append(pt_L1)
                    pt_R1_list.append(pt_R1)
                    original_matches.append(cv2.DMatch(_queryIdx=idx_L0, _trainIdx=idx_L1, _distance=0))

    if len(X_3d_list) == 0:
        return None, None, None, None

    X_3d_arr = np.array(X_3d_list, dtype=np.float32)
    pt_L1_arr = np.array(pt_L1_list, dtype=np.float32)
    pt_R1_arr = np.array(pt_R1_list, dtype=np.float32)

    return X_3d_arr, pt_L1_arr, pt_R1_arr, original_matches


def calculate_supporters_vectorized(R, t, X_3d_arr, pt_L1_arr, pt_R1_arr, P, Q, threshold):
    """
    Projects a batch of 3D points into left_1 and right_1 using [R|t].
    Returns the boolean mask of valid supporters and their total count.
    """
    X_new_3d = (R @ X_3d_arr.T).T + t.flatten()
    X_new_homog = np.hstack((X_new_3d, np.ones((len(X_new_3d), 1))))

    proj_L1_uvw = P @ X_new_homog.T
    proj_L1 = (proj_L1_uvw[:2] / proj_L1_uvw[2]).T
    err_L1 = np.linalg.norm(proj_L1 - pt_L1_arr, axis=1)

    proj_R1_uvw = Q @ X_new_homog.T
    proj_R1 = (proj_R1_uvw[:2] / proj_R1_uvw[2]).T
    err_R1 = np.linalg.norm(proj_R1 - pt_R1_arr, axis=1)

    valid_mask = (err_L1 < threshold) & (err_R1 < threshold)

    return np.sum(valid_mask), valid_mask


def RANSAC(inliers_0, temporal_matches, inliers_1, points_3d_0,
           left_kp_0, right_kp_0, left_kp_1, right_kp_1,
           P, Q, K, obj_points_3d, img_points_2d,
           p=0.99, threshold=2.0):
    """
    Main RANSAC loop, utilizing helper functions for readability and speed.
    """
    X_3d_arr, pt_L1_arr, pt_R1_arr, original_matches = prepare_ransac_data(
        inliers_0, temporal_matches, inliers_1, points_3d_0,
        left_kp_0, right_kp_0, left_kp_1, right_kp_1, P, Q, threshold
    )

    if X_3d_arr is None:
        return None, None, []

    best_R, best_t, best_supporters = None, None, []
    max_supporters_count = 0
    total_points = len(obj_points_3d)
    s = 4

    N = 100000
    i = 0

    while i < N:
        random_indices = random.sample(range(total_points), s)
        sample_3d = obj_points_3d[random_indices]
        sample_2d = img_points_2d[random_indices]

        #  PnP
        R, t, _ = calculate_pnp(sample_3d, sample_2d, K, quiet=True)

        if R is None:
            i += 1
            continue

        current_count, valid_mask = calculate_supporters_vectorized(
            R, t, X_3d_arr, pt_L1_arr, pt_R1_arr, P, Q, threshold
        )

        if current_count > max_supporters_count:
            max_supporters_count = current_count
            best_R = R
            best_t = t

            best_supporters = [original_matches[idx] for idx, is_valid in enumerate(valid_mask) if is_valid]

            w = current_count / total_points
            if w > 0:
                w_s = w ** s
                if w_s < 1.0:
                    N_dynamic = math.log(1 - p) / math.log(1 - w_s)
                    N = min(N, int(N_dynamic) + 1)
        i += 1

    return best_R, best_t, best_supporters

def rodriguez_to_mat(rvec, tvec):
    """
    Helper function provided in the exercise instructions.
    Converts a rotation vector to a rotation matrix and appends the translation vector.
    """
    rot, _ = cv2.Rodrigues(rvec)
    return np.hstack((rot, tvec))


def calc_the_transformations_to_RANSAC_inliers(best_supporters, inliers_0, points_3d_0, kp_left_1, K):
    """
        Refines the extrinsic camera matrix [R|t] using ALL RANSAC inliers.
        Using multiple points reduces noise and yields a more accurate transformation.
        """
    # 1. Create a dictionary to easily map left_0 indices to their 3D points
    dict_3d = {}
    for i, match in enumerate(inliers_0):
        dict_3d[match.queryIdx] = points_3d_0[i]

    refined_3d = []
    refined_2d = []

    # 2. Extract the exact 3D and 2D coordinates for the validated inliers
    for match in best_supporters:
        idx_L0 = match.queryIdx  # Original index in left_0
        idx_L1 = match.trainIdx  # Matched index in left_1

        refined_3d.append(dict_3d[idx_L0])
        refined_2d.append(kp_left_1[idx_L1].pt)

    # 3. Convert lists to numpy arrays of type float32 for OpenCV compatibility
    refined_3d = np.array(refined_3d, dtype=np.float32)
    refined_2d = np.array(refined_2d, dtype=np.float32)

    # 4. Run PnP on the entire set of inliers
    success, rvec, tvec = cv2.solvePnP(refined_3d, refined_2d, K, distCoeffs=None)

    if not success:
        print("Refined PnP failed!")
        return None, None, None

    # 5. Build the refined 3x4 Extrinsic Transformation Matrix (T)
    T = rodriguez_to_mat(rvec, tvec)

    # Extract R
    R = T[:, :3]

    print(f"Successfully refined transformation using {len(refined_3d)} inliers.")

    return R, tvec, T

def plot_3D_points_cloud_to_RANSAC(points_3d_0, points_3d_1, T):
    """
        Transforms the first point cloud using T, and plots both clouds from a top-down view.
        Crops extreme points to present the clouds in a meaningful manner.
        """
    # 1. Convert points_3d_0 to homogeneous coordinates by adding a column of 1s
    num_points = points_3d_0.shape[0]
    ones = np.ones((num_points, 1))
    points_3d_0_homog = np.hstack((points_3d_0, ones))  # Shape becomes (N, 4)

    # 2. Apply transformation T to cloud 0
    # T is (3x4), points_3d_0_homog.T is (4xN). Result is (3xN). Transpose back to (N, 3)
    transformed_cloud_0 = (T @ points_3d_0_homog.T).T

    # 3. Filter points "at infinity" or behind the camera for a meaningful plot
    # We keep points where Z (depth) is between 0 and 60 meters,
    # and X (left/right) is within a reasonable road width (-30 to 30 meters).
    def crop_cloud(cloud):
        mask = (cloud[:, 2] > 0) & (cloud[:, 2] < 60) & (cloud[:, 0] > -30) & (cloud[:, 0] < 30)
        return cloud[mask]

    cloud_0_cropped = crop_cloud(transformed_cloud_0)
    cloud_1_cropped = crop_cloud(points_3d_1)

    # 4. Plotting
    plt.figure(figsize=(10, 8))

    # Plot transformed cloud 0 (e.g., Blue)
    plt.scatter(cloud_0_cropped[:, 0], cloud_0_cropped[:, 2],
                c='blue', s=5, alpha=0.6, label='Cloud 0 (Transformed)')

    # Plot cloud 1 (e.g., Red)
    plt.scatter(cloud_1_cropped[:, 0], cloud_1_cropped[:, 2],
                c='red', s=5, alpha=0.6, label='Cloud 1 (Target)')

    plt.title('3D Point Clouds Alignment (Top-Down View, X-Z Plane)')
    plt.xlabel('X (Right/Left) [meters]')
    plt.ylabel('Z (Forward Depth) [meters]')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # Important for correct physical proportions
    plt.show()


def plot_left0_left1_inliers_outliers(img_left_0, kp_left_0, img_left_1, kp_left_1, all_matches, inliers):
    """
    Plots the matches between left_0 and left_1.
    Outliers are colored red, and RANSAC inliers are colored green on top of them.
    """
    # 1. Identify outliers
    # We create a fast lookup set for the indices of our confirmed inliers
    inlier_query_indices = set(match.queryIdx for match in inliers)

    # Any match from 'all_matches' that is not in the inlier set is an outlier
    outliers = [m for m in all_matches if m.queryIdx not in inlier_query_indices]

    # 2. Draw outliers in red
    img_outliers = cv2.drawMatches(img_left_0, kp_left_0, img_left_1, kp_left_1, outliers, None,
                                   matchColor=(0, 0, 255),  # BGR Red
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 3. Draw inliers in green ON TOP of the outliers image
    img_final = cv2.drawMatches(img_left_0, kp_left_0, img_left_1, kp_left_1, inliers, img_outliers,
                                matchColor=(0, 255, 0),  # BGR Green
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS | cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG)

    # Convert BGR to RGB for correct matplotlib display
    img_final_rgb = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)

    # 4. Display the result
    plt.figure(figsize=(20, 10))
    plt.imshow(img_final_rgb)
    plt.title(f"RANSAC Results | Inliers: {len(inliers)} (Green) | Outliers: {len(outliers)} (Red)", fontsize=16)
    plt.axis('off')
    plt.show()

####### 3_6 #######
def track_full_sequence(sequence_dir, K, P, Q, log_filename="movement_log.csv", max_frames=None):
    """
    Iterates over all frames in the dataset sequence (or up to max_frames).
    """
    print("\nStarting Full Sequence Tracking...")
    start_time = time.time()

    # 1. Find the total number of frames in the sequence
    img_0_dir = os.path.join(sequence_dir, 'image_0')
    num_frames = len([name for name in os.listdir(img_0_dir) if name.endswith('.png')])

    # --- התיקון: הגבלת מספר הפריימים אם התבקשנו ---
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)

    print(f"Found {num_frames} frames to process.")

    all_T = []  # List to store all relative Extrinsic matrices

    # נפתח את קובץ הלוג לכתיבה
    with open(log_filename, 'w') as log_file:
        # כתיבת כותרות העמודות בקובץ ה-CSV
        log_file.write("Frame,Distance_Meters,Status\n")

        # 2. Initialization: Process the very first frame (Frame 0) outside the loop
        prev_left_img, prev_left_kp, prev_left_desc, prev_inliers, prev_points_3d, prev_right_kp = process_single_point_cloud(
            0)

        # 3. Main Tracking Loop (From frame 1 to the end)
        for i in tqdm(range(1, num_frames), desc="Tracking Progress", unit="frame"):
            # A. Process the current frame
            curr_left_img, curr_left_kp, curr_left_desc, curr_inliers, curr_points_3d, curr_right_kp = process_single_point_cloud(
                i)

            # B. Match features over time
            temporal_matches = left_matches(prev_left_desc, curr_left_desc)

            # C. Get data for PnP
            obj_points_3d, img_points_2d = get_pnp_data(prev_inliers, temporal_matches, prev_points_3d, curr_left_kp)

            # Default fallback transformation
            T_refined = np.hstack((np.eye(3), np.zeros((3, 1))))
            success_flag = False
            step_distance = 0.0
            frame_status = "Failed PnP/RANSAC"

            if len(obj_points_3d) >= 4:
                # D. Run RANSAC to find the best inliers
                best_R, best_t, best_supporters = RANSAC(
                    prev_inliers, temporal_matches, curr_inliers, prev_points_3d,
                    prev_left_kp, prev_right_kp, curr_left_kp, curr_right_kp,
                    P, Q, K, obj_points_3d, img_points_2d, p=0.99, threshold=2.0
                )

                # E. Refine the transformation using ALL inliers (ההחלקה חזרה!)
                if best_R is not None and len(best_supporters) >= 4:
                    R_refined, t_refined, T_calculated = calc_the_transformations_to_RANSAC_inliers(
                        best_supporters, prev_inliers, prev_points_3d, curr_left_kp, K
                    )

                    if T_calculated is not None:
                        t_vec = T_calculated[:, 3]
                        step_distance = np.linalg.norm(t_vec)

                        # מוודאים שהעידון לא הרס את הפיזיקה (בדיקת שפיות אחרונה)
                        if step_distance < 5.0:
                            T_refined = T_calculated
                            success_flag = True
                            frame_status = "Success"
                        else:
                            frame_status = "Rejected Refinement (Crazy Distance)"
            # F. Append the result and log it
            all_T.append(T_refined)

            # כתיבת השורה לקובץ הלוג
            log_file.write(f"{i},{step_distance:.6f},{frame_status}\n")

            if not success_flag and frame_status == "Failed PnP/RANSAC":
                tqdm.write(f"Frame {i}: Tracking failed. Using Identity matrix fallback.")

            # G. STATE UPDATE
            prev_left_img = curr_left_img
            prev_left_kp = curr_left_kp
            prev_left_desc = curr_left_desc
            prev_right_kp = curr_right_kp
            prev_inliers = curr_inliers
            prev_points_3d = curr_points_3d

    # 4. End tracking and calculate total time
    end_time = time.time()
    tracking_time = end_time - start_time

    print(f"\nTracking completed successfully!")
    print(f"Total time taken: {tracking_time:.2f} seconds.")
    print(f"Average time per frame: {tracking_time / num_frames:.2f} seconds.")
    print(f"Movement log saved to: {log_filename}")

    return all_T, tracking_time

def read_ground_truth(file_path):
    """
    Reads the ground-truth poses from 00.txt.
    Converts each line of 12 numbers into a 3x4 Extrinsic matrix.
    """
    ground_truth_poses = []

    # 1. Check if the file exists to avoid crashes
    if not os.path.exists(file_path):
        print(f"Error: Ground truth file not found at {file_path}")
        return ground_truth_poses

    # 2. Open the file and read line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Strip whitespace and split the line by spaces
            string_values = line.strip().split()

            # Ensure the line has exactly 12 numbers before processing
            if len(string_values) == 12:
                # 3. Convert strings to floats
                float_values = [float(val) for val in string_values]

                # 4. Create a numpy array and reshape it to 3 rows and 4 columns
                pose_matrix = np.array(float_values, dtype=np.float32).reshape(3, 4)

                ground_truth_poses.append(pose_matrix)

    print(f"Successfully loaded {len(ground_truth_poses)} ground-truth poses.")

    return ground_truth_poses

def calculate_global_poses(relative_transforms):
    """
    Takes the list of relative transformations [R|t] between consecutive frames,
    and chains them to calculate the global Extrinsic matrix for each camera
    relative to the very first camera (left_0).
    """
    # 1. Initialize the global poses list with the first camera
    # Camera 0 is the origin of our global coordinate system: [I | 0]
    T_0 = np.hstack((np.eye(3), np.zeros((3, 1))))
    global_poses = [T_0]

    # 2. Iterate through each relative transformation to accumulate the global pose
    for T_rel in relative_transforms:
        # Extract R and t from the current relative transformation
        R_rel = T_rel[:, :3]
        # Reshape to ensure t is a column vector (3, 1) for matrix operations
        t_rel = T_rel[:, 3].reshape(3, 1)

        # Extract R and t from the previous global pose
        prev_global_T = global_poses[-1]
        R_prev = prev_global_T[:, :3]
        t_prev = prev_global_T[:, 3].reshape(3, 1)

        # 3. Chain the transformations using the mathematical formula:
        # R_new = R_rel * R_prev
        # t_new = R_rel * t_prev + t_rel
        R_curr = R_rel @ R_prev
        t_curr = (R_rel @ t_prev) + t_rel

        # 4. Construct the new 3x4 global extrinsic matrix and store it
        curr_global_T = np.hstack((R_curr, t_curr))
        global_poses.append(curr_global_T)

    print(f"Calculated {len(global_poses)} global poses.")
    return global_poses

def plot_final_trajectories(estimated_global_poses, gt_poses):
    """
    Extracts the (X, Z) locations from both estimated and ground-truth global poses,
    and plots them on a 2D top-down map.
    """
    est_x = []
    est_z = []

    for T in estimated_global_poses:
        R = T[:, :3]
        t = T[:, 3]
        C = -np.dot(R.T, t)
        est_x.append(C[0])  #X
        est_z.append(C[2])  #Z

    gt_x = []
    gt_z = []

    for T in gt_poses:
        R = T[:, :3]
        t = T[:, 3]
        C = -np.dot(R.T, t)
        gt_x.append(C[0])  # X
        gt_z.append(C[2])  # Z

    plt.figure(figsize=(12, 10))

    plt.plot(gt_x, gt_z, c='black', label='Ground Truth', linewidth=2, linestyle='--')

    plt.plot(est_x, est_z, c='blue', label='Estimated Trajectory', linewidth=2)

    plt.scatter([0], [0], c='red', marker='*', s=200, label='Start (0,0)', zorder=5)

    plt.title('Vehicle Trajectory: Estimated vs Ground Truth (Top-Down View)', fontsize=16)
    plt.xlabel('X (Right/Left) [meters]', fontsize=12)
    plt.ylabel('Z (Forward Depth) [meters]', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.axis('equal')
    plt.show()



########### debug ##############
def plot_only_ground_truth(gt_poses):
    """
    Plots only the ground truth trajectory to verify its shape and scale.
    """
    gt_x = []
    gt_z = []

    # חילוץ המיקומים מה-GT בעזרת הנוסחה הנכונה למטריצות Extrinsic
    for T in gt_poses:
        R = T[:, :3]
        t = T[:, 3]
        # מציאת מיקום המצלמה בעולם
        C = -np.dot(R.T, t)

        gt_x.append(C[0])  # ציר ה-X
        gt_z.append(C[2])  # ציר ה-Z

    # יצירת הגרף
    plt.figure(figsize=(10, 8))

    plt.plot(gt_x, gt_z, c='black', label='Ground Truth Trajectory', linewidth=2)
    plt.scatter([gt_x[0]], [gt_z[0]], c='red', marker='*', s=200, label='Start', zorder=5)

    plt.title('Ground Truth Trajectory Only (Corrected)', fontsize=16)
    plt.xlabel('X (Right/Left) [meters]', fontsize=12)
    plt.ylabel('Z (Forward Depth) [meters]', fontsize=12)

    plt.legend(fontsize=12)
    plt.grid(True)

    # חובה לשמור על פרופורציות
    plt.axis('equal')
    plt.show()
############################3##3
if __name__ == '__main__':
    sequence_dir = r"C:\university\SHANA 5\semester B\67604-slam\VAN_SLAM\VAN_ex\dataset\dataset_2026\sequences\00"

    estimated_relative_transforms, total_time = track_full_sequence(sequence_dir, K, P, Q)

    gt_file_path = r"C:\university\SHANA 5\semester B\67604-slam\VAN_SLAM\VAN_ex\dataset\dataset_2026\poses\00.txt"
    ground_truth_poses = read_ground_truth(gt_file_path)

    estimated_global_poses = calculate_global_poses(estimated_relative_transforms)

    ground_truth_poses_cropped = ground_truth_poses[:len(estimated_global_poses)]

    plot_final_trajectories(estimated_global_poses, ground_truth_poses_cropped)



    # --- סעיף 3.1: יצירת ענני הנקודות ---
    # (הפונקציה שכבר כתבת שעושה את העבודה לזוג 0 ולזוג 1)
    # left_img_0, left_kp_0, left_desc_0, triangulate_points_0, inliers_0, right_kp_0, \
    #     left_img_1, left_kp_1, left_desc_1, triangulate_points_1, inliers_1, right_kp_1 = main_3_1()
    #
    # # --- סעיף 3.2: מציאת התאמות בזמן (מ-left0 ל-left1) ---
    # temporal_matches = left_matches(left_desc_0, left_desc_1)
    #
    # # --- סעיף 3.3: מציאת מיקום המצלמה (PnP) ---
    # print("\n--- Starting Exercise 3.3: PnP ---")
    #
    # obj_points_3d, img_points_2d = get_pnp_data(inliers_0, temporal_matches, triangulate_points_0, left_kp_1)
    #
    # if len(obj_points_3d) >= 4:
    #     R, t, Rt_matrix = calculate_pnp(obj_points_3d, img_points_2d, K)
    #     plot_cameras_top_down(R, t)
    #
    #     print("\n--- Starting Exercise 3.4: Finding Supporters ---")
    #
    #     # הרצת הפונקציה לבדיקת התומכים מתוך כל ההתאמות
    #     supporters, _ = matches_and_supports(inliers_0, temporal_matches, inliers_1, triangulate_points_0,
    #                                          left_kp_0, right_kp_0, left_kp_1, right_kp_1,
    #                                          P, Q, R, t, threshold=2.0)
    #
    #     # הדפסת התמונה המבוקשת עם הצבעים המבדילים
    #     plot_supporters(left_img_0, left_kp_0, left_img_1, left_kp_1, temporal_matches, supporters)
    #
    #     best_R, best_t, best_supporters = RANSAC(inliers_0, temporal_matches, inliers_1, triangulate_points_0,
    #                                              left_kp_0, right_kp_0, left_kp_1, right_kp_1,
    #                                              P, Q, K, obj_points_3d, img_points_2d,
    #                                              p=0.99, threshold=2.0)
    #
    #     R_refined, t_refined, T_refined = calc_the_transformations_to_RANSAC_inliers(
    #         best_supporters, inliers_0, triangulate_points_0, left_kp_1, K
    #     )
    #
    #     if T_refined is not None:
    #         plot_3D_points_cloud_to_RANSAC(triangulate_points_0, triangulate_points_1, T_refined)
    #
    #         plot_left0_left1_inliers_outliers(
    #             left_img_0, left_kp_0, left_img_1, left_kp_1,
    #             temporal_matches, best_supporters
    #         )
    # else:
    #     print("Not enough points found to run PnP! Need at least 4.")
