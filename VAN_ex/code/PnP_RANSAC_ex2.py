from geometric_outline_rejection_ex1 import *

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

    return left_img, left_kp, left_desc, inliers, triangulate_points

def main_3_1():
    print("doing 3_1")

    #for the pair (left_0, right_0)
    left_img_0, left_kp_0, left_desc_0, inliers_0, triangulate_points_0 = process_single_point_cloud(0)

    #for the pair (left_1, right_1)
    left_img_1, left_kp_1, left_desc_1, inliers_1, triangulate_points_1 = process_single_point_cloud(1)

    print(f"Generated 3D point cloud for pair 0 with {len(triangulate_points_0)} points.")
    print(f"Generated 3D point cloud for pair 1 with {len(triangulate_points_1)} points.")

    return left_img_0, left_kp_0, left_desc_0, triangulate_points_0, inliers_0, \
        left_img_1, left_kp_1, left_desc_1, triangulate_points_1, inliers_1


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

    left_img_0, left_kp_0, left_desc_0, triangulate_points_0, inliers_0, \
        left_img_1, left_kp_1, left_desc_1, triangulate_points_1, inliers_1 = main_3_1()

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

    dict_3d = {}
    for i, match in enumerate(inliers_0):
        idx_left_0 = match.queryIdx
        dict_3d[idx_left_0] = points_3d_0[i]

    for match in temporal_matches:
        idx_left_0 = match.queryIdx
        idx_left_1 = match.trainIdx

        if idx_left_0 in dict_3d:
            obj_points_3d.append(dict_3d[idx_left_0])
            img_points_2d.append(kp_left_1[idx_left_1].pt)

    obj_points_3d = np.array(obj_points_3d, dtype=np.float32)
    img_points_2d = np.array(img_points_2d, dtype=np.float32)
    print(f"Found {len(img_points_2d)} 3D points.")
    print(f"Found {len(obj_points_3d)} points matched across all 4 images.")
    return obj_points_3d, img_points_2d


def calculate_pnp(obj_points_3d, img_points_2d, K):
    """
    Applies PnP on 4 matched points to calculate the extrinsic matrix [R|t].
    """
    subset_3d = obj_points_3d[:4]
    subset_2d = img_points_2d[:4]

    # todo this is 4 points, need to change
    success, rvec, tvec = cv2.solvePnP(subset_3d, subset_2d, K, distCoeffs=None, flags=cv2.SOLVEPNP_AP3P)

    if not success:
        print("PnP failed to find a solution.")
        return None, None

    R, _ = cv2.Rodrigues(rvec)

    Rt_matrix = np.hstack((R, tvec))

    print("Extrinsic Camera Matrix [R|t] for left_1:")
    print(Rt_matrix)

    return R, tvec, Rt_matrix


def plot_cameras_top_down(R, tvec, baseline=-0.54):
    """
    Plots the relative positions of the 4 cameras from a top-down view (X-Z plane).
    """
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

    plt.title('Top-Down View of Cameras (X-Z Plane) [cite: 1]')
    plt.xlabel('X (Right/Left) [meters]')
    plt.ylabel('Z (Forward Depth) [meters]')

    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    # --- סעיף 3.1: יצירת ענני הנקודות ---
    # (הפונקציה שכבר כתבת שעושה את העבודה לזוג 0 ולזוג 1)
    left_img_0, left_kp_0, left_desc_0, points_3d_0, inliers_0, \
        left_img_1, left_kp_1, left_desc_1, points_3d_1, inliers_1 = main_3_1()

    # --- סעיף 3.2: מציאת התאמות בזמן (מ-left0 ל-left1) ---
    temporal_matches = left_matches(left_desc_0, left_desc_1)

    # --- סעיף 3.3: מציאת מיקום המצלמה (PnP) ---
    print("\n--- Starting Exercise 3.3: PnP ---")

    # 1. מוצאים את הנקודות שקיימות בכל 4 התמונות (החיתוך)
    obj_points_3d, img_points_2d = get_pnp_data(inliers_0, temporal_matches, points_3d_0, left_kp_1)

    # 2. מוודאים שיש לנו לפחות 4 נקודות כדי שהאלגוריתם יוכל לעבוד
    if len(obj_points_3d) >= 4:
        # מפעילים את ה-PnP!
        # שים לב שמטריצת K הגלובלית שכבר קראת בתחילת הקוד מתרגיל 2 מועברת לכאן
        R, t, Rt_matrix = calculate_pnp(obj_points_3d, img_points_2d, K)
    else:
        print("Not enough points found to run PnP! Need at least 4.")

    plot_cameras_top_down(R,t)