import cv2
import random
import matplotlib.pyplot as plt
import os
import numpy as np
#

DATA_PATH = r'C:\university\SHANA 5\semester B\67604-slam\VAN_SLAM\VAN_ex\dataset\dataset_2026\sequences\00\\'

NUM_FEATURES_TO_SHOW = 20
MAX_FEATURES = 501
FEATURE = cv2.ORB_create(MAX_FEATURES)
MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def read_images(idx):
    """
    read images from the dataset
    :param idx:     index of the image
    :return:    img_left, img_right  images  (left and right)
    """
    img_name = '{:06d}.png'.format(idx)
    path1 = os.path.join(DATA_PATH, 'image_0', img_name)
    path2 = os.path.join(DATA_PATH, 'image_1', img_name)
    img1 = cv2.imread(path1, 0)
    img2 = cv2.imread(path2, 0)
    print(img1.shape)
    # img_left = cv2.imread(DATA_PATH+'image_0\\'+img_name, 0)
    # img_right = cv2.imread(DATA_PATH+'image_1\\'+img_name, 0)
    return img1, img2
#
#
def read_cameras():
    data_path = DATA_PATH
    with open(data_path + 'calib.txt') as f:
        l1 = f.readline().split()[1:]  # skip first token
        l2 = f.readline().split()[1:]  # skip first token
    l1 = [float(i) for i in l1]
    m1 = np.array(l1).reshape(3, 4)
    l2 = [float(i) for i in l2]
    m2 = np.array(l2).reshape(3, 4)
    k = m1[:, :3]
    m1 = np.linalg.inv(k) @ m1
    m2 = np.linalg.inv(k) @ m2
    return k, m1, m2


K, M1, M2 = read_cameras()
P, Q = K @ M1, K @ M2  # multiply by intrinsic camera matrix


def read_and_extract_matches(index=0):
    left_img, right_img = read_images(index)
    left_kp, left_desc = FEATURE.detectAndCompute(left_img, None)
    right_kp, desc_right = FEATURE.detectAndCompute(right_img, None)
    matches = MATCHER.match(left_desc, desc_right)
    return left_img, right_img, left_kp, right_kp, left_desc, desc_right,matches

####### 2.1 #######
def analyze_rectified_pattern(left_kp, right_kp, matches, threshold=2.0):
    """
    Analyzes the y-coordinate deviations of matches in a rectified stereo pair.

    :param left_kp:   Keypoints from the left image
    :param right_kp:  Keypoints from the right image
    :param matches:   Matches found by the matcher
    :param threshold: The pixel deviation threshold to consider a match "bad"
    :return:          A numpy array of all deviations
    """
    deviations = []

    #calc the deviation for each match
    for match in matches:
        y_left = left_kp[match.queryIdx].pt[1]
        y_right = right_kp[match.trainIdx].pt[1]

        deviation = abs(y_left - y_right)
        deviations.append(deviation)

    deviations = np.array(deviations)

    #histograma
    plt.figure(figsize=(10, 6))
    plt.hist(deviations, bins=50, range=(0, max(deviations)), color='tab:blue')
    plt.title('Deviation from Rectified Stereo Pattern')
    plt.xlabel('deviation from rectified stereo pattern')
    plt.ylabel('Number of matches')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    bad_matches_count = np.sum(deviations > threshold)
    total_matches = len(deviations)

    if total_matches > 0:
        percentage_bad = (bad_matches_count / total_matches) * 100
        print(f"Percentage of matches that deviate by more than {threshold} pixels: {percentage_bad:.2f}%")
    else:
        print("No matches found to analyze.")

    return deviations

def main_2_1():
    left_img, right_img, left_kp, right_kp, _,_,matches = read_and_extract_matches(0)
    deviations = analyze_rectified_pattern(left_kp, right_kp, matches)
    return deviations

######## 2.2 #######


def filter_rectified_matches(matches, kp_left, kp_right, y_threshold=2.0):
    """
    Filters matches based on the rectified stereo constraint (y-axis deviation).

    :param matches: List of original matches from BFMatcher
    :param kp_left: List of keypoints in the left image
    :param kp_right: List of keypoints in the right image
    :param y_threshold: Maximum allowed vertical pixel deviation
    :return: inliers (accepted), outliers (rejected)
    """
    inliers = []
    outliers = []

    for match in matches:
        # קבלת הקואורדינטות (x, y) של הנקודות שהותאמו
        pt_left = kp_left[match.queryIdx].pt
        pt_right = kp_right[match.trainIdx].pt

        # חישוב ההפרש בציר ה-y
        y_diff = abs(pt_left[1] - pt_right[1])

        if y_diff <= y_threshold:
            inliers.append(match)
        else:
            outliers.append(match)

    return inliers, outliers


def plot_rectified_matches(img_left, img_right, kp_left, kp_right, inliers, outliers,
                           title="Rectified Stereo Pattern Rejection"):
    """
    Plots the inliers and outliers on side-by-side images.
    """
    img_inliers = cv2.drawMatches(img_left, kp_left, img_right, kp_right, inliers, None,
                                  matchColor=(0, 255, 0), flags=2)

    img_outliers = cv2.drawMatches(img_left, kp_left, img_right, kp_right, outliers, None,
                                   matchColor=(255, 0, 0), flags=2)

    plt.figure(figsize=(20, 10))

    plt.subplot(2, 1, 1)
    plt.imshow(img_inliers)
    plt.title(f"{title} - Inliers ({len(inliers)} matches)")
    plt.axis('off')

    plt.subplot(2, 1, 2)
    plt.imshow(img_outliers)
    plt.title(f"{title} - Outliers ({len(outliers)} matches)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def main_2_2():
    left_img, right_img, left_kp, right_kp, _,_,matches = read_and_extract_matches(0)

    inliers, outliers = filter_rectified_matches(matches, left_kp, right_kp)

    plot_rectified_matches(left_img, right_img, left_kp, right_kp, inliers, outliers)

    return inliers, outliers


##### 2.3 ######


def linear_least_squares_triangulation(P, Q, kp_left, kp_right):
    """
    Linear least squares triangulation
    :param P: 2D point in image 1
    :param Q: Camera matrix of image 2
    :param kp_left: key points in image 1
    :param kp_right: key points in image 2
    :return:    3D point
    """
    A = np.zeros((4, 4))
    # p_left, p_right = kp_left[inliers[ind].queryIdx], kp_right[inliers[ind].trainIdx]
    p_x, p_y = kp_left
    q_x, q_y = kp_right
    A[0] = P[2] * p_x - P[0]
    A[1] = P[2] * p_y - P[1]
    A[2] = Q[2] * q_x - Q[0]
    A[3] = Q[2] * q_y - Q[1]
    _, _, V = np.linalg.svd(A)
    if V[-1, 3] == 0:
        return V[-1, :3] / (V[-1, 3] + 1e-20)
    return V[-1, :3] / V[-1, 3]


def triangulate_matched_points(P, Q, inliers, kp_left, kp_right):
    """
    Triangulate the matched points
    :param P: Camera matrix of image 1
    :param Q: Camera matrix of image 2
    :param inliers: list of inliers
    :param kp_left: Key points in image 1
    :param kp_right: Key points in image 2
    :return:   3D points
    """
    X = np.zeros((len(inliers), 3))
    for i in range(len(inliers)):
        p_left, p_right = kp_left[inliers[i].queryIdx], kp_right[inliers[i].trainIdx]
        X[i] = linear_least_squares_triangulation(P, Q, p_left.pt, p_right.pt)
    return X


def cv_triangulate_matched_points(inliers, kp_left, kp_right, P, Q):
    """
    Triangulate the matched points using OpenCV
    :param inliers:
    :return:
    """
    X = np.zeros((len(inliers), 3))
    for i in range(len(inliers)):
        p_left, p_right = kp_left[inliers[i].queryIdx], kp_right[inliers[i].trainIdx]
        X_4d = cv2.triangulatePoints(P, Q, p_left.pt, p_right.pt)
        X_4d /= (X_4d[3] + 1e-10)
        X[i] = X_4d[:-1].T
    return X


def find_median_distance(X, X_cv):
    """
    Find the median distance between the triangulated points
    :param X:
    :param X_cv:
    :return:
    """
    norm = np.linalg.norm(X - X_cv, axis=1)
    return np.median(norm)


def q3(inliers, kp_left, kp_right):
    """
    Triangulate the matched points and compare the results with OpenCV
    :param inliers:
    :param kp_left:
    :param kp_right:
    :return:
    """
    x = triangulate_matched_points(P, Q, inliers, kp_left, kp_right)
    x_cv = cv_triangulate_matched_points(inliers, kp_left, kp_right, P, Q)
    plot3d_points(x, title="Q2_3 Triangulated points using linear least squares method")
    plot3d_points(x_cv, title="Q2_3 Triangulated points using OpenCV")
    median_distance = find_median_distance(x, x_cv)
    print('Median distance between the triangulated points: ', median_distance)
    return x, x_cv


def plot3d_points(points_vector, title="Triangulated points using linear least squares method"):
    x1, y1, z1 = points_vector[:, 0], points_vector[:, 1], points_vector[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    s1 = ax.scatter3D(x1, y1, z1, color='orange', s=1)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    # ax.set_zlim(-5, 10)
    # plt.show()


def main_2_3(inliers, kp_left, kp_right):
    """
    Main execution function for Exercise 2.3
    Reads camera matrices, triangulates 3D points manually and via OpenCV,
    plots the results, and compares them.
    """
    print("\n--- Starting Exercise 2.3: Triangulation ---")

    # P, Q = K @ M1, K @ M2
    K, M1, M2 = read_cameras()
    P = K @ M1
    Q = K @ M2

    print("Calculating manual triangulation...")
    X_manual = triangulate_matched_points(P, Q, inliers, kp_left, kp_right)

    print("Calculating OpenCV triangulation...")
    X_cv = cv_triangulate_matched_points(inliers, kp_left, kp_right, P, Q)

    median_dist = find_median_distance(X_manual, X_cv)
    print(f"Median distance between manual and OpenCV 3D points: {median_dist:.10e}")

    print("Plotting results...")
    plot3d_points(X_manual, title="Manual Triangulation (Least Squares)")
    plot3d_points(X_cv, title="OpenCV Triangulation")

    plt.show()

    return X_manual, X_cv

###### 2.4 ######

def main_2_4():
    print("\n--- Starting Exercise 2.4: Running over multiple pairs ---")
    images_index = random.sample(range(0, 3000), 3)
    for i in images_index:
        print(f"\nProcessing image pair {i}...")

        left_img, right_img, left_kp, right_kp, _,_,matches = read_and_extract_matches(i)

        inliers, outliers = filter_rectified_matches(matches, left_kp, right_kp)
        plot_rectified_matches(
            left_img, right_img, left_kp, right_kp, inliers, outliers,
            title=f"Rectified Matches for image {i}"
        )

        x_3d = triangulate_matched_points(P, Q, inliers, left_kp, right_kp)
        plot3d_points(x_3d, title=f"3D Triangulated points for image {i}")
        plt.show()

if __name__ == '__main__':
    # #2.1
    # left_img, right_img, left_kp, right_kp, matches = read_and_extract_matches(0)
    #
    # # 2.2
    # inliers, outliers = filter_and_plot_rectified_matches(left_img, right_img, matches, left_kp, right_kp)
    #
    # # 2.3
    # points_3d_manual, points_3d_cv = main_2_3(inliers, left_kp, right_kp)
    main_2_4()