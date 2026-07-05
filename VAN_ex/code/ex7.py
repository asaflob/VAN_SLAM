from ex6 import *
import numpy as np
import heapq
from collections import defaultdict
import pickle
import math
#process_all_ba_windows
"""
def process_all_ba_windows(all_ba_results, plot_first_only=True):
    Iterates over all Bundle Adjustment optimizations to estimate the relative motion
    between every two consecutive keyframes and their relative covariance[cite: 1].

    Args:
        all_ba_results (list of dict): A list where each element contains the data
                                       from one BA window {'graph': graph, 'result': result,
                                       'c0_key': int, 'ck_key': int}.
        plot_first_only (bool): If True, only plots the 3D graph for the first window.

    Returns:
        list of dict: The constraints ready to be added to the Pose Graph.
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
"""
#q6_2
"""
def q6_2(pose_graph_constraints, initial_poses_dict, ground_truth_path):
    Builds and optimizes a Pose Graph based on the relative motions estimated in Section 6.1.

    Args:
        pose_graph_constraints (list of dict): Extracted relative poses and covariances.
            Each dict contains: 'c0_key', 'ck_key', 'relative_pose', and 'noise_model'.
        initial_poses_dict (dict): A mapping from GTSAM keys to global gtsam.Pose3 objects
            to serve as the initial guess.

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
"""


def dijkstra_covariance(graph, start_node):
    """
    Finds the shortest path in a Pose Graph based on covariance determinants.

    Args:
        graph (dict): Adjacency list representing the pose graph.
                      Format: {node_id: [(neighbor_id, covariance_matrix_6x6), ...]}
        start_node (int): The starting Keyframe ID (e.g., c_n).

    Returns:
        Sigma (dict): Minimum accumulated covariance matrices for each reached node.
        pi (dict): Predecessors dictionary to reconstruct the shortest paths.
    """
    # INIT-SINGLE-SOURCE
    Sigma = {}
    pi = {}

    Sigma[start_node] = np.zeros((6, 6))
    pi[start_node] = None

    pq = [(0.0, start_node)]

    visited = set()

    while pq:
        # EXTRACT-MIN(Q)
        current_det, u = heapq.heappop(pq)

        if u in visited:
            continue
        visited.add(u)

        for v, w_uv in graph.get(u, []):

            #  (Sigma[u] + w(u,v))
            proposed_cov = Sigma[u] + w_uv
            proposed_det = np.linalg.det(proposed_cov)

            # RELAX(u, v, w)
            if v not in Sigma or proposed_det < np.linalg.det(Sigma[v]):
                Sigma[v] = proposed_cov
                pi[v] = u

                heapq.heappush(pq, (proposed_det, v))

    return Sigma, pi


def build_covariance_graph(pose_graph_constraints):
    """
    Converts the pose graph constraints from Exercise 6 into a bidirectional adjacency list.

    Args:
        pose_graph_constraints (list of dict): The constraints list where each element
                                               has 'c0_key', 'ck_key', and 'noise_model'.

    Returns:
        dict: Adjacency list in the format {node_id: [(neighbor_id, covariance_matrix_6x6), ...]}
    """
    graph = defaultdict(list)

    for constraint in pose_graph_constraints:
        u = constraint['c0_key']
        v = constraint['ck_key']

        noise_model = constraint['noise_model']
        cov_matrix = noise_model.covariance()

        graph[u].append((v, cov_matrix))
        graph[v].append((u, cov_matrix))

    return dict(graph)


def q7_1(pose_graph_constraints, optimized_poses, chi2_threshold=10000.0, min_frame_gap=100, max_dist_meters=30.0):
    cov_graph = build_covariance_graph(pose_graph_constraints)

    keys = set()
    for constraint in pose_graph_constraints:
        keys.add(constraint['c0_key'])
        keys.add(constraint['ck_key'])

    sorted_keys = sorted(list(keys), key=lambda k: gtsam.Symbol(k).index())

    loop_closure_candidates = []

    for n_idx, c_n in enumerate(sorted_keys):
        if n_idx % 100 == 0:
            print(f"Searching loop closures for Keyframe index {n_idx}...")

        Sigma_dict, _ = dijkstra_covariance(cov_graph, c_n)

        for i_idx in range(n_idx):
            if (n_idx - i_idx) < min_frame_gap:
                continue

            c_i = sorted_keys[i_idx]

            if c_i not in Sigma_dict:
                continue

            Sigma_ni = Sigma_dict[c_i]

            pose_n = optimized_poses.atPose3(c_n)
            pose_i = optimized_poses.atPose3(c_i)

            delta_pose = pose_n.between(pose_i)
            delta_c_ni = gtsam.Pose3.Logmap(delta_pose)

            try:
                Sigma_ni_inv = np.linalg.inv(Sigma_ni)
                mahalanobis_dist2 = delta_c_ni.T @ Sigma_ni_inv @ delta_c_ni
            except np.linalg.LinAlgError:
                mahalanobis_dist2 = float('inf')

            euclidean_dist = np.linalg.norm(delta_pose.translation())

            if mahalanobis_dist2 < chi2_threshold or euclidean_dist < max_dist_meters:
                loop_closure_candidates.append({
                    'c_n_key': c_n,
                    'c_i_key': c_i,
                    'mahalanobis_dist2': mahalanobis_dist2
                })

    return loop_closure_candidates


def q7_2(db, c_i_idx, c_n_idx, K, P, Q, inlier_threshold=25):
    """
    Performs a consensus match between a candidate loop closure pair using custom RANSAC.
    """
    desc_i = db.features(c_i_idx)
    desc_n = db.features(c_n_idx)

    if desc_i is None or desc_n is None:
        return False, []

    temporal_matches = left_matches(desc_i, desc_n)

    links_i = db.all_frame_links(c_i_idx)
    links_n = db.all_frame_links(c_n_idx)

    pts_3d_i = []
    kp_left_i = []
    kp_right_i = []
    inliers_i = []

    for j, link in enumerate(links_i):
        kp_left_i.append(cv2.KeyPoint(x=link.x_left, y=link.y, size=1))
        kp_right_i.append(cv2.KeyPoint(x=link.x_right, y=link.y, size=1))
        inliers_i.append(cv2.DMatch(_queryIdx=j, _trainIdx=j, _distance=0))

        pt_l = np.array([[link.x_left], [link.y]], dtype=np.float32)
        pt_r = np.array([[link.x_right], [link.y]], dtype=np.float32)

        pt_4d_hom = cv2.triangulatePoints(P, Q, pt_l, pt_r)

        pt_3d = (pt_4d_hom[:3] / pt_4d_hom[3]).flatten()
        pts_3d_i.append(pt_3d)

    pts_3d_i = np.array(pts_3d_i, dtype=np.float32)
    kp_left_n = []
    kp_right_n = []
    inliers_n = []

    for j, link in enumerate(links_n):
        kp_left_n.append(cv2.KeyPoint(x=link.x_left, y=link.y, size=1))
        kp_right_n.append(cv2.KeyPoint(x=link.x_right, y=link.y, size=1))
        inliers_n.append(cv2.DMatch(_queryIdx=j, _trainIdx=j, _distance=0))

    obj_points_3d, img_points_2d = get_pnp_data(inliers_i, temporal_matches, pts_3d_i, kp_left_n)

    if len(obj_points_3d) < 4:
        return False, []

    best_R, best_t, best_supporters = RANSAC(
        inliers_0=inliers_i,
        temporal_matches=temporal_matches,
        inliers_1=inliers_n,
        points_3d_0=pts_3d_i,
        left_kp_0=kp_left_i,
        right_kp_0=kp_right_i,
        left_kp_1=kp_left_n,
        right_kp_1=kp_right_n,
        P=P, Q=Q, K=K,
        obj_points_3d=obj_points_3d,
        img_points_2d=img_points_2d,
        p=0.99, threshold=3.0
    )

    if best_supporters is not None and len(best_supporters) >= inlier_threshold:
        print(f"Loop Closure confirmed! {len(best_supporters)} inliers found.")
        return True, best_supporters

    return False, []


def q7_3(db,c_i_idx, c_n_idx, best_supporters, pts_3d_i, kp_left_n, K, T_initial_guess):
    """
    Performs a small BA to estimate relative pose and covariance between two matched frames.
    """
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    sym_i = gtsam.symbol('x', c_i_idx)
    sym_n = gtsam.symbol('x', c_n_idx)

    pose_i_identity = gtsam.Pose3()  # [I | 0]
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6] * 6))
    graph.add(gtsam.PriorFactorPose3(sym_i, pose_i_identity, prior_noise))
    initial_estimate.insert(sym_i, pose_i_identity)

    pose_n_guess = gtsam.Pose3(T_initial_guess)  # המרת מטריצת ה-4x4
    initial_estimate.insert(sym_n, pose_n_guess)

    pixel_noise = gtsam.noiseModel.Isotropic.Sigma(2, 2.0)
    cal3_s2 = gtsam.Cal3_S2(K[0, 0], K[1, 1], K[0, 1], K[0, 2], K[1, 2])

    links_i = db.all_frame_links(c_i_idx)
    links_n = db.all_frame_links(c_n_idx)

    for idx, match in enumerate(best_supporters):
        pt_key = gtsam.symbol('l', idx)

        pt_3d = pts_3d_i[match.queryIdx]
        initial_estimate.insert(pt_key, gtsam.Point3(pt_3d[0], pt_3d[1], pt_3d[2]))


        ###########
        pt_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)  # 10cm uncertainty
        graph.add(gtsam.PriorFactorPoint3(pt_key, gtsam.Point3(pt_3d[0], pt_3d[1], pt_3d[2]), pt_noise))
        ########
        link_i = links_i[match.queryIdx]
        link_n = links_n[match.trainIdx]

        pt_2d_i_arr = link_i.left_keypoint()
        pt_2d_n_arr = link_n.left_keypoint()

        gtsam_pt_2d_i = gtsam.Point2(pt_2d_i_arr[0], pt_2d_i_arr[1])
        gtsam_pt_2d_n = gtsam.Point2(pt_2d_n_arr[0], pt_2d_n_arr[1])

        graph.add(gtsam.GenericProjectionFactorCal3_S2(
            gtsam_pt_2d_i, pixel_noise, sym_i, pt_key, cal3_s2
        ))

        graph.add(gtsam.GenericProjectionFactorCal3_S2(
            gtsam_pt_2d_n, pixel_noise, sym_n, pt_key, cal3_s2
        ))

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
    result = optimizer.optimize()

    relative_pose = result.atPose3(sym_n)

    marginals = gtsam.Marginals(graph, result)
    relative_covariance = marginals.marginalCovariance(sym_n)

    return relative_pose, relative_covariance


def q7_4(original_graph, current_estimate, successful_loop_closures):
    """
        Updates the Pose Graph with new loop closure constraints and re-optimizes.

        Args:
            original_graph (gtsam.NonlinearFactorGraph): The graph generated in Ex 6.
            current_estimate (gtsam.Values): The optimized trajectory from Ex 6.
            successful_loop_closures (list of dict): A list of dictionaries, each containing:
                {'c_i_key': gtsam.symbol, 'c_n_key': gtsam.symbol,
                 'relative_pose': gtsam.Pose3, 'covariance': np.ndarray}

        Returns:
            gtsam.NonlinearFactorGraph: The updated graph.
            gtsam.Values: The new, drift-corrected optimized trajectory.
        """
    updated_graph = gtsam.NonlinearFactorGraph(original_graph)

    for lc in successful_loop_closures:
        key_i = lc['c_i_key']
        key_n = lc['c_n_key']
        rel_pose = lc['relative_pose']
        cov_matrix = lc['covariance']

        noise_model = gtsam.noiseModel.Gaussian.Covariance(cov_matrix)

        updated_graph.add(gtsam.BetweenFactorPose3(key_i, key_n, rel_pose, noise_model))

        print(f"Added Loop Closure constraint between key {gtsam.Symbol(key_i).index()} "
              f"and key {gtsam.Symbol(key_n).index()}")

    print("\nOptimizing updated Pose Graph with Loop Closures...")

    optimizer = gtsam.LevenbergMarquardtOptimizer(updated_graph, current_estimate)

    print(f"Error BEFORE loop closure optimization: {optimizer.error()}")

    updated_result = optimizer.optimize()

    print(f"Error AFTER loop closure optimization: {optimizer.error()}")

    return updated_graph, updated_result


def extract_pnp_data_from_db(db, c_i_idx, c_n_idx, P, Q):  # <--- הוספנו את P לחתימה
    """
    Extracts and reconstructs the data needed for PnP directly from TrackingDB.
    """
    links_i = db.all_frame_links(c_i_idx)
    links_n = db.all_frame_links(c_n_idx)

    pts_3d_i = []
    inliers_i = []

    for j, link in enumerate(links_i):
        pt_l = np.array([[link.x_left], [link.y]], dtype=np.float32)
        pt_r = np.array([[link.x_right], [link.y]], dtype=np.float32)

        pt_4d_hom = cv2.triangulatePoints(P, Q, pt_l, pt_r)
        pt_3d = (pt_4d_hom[:3] / pt_4d_hom[3]).flatten()

        pts_3d_i.append(pt_3d)
        inliers_i.append(cv2.DMatch(_queryIdx=j, _trainIdx=j, _distance=0))

    kp_left_n = []
    for link in links_n:
        kp = cv2.KeyPoint(x=link.x_left, y=link.y, size=1)
        kp_left_n.append(kp)

    pts_3d_i = np.array(pts_3d_i, dtype=np.float32)

    return pts_3d_i, kp_left_n, inliers_i


def process_loop_closures(db, candidates, K, P, Q):
    successful_loop_closures = []
    best_match_data = None

    print("\n[Ex 7.2 & 7.3] Validating Candidates and Estimating Relative Pose...")
    for cand in candidates:
        c_i_key = cand['c_i_key']
        c_n_key = cand['c_n_key']

        c_i_idx = gtsam.Symbol(c_i_key).index()
        c_n_idx = gtsam.Symbol(c_n_key).index()

        is_valid_loop, best_supporters = q7_2(db, c_i_idx, c_n_idx, K, P, Q)

        if is_valid_loop:
            print(f"--> Valid Loop Closure found between Frame {c_i_idx} and Frame {c_n_idx}!")

            if best_match_data is None:
                best_match_data = (c_i_idx, c_n_idx, best_supporters)

            pts_3d_i, kp_left_n, inliers_i = extract_pnp_data_from_db(db, c_i_idx, c_n_idx, P,Q)
            R_init, tvec_init, T_guess = calc_the_transformations_to_RANSAC_inliers(
                best_supporters, inliers_i, pts_3d_i, kp_left_n, K
            )

            if T_guess is not None:
                rel_pose, rel_cov = q7_3(
                    db, c_i_idx, c_n_idx, best_supporters, pts_3d_i, kp_left_n, K, T_guess
                )
                successful_loop_closures.append({
                    'c_i_key': c_i_key,
                    'c_n_key': c_n_key,
                    'relative_pose': rel_pose,
                    'covariance': rel_cov
                })

                if len(successful_loop_closures) == 15:#20
                    print("\nReached 20 successful loop closures. Stopping early to proceed to optimization!")
                    break

    return successful_loop_closures, best_match_data


############# 7.5 #########
def plot_single_consensus_match(db, sequence_dir, c_i_idx, c_n_idx, inlier_matches):
    """
    Plots a specific consensus match, coloring inliers in cyan and outliers in magenta.
    """
    img_i_path = os.path.join(sequence_dir, 'image_0', f'{c_i_idx:06d}.png')
    img_n_path = os.path.join(sequence_dir, 'image_0', f'{c_n_idx:06d}.png')
    img_i = cv2.imread(img_i_path, cv2.IMREAD_GRAYSCALE)
    img_n = cv2.imread(img_n_path, cv2.IMREAD_GRAYSCALE)

    links_i = db.all_frame_links(c_i_idx)
    links_n = db.all_frame_links(c_n_idx)
    desc_i = db.features(c_i_idx)
    desc_n = db.features(c_n_idx)

    kp_i = [cv2.KeyPoint(x=link.x_left, y=link.y, size=1) for link in links_i]
    kp_n = [cv2.KeyPoint(x=link.x_left, y=link.y, size=1) for link in links_n]

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = matcher.knnMatch(desc_i, desc_n, k=2)
    all_good_matches = [m[0] for m in knn_matches if m[0].distance < 0.75 * m[1].distance]

    inlier_pairs = {(m.queryIdx, m.trainIdx) for m in inlier_matches}

    outlier_matches = []
    actual_inliers = []
    for m in all_good_matches:
        if (m.queryIdx, m.trainIdx) in inlier_pairs:
            actual_inliers.append(m)
        else:
            outlier_matches.append(m)

    img_matches = cv2.drawMatches(img_i, kp_i, img_n, kp_n, outlier_matches, None,
                                  matchColor=(255, 0, 255), singlePointColor=(255, 0, 255), flags=2)
    img_matches = cv2.drawMatches(img_i, kp_i, img_n, kp_n, actual_inliers, img_matches,
                                  matchColor=(255, 255, 0), singlePointColor=(255, 255, 0), flags=2)

    plt.figure(figsize=(15, 5))
    plt.imshow(img_matches)
    plt.title(f'Consensus Match: Frame {c_i_idx} (Left) to Frame {c_n_idx} (Right)\nCyan: Inliers, Magenta: Outliers')
    plt.axis('off')
    plt.show()


def plot_trajectory_vs_ground_truth(poses_no_lc, poses_with_lc, ground_truth_path):
    """Plots estimated trajectories (with/without LC) vs Ground Truth."""
    gt_poses = load_kitti_poses(ground_truth_path)

    keys = sorted([k for k in poses_no_lc.keys()])

    x_no_lc = []
    z_no_lc = []
    x_with_lc = []
    z_with_lc = []
    gt_x = []
    gt_z = []

    for k in keys:
        frame_idx = gtsam.symbolIndex(k)

        pose_no = poses_no_lc.atPose3(k)
        x_no_lc.append(pose_no.x())
        z_no_lc.append(pose_no.z())

        pose_with = poses_with_lc.atPose3(k)
        x_with_lc.append(pose_with.x())
        z_with_lc.append(pose_with.z())

        T_gt = gt_poses[frame_idx]
        C_gt = -T_gt[:3, :3].T @ T_gt[:3, 3]
        gt_x.append(C_gt[0])
        gt_z.append(C_gt[2])

    plt.figure(figsize=(10, 8))
    plt.plot(gt_x, gt_z, label='Ground Truth', color='black', linestyle='--')
    plt.plot(x_no_lc, z_no_lc, label='Optimized (No LC)', color='red')
    plt.plot(x_with_lc, z_with_lc, label='Final (With LC)', color='green')
    plt.title('Trajectory Comparison (X-Z plane)')
    plt.xlabel('X [m]')
    plt.ylabel('Z [m]')

    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_absolute_location_error(poses_no_lc, poses_with_lc, ground_truth_path):
    """Plots the absolute location error."""
    gt_data = np.loadtxt(ground_truth_path)
    keys = sorted([k for k in poses_no_lc.keys()])

    error_no_lc = []
    error_with_lc = []
    indices = []

    for k in keys:
        idx = gtsam.Symbol(k).index()
        indices.append(idx)

        gt_t = np.array([gt_data[idx, 3], gt_data[idx, 7], gt_data[idx, 11]])

        t_no_lc = poses_no_lc.atPose3(k).translation()
        t_with_lc = poses_with_lc.atPose3(k).translation()

        error_no_lc.append(np.linalg.norm(t_no_lc - gt_t))
        error_with_lc.append(np.linalg.norm(t_with_lc - gt_t))

    plt.figure(figsize=(10, 5))
    plt.plot(indices, error_no_lc, label='Error (No LC)', color='red')
    plt.plot(indices, error_with_lc, label='Error (With LC)', color='green')
    plt.title('Absolute Location Error')
    plt.xlabel('Keyframe Index')
    plt.ylabel('Error [m]')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_location_uncertainty(graph_no_lc, values_no_lc, graph_with_lc, values_with_lc):
    """Plots location uncertainty size using the determinant of the translational covariance."""
    print("Calculating marginals for uncertainty plot (this might take a moment)...")
    marginals_no_lc = gtsam.Marginals(graph_no_lc, values_no_lc)
    marginals_with_lc = gtsam.Marginals(graph_with_lc, values_with_lc)

    keys = sorted([k for k in values_no_lc.keys()])
    indices = [gtsam.Symbol(k).index() for k in keys]

    uncert_no_lc = []
    uncert_with_lc = []

    for k in keys:
        cov_no_lc = marginals_no_lc.marginalCovariance(k)[3:, 3:]
        cov_with_lc = marginals_with_lc.marginalCovariance(k)[3:, 3:]

        uncert_no_lc.append(np.linalg.det(cov_no_lc))
        uncert_with_lc.append(np.linalg.det(cov_with_lc))

    plt.figure(figsize=(10, 5))
    plt.plot(indices, uncert_no_lc, label='Uncertainty Volume (No LC)', color='red')
    plt.plot(indices, uncert_with_lc, label='Uncertainty Volume (With LC)', color='green')
    plt.title('Location Uncertainty Size (Determinant of Translation Covariance)')
    plt.xlabel('Keyframe Index')
    plt.ylabel('Determinant Volume')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_pose_graph_evolution(stages_data):
    """
    Plots 4 stages of the pose graph.
    stages_data is a list of tuples: (values, title, color)
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    for i, (values, title, color) in enumerate(stages_data):
        if i >= 4: break

        keys = sorted([k for k in values.keys()])
        x = [values.atPose3(k).x() for k in keys]
        z = [values.atPose3(k).z() for k in keys]

        axs[i].plot(x, z, marker='o', markersize=2, linestyle='-', color=color)
        axs[i].set_title(title)
        axs[i].set_xlabel('X [m]')
        axs[i].set_ylabel('Z [m]')
        axs[i].grid(True)
        axs[i].axis('equal')

    plt.tight_layout()
    plt.show()


#########################


def q7_5(db, sequence_dir, ground_truth_path,
         graph_no_lc, values_no_lc,
         graph_with_lc, values_with_lc,
         successful_loop_closures,
         four_stages_data, best_match_data):
    """
    Executes all plotting and reporting requirements for Section 7.5.
    """
    print("\n" + "="*40)
    print("=== Section 7.5: Results & Evaluation ===")
    print("="*40)

    print(f"\n[1] Number of successful loop closures detected: {len(successful_loop_closures)}")

    if best_match_data:
        print("\n[2] Plotting a single successful consensus match...")
        c_i, c_n, inliers = best_match_data
        plot_single_consensus_match(db, sequence_dir, c_i, c_n, inliers)

    print("\n[3] Plotting 4 versions of the pose graph along the process.")
    plot_pose_graph_evolution(four_stages_data)

    print("\n[4] Plotting Trajectories vs Ground Truth...")
    plot_trajectory_vs_ground_truth(values_no_lc, values_with_lc, ground_truth_path)

    print("\n[5] Plotting Absolute Location Error...")
    plot_absolute_location_error(values_no_lc, values_with_lc, ground_truth_path)

    plot_location_uncertainty(graph_no_lc, values_no_lc, graph_with_lc, values_with_lc)

    print("\nDone with Section 7.5!")


if __name__ == '__main__':
    project_root = r"C:\university\SHANA 5\semester B\67604-slam\VAN_SLAM\VAN_ex"
    sequence_dir = os.path.join(project_root, 'dataset', 'dataset_2026', 'sequences', '00')
    poses_dir = os.path.join(project_root, 'dataset', 'dataset_2026', 'poses')
    ground_truth_path = os.path.join(poses_dir, '00.txt')

    db = TrackingDB()
    db_filename = 'my_tracking_data_ex4'
    poses_npy_filename = 'my_estimated_poses_ex4.npy'

    K, M1, M2 = read_cameras()
    P, Q = K @ M1, K @ M2

    db.load(db_filename)
    estimated_poses = np.load(poses_npy_filename, allow_pickle=True)
    estimated_poses = list(estimated_poses)
    print("Data and estimated poses loaded successfully from file.")

    cache_filename = 'ex6_cached_results.pkl'

    if os.path.exists(cache_filename):
        print("\nLoading cached Ex 6 results from file. Skipping computation!")
        with open(cache_filename, 'rb') as f:
            cached_data = pickle.load(f)

        pose_graph_constraints = cached_data['pose_graph_constraints']
        initial_values_gtsam = cached_data['initial_values_gtsam']
        original_graph = cached_data['original_graph']
        optimized_poses = cached_data['optimized_poses']
    else:
        print("\nComputing Ex 6 results (this might take a while)...")

        all_ba_results = generate_all_ba_results(db, estimated_poses, K, P, Q)
        pose_graph_constraints = process_all_ba_windows(all_ba_results, plot_first_only=False)
        initial_poses_dict = {}

        first_key = pose_graph_constraints[0]['c0_key']
        current_pose = gtsam.Pose3()
        initial_poses_dict[first_key] = current_pose

        for constraint in pose_graph_constraints:
            c0_key = constraint['c0_key']
            ck_key = constraint['ck_key']
            rel_pose = constraint['relative_pose']

            if c0_key in initial_poses_dict:
                next_pose = initial_poses_dict[c0_key].compose(rel_pose)
                initial_poses_dict[ck_key] = next_pose

        initial_values_gtsam = gtsam.Values()
        for key, pose in initial_poses_dict.items():
            initial_values_gtsam.insert(key, pose)

        original_graph, optimized_poses = q6_2(pose_graph_constraints, initial_poses_dict, ground_truth_path)

        print("Saving Ex 6 results to cache file...")
        with open(cache_filename, 'wb') as f:
            pickle.dump({
                'pose_graph_constraints': pose_graph_constraints,
                'initial_values_gtsam': initial_values_gtsam,
                'original_graph': original_graph,
                'optimized_poses': optimized_poses
            }, f)

    print("\n[Ex 7.1] Detecting Loop Closure Candidates...")
    candidates = q7_1(pose_graph_constraints, optimized_poses)
    print(f"Found {len(candidates)} potential candidates based on Mahalanobis distance.")

    successful_loop_closures, best_match_data = process_loop_closures(db, candidates, K, P, Q)

    print("\n[Ex 7.4] Updating Pose Graph with Loop Closures...")
    if successful_loop_closures:
        updated_graph, final_trajectory = q7_4(
            original_graph, optimized_poses, successful_loop_closures
        )
        print("Pose graph updated successfully!")

        temp_graph = gtsam.NonlinearFactorGraph(original_graph)
        first_lc = successful_loop_closures[0]
        noise_model = gtsam.noiseModel.Gaussian.Covariance(first_lc['covariance'])
        temp_graph.add(
            gtsam.BetweenFactorPose3(first_lc['c_i_key'], first_lc['c_n_key'], first_lc['relative_pose'], noise_model))
        temp_optimizer = gtsam.LevenbergMarquardtOptimizer(temp_graph, optimized_poses)
        first_lc_values = temp_optimizer.optimize()

        four_stages_data = [
            (initial_values_gtsam, "1. Initial Odometry", "blue"),
            (optimized_poses, "2. Optimized Graph (Ex 6)", "red"),
            (first_lc_values, "3. After First Loop Closure", "orange"),
            (final_trajectory, "4. Final Graph (All LCs)", "green")
        ]

        q7_5(db, sequence_dir, ground_truth_path,
             original_graph, optimized_poses,
             updated_graph, final_trajectory,
             successful_loop_closures,
             four_stages_data, best_match_data)
    else:
        print("No valid loop closures were confirmed by RANSAC. Pose graph remains unchanged.")

