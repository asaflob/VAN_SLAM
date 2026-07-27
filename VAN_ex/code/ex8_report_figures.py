"""
ex8_report_figures.py
======================
Figure & statistic generation for the final project report.

This file DOES NOT modify any submitted exercise (ex1..ex7). It imports their
primitives and re-uses the cached run artifacts:
    - my_tracking_data_ex4.pkl   (TrackingDB)
    - my_estimated_poses_ex4.npy (PnP global poses, world-to-camera 3x4)
    - ex6_cached_results.pkl     (pose_graph_constraints, initial_values_gtsam,
                                  original_graph, optimized_poses)
so that most figures regenerate in seconds. The heavier recomputations
(per-window bundle error details, loop-closure pipeline) are cached to their
own pkl files (ex8_*.pkl).

Pose convention (verified empirically against the data, see report):
    The stored 3x4 PnP/GT matrices are WORLD-TO-CAMERA extrinsics [R|t]:
        X_cam = R X_world + t     =>   camera center in world C = -R^T t
    We convert every raw pose to a CAMERA-TO-WORLD 4x4 matrix M = [R^T | C]
    up front, so that all downstream error math uses one clean convention:
        - camera center            = M[:3, 3]
        - camera orientation (c2w) = M[:3, :3]
        - relative pose a->b       = inv(M_a) @ M_b   (b in a's frame)
    GTSAM Pose3 results (bundle / pose-graph) are already camera-to-world, so
    their .matrix() plugs straight into the same helpers.



"""
import matplotlib
matplotlib.use('Agg')

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2

from ex7 import *

# --------------------------------------------------------------------------- #
#  Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = r"C:\university\SHANA 5\semester B\67604-slam\VAN_SLAM\VAN_ex"
SEQUENCE_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'dataset_2026', 'sequences', '00')
POSES_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'dataset_2026', 'poses')
GT_PATH = os.path.join(POSES_DIR, '00.txt')
FIG_DIR = os.path.join(PROJECT_ROOT, 'report_figures')
os.makedirs(FIG_DIR, exist_ok=True)

DB_FILE = 'my_tracking_data_ex4'
EST_POSES_FILE = 'my_estimated_poses_ex4.npy'
EX6_CACHE = 'ex6_cached_results.pkl'


# --------------------------------------------------------------------------- #
#  Pose helpers  (everything works on camera-to-world 4x4 matrices)
# --------------------------------------------------------------------------- #
def raw_to_c2w(T):
    """Convert a world-to-camera [R|t] (3x4 or 4x4) to camera-to-world 4x4."""
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    M = np.eye(4)
    M[:3, :3] = R.T
    M[:3, 3] = -R.T @ t
    return M


def center(M):
    """Camera center in world coords from a camera-to-world 4x4."""
    return M[:3, 3]


def rot_angle_deg(R):
    """Magnitude of a rotation matrix in degrees (Rodrigues)."""
    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    return float(np.linalg.norm(rvec) * 180.0 / np.pi)


def relative(M_a, M_b):
    """Relative pose a->b (b expressed in a's frame), camera-to-world 4x4 in/out."""
    return np.linalg.inv(M_a) @ M_b


def abs_pose_error(M_est, M_gt):
    """(dx, dy, dz, loc_norm[m], angle[deg]) between two camera-to-world poses."""
    d = center(M_est) - center(M_gt)
    ang = rot_angle_deg(M_est[:3, :3] @ M_gt[:3, :3].T)
    return d[0], d[1], d[2], float(np.linalg.norm(d)), ang


def rel_pose_error(Ma_e, Mb_e, Ma_g, Mb_g):
    """Relative-pose error between est displacement a->b and gt displacement a->b.
    Returns (loc_norm[m], angle[deg])."""
    d_est = relative(Ma_e, Mb_e)
    d_gt = relative(Ma_g, Mb_g)
    err = np.linalg.inv(d_gt) @ d_est
    loc = float(np.linalg.norm(err[:3, 3]))
    ang = rot_angle_deg(err[:3, :3])
    return loc, ang


def gt_cumulative_distance(gt_c2w):
    """Cumulative traveled distance along the GT trajectory (per-frame prefix sum).
    cumdist[j] = sum_{i<j} |C_{i+1}-C_i|.  total_distance(a,b)=cumdist[b]-cumdist[a]."""
    centers = np.array([center(M) for M in gt_c2w])
    step = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(step)])


def load_cache(path):
    """Load a pickle cache, returning None if missing or corrupt (so it recomputes)."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"  cache {path} is corrupt ({e}); recomputing")
        return None


def dump_cache(obj, path):
    """Atomic pickle write: dump to a temp file then replace, so a crash mid-write
    never leaves a corrupt cache."""
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def savefig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {name}")
    return path


# --------------------------------------------------------------------------- #
#  Data loading
# --------------------------------------------------------------------------- #
def load_all():
    """Load DB, PnP poses, GT, and the cached ex6 pose-graph products.
    All trajectories are returned as camera-to-world 4x4 matrices / dicts."""
    print("Loading cached artifacts...")
    db = TrackingDB()
    db.load(DB_FILE)

    est_raw = list(np.load(EST_POSES_FILE, allow_pickle=True))
    est_c2w = [raw_to_c2w(T) for T in est_raw]

    gt_raw = load_kitti_poses(GT_PATH)
    gt_c2w = [raw_to_c2w(T) for T in gt_raw]

    with open(EX6_CACHE, 'rb') as f:
        ex6 = pickle.load(f)
    constraints = ex6['pose_graph_constraints']
    initial_values = ex6['initial_values_gtsam']   # chained bundle odometry (keyframes)
    original_graph = ex6['original_graph']
    optimized_poses = ex6['optimized_poses']        # pose graph WITHOUT loop closure

    # Keyframe frame-indices (sorted), derived from the constraints.
    kf_set = set()
    for c in constraints:
        kf_set.add(gtsam.Symbol(c['c0_key']).index())
        kf_set.add(gtsam.Symbol(c['ck_key']).index())
    keyframes = sorted(kf_set)

    print(f"  frames={len(est_c2w)}  gt={len(gt_c2w)}  keyframes={len(keyframes)}  "
          f"constraints={len(constraints)}")
    return dict(db=db, est=est_c2w, gt=gt_c2w, constraints=constraints,
                initial_values=initial_values, original_graph=original_graph,
                optimized_poses=optimized_poses, keyframes=keyframes)


def values_to_c2w_dict(values):
    """gtsam.Values (pose keys) -> {frame_index: camera-to-world 4x4}."""
    out = {}
    for k in values.keys():
        if gtsam.symbolChr(k) == ord('x'):
            out[gtsam.symbolIndex(k)] = values.atPose3(k).matrix()
    return out


# --------------------------------------------------------------------------- #
#  SECTION 3a  -  Tracking statistics (need only the DB)
# --------------------------------------------------------------------------- #
def stats_tracking(db):
    """Print the required scalar tracking statistics."""
    total_tracks = db.get_total_tracks()
    total_frames = db.get_total_frames()
    mean_len, max_len, min_len = db.get_mean_track_length()
    mean_links = db.get_mean_frame_links()
    lines = [
        "==== Tracking statistics ====",
        f"Total number of tracks : {total_tracks}",
        f"Number of frames       : {total_frames}",
        f"Mean track length      : {mean_len:.2f}  (max {max_len}, min {min_len})",
        f"Mean frame links       : {mean_links:.2f}",
    ]
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(FIG_DIR, 'tracking_statistics.txt'), 'w') as f:
        f.write(txt + "\n")
    return dict(total_tracks=total_tracks, total_frames=total_frames,
                mean_len=mean_len, max_len=max_len, min_len=min_len,
                mean_links=mean_links)


def fig_matches_per_frame(db):
    """Number of (stereo) links per frame = tracked features present in each frame."""
    frames = list(db.all_frames())
    counts = [len(db.tracks(f)) for f in frames]
    mean_c = float(np.mean(counts))
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(frames, counts, color='tab:blue', linewidth=0.8, label='matches (tracked links)')
    ax.axhline(mean_c, color='darkgreen', linewidth=1.5, label=f'Mean: {mean_c:.0f}')
    ax.set_title('Number of matches per frame')
    ax.set_xlabel('frame')
    ax.set_ylabel('# matches (links)')
    ax.set_xlim(0, frames[-1])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return savefig(fig, 'fig_matches_per_frame.png')


def fig_connectivity(db):
    frames = list(db.all_frames())
    outgoing = []
    for i in range(len(frames) - 1):
        a = set(db.tracks(frames[i]))
        b = set(db.tracks(frames[i + 1]))
        outgoing.append(len(a & b))
    mean_o = float(np.mean(outgoing))
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(frames[:-1], outgoing, color='tab:blue', linewidth=0.8, label='outgoing tracks')
    ax.axhline(mean_o, color='darkgreen', linewidth=1.5, label=f'Mean: {mean_o:.0f}')
    ax.set_title('Connectivity (outgoing tracks to next frame)')
    ax.set_xlabel('frame')
    ax.set_ylabel('outgoing tracks')
    ax.set_xlim(0, frames[-2])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return savefig(fig, 'fig_connectivity.png')


def fig_inlier_percentage(db):
    frames = list(db.all_frames())
    pct = []
    for i in range(len(frames) - 1):
        a = set(db.tracks(frames[i]))
        b = set(db.tracks(frames[i + 1]))
        inl = len(a & b)
        tot = db.frameId_to_lfeature[frames[i]].shape[0]
        pct.append(100.0 * inl / tot if tot else 0.0)
    mean_p = float(np.mean(pct))
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(frames[1:], pct, color='tab:orange', linewidth=0.8, label='inliers %')
    ax.axhline(mean_p, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_p:.1f}%')
    ax.set_title('Percentage of inliers per frame')
    ax.set_xlabel('frame')
    ax.set_ylabel('inliers (%)')
    ax.set_xlim(0, frames[-1])
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return savefig(fig, 'fig_inlier_percentage.png')


def fig_track_length_histogram(db):
    lengths = [len(v) for v in db.trackId_to_frames.values()]
    max_len = max(lengths)
    bins = np.arange(2, max_len + 2) - 0.5
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(lengths, bins=bins, log=True, color='tab:blue')
    ax.set_title('Track length histogram')
    ax.set_xlabel('track length (frames)')
    ax.set_ylabel('# tracks (log scale)')
    ax.set_xlim(left=1)
    ax.grid(axis='y', alpha=0.3)
    return savefig(fig, 'fig_track_length_histogram.png')


# --------------------------------------------------------------------------- #
#  SECTION 3b  -  Absolute PnP error (all frames)  &  Relative errors
# --------------------------------------------------------------------------- #
def fig_absolute_pnp_error(est, gt):
    n = len(est)
    xs, ys, zs, norms, angs = [], [], [], [], []
    for i in range(n):
        dx, dy, dz, nrm, ang = abs_pose_error(est[i], gt[i])
        xs.append(abs(dx)); ys.append(abs(dy)); zs.append(abs(dz))
        norms.append(nrm); angs.append(ang)
    frames = np.arange(n)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    ax1.plot(frames, xs, label='X error', color='tab:blue', linewidth=0.9)
    ax1.plot(frames, ys, label='Y error', color='tab:orange', linewidth=0.9)
    ax1.plot(frames, zs, label='Z error', color='tab:green', linewidth=0.9)
    ax1.plot(frames, norms, label='Total norm', color='tab:red', linewidth=1.3)
    ax1.set_title('Absolute PnP estimation error - location')
    ax1.set_ylabel('location error [m]')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(frames, angs, color='tab:purple', linewidth=0.9)
    ax2.set_title('Absolute PnP estimation error - angle')
    ax2.set_xlabel('frame')
    ax2.set_ylabel('angle error [deg]')
    ax2.set_xlim(0, n - 1)
    ax2.grid(True, alpha=0.3)
    return savefig(fig, 'fig_absolute_pnp_error.png')


def fig_relative_consecutive_keyframes(est, gt, constraints, keyframes):
    """Relative error between consecutive keyframes for PnP and Bundle."""
    # Bundle relative poses come straight from the BA constraints (c0 -> ck).
    pnp_loc, pnp_ang, ba_loc, ba_ang, x_idx = [], [], [], [], []
    for c in constraints:
        a = gtsam.Symbol(c['c0_key']).index()
        b = gtsam.Symbol(c['ck_key']).index()
        x_idx.append(b)
        # PnP relative (accumulated global poses between the two keyframes)
        l, an = rel_pose_error(est[a], est[b], gt[a], gt[b])
        pnp_loc.append(l); pnp_ang.append(an)
        # Bundle relative (c['relative_pose'] is c0->ck camera-to-world)
        d_ba = c['relative_pose'].matrix()
        d_gt = relative(gt[a], gt[b])
        err = np.linalg.inv(d_gt) @ d_ba
        ba_loc.append(float(np.linalg.norm(err[:3, 3])))
        ba_ang.append(rot_angle_deg(err[:3, :3]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    ax1.plot(x_idx, pnp_loc, label='PnP', color='tab:blue', linewidth=1.0)
    ax1.plot(x_idx, ba_loc, label='Bundle', color='tab:red', linewidth=1.0)
    ax1.set_title('Relative pose error between consecutive keyframes - location')
    ax1.set_ylabel('location error [m]')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(x_idx, pnp_ang, label='PnP', color='tab:blue', linewidth=1.0)
    ax2.plot(x_idx, ba_ang, label='Bundle', color='tab:red', linewidth=1.0)
    ax2.set_title('Relative pose error between consecutive keyframes - angle')
    ax2.set_xlabel('keyframe (frame index)')
    ax2.set_ylabel('angle error [deg]')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    return savefig(fig, 'fig_relative_consecutive_keyframes.png')


def _subsection_errors(traj_c2w, gt_c2w, cumdist, lengths, frame_of):
    """Generic KITTI-style per-sub-section relative error.
    traj_c2w : list/dict of camera-to-world matrices sampled at frames `frame_of`
    frame_of : sorted list of frame indices that traj_c2w corresponds to
    Returns per-length dict: {L: (start_frames, loc_pct, ang_per_m, avg_loc, avg_ang)}."""
    out = {}
    # map frame index -> position in the sampled trajectory
    idx_of = {f: p for p, f in enumerate(frame_of)}
    frames_arr = np.array(frame_of)
    for L in lengths:
        s_frames, loc_pct, ang_pm = [], [], []
        for p, a in enumerate(frame_of):
            b_target = a + L
            if b_target > frame_of[-1]:
                break
            # closest sampled frame to a+L
            q = int(np.argmin(np.abs(frames_arr - b_target)))
            b = frame_of[q]
            if b <= a:
                continue
            dist = cumdist[b] - cumdist[a]
            if dist <= 1e-6:
                continue
            loc, ang = rel_pose_error(traj_c2w[idx_of[a]], traj_c2w[idx_of[b]],
                                      gt_c2w[a], gt_c2w[b])
            s_frames.append(a)
            loc_pct.append(100.0 * loc / dist)   # % (m/m *100)
            ang_pm.append(ang / dist)             # deg/m
        out[L] = (np.array(s_frames), np.array(loc_pct), np.array(ang_pm),
                  float(np.mean(loc_pct)) if loc_pct else float('nan'),
                  float(np.mean(ang_pm)) if ang_pm else float('nan'))
    return out


def fig_relative_subsections(traj_c2w, gt, frame_of, cumdist, tag, lengths=(100, 400, 800)):
    """Relative error over sub-sections of given lengths. `tag` in {'PnP','Bundle'}."""
    res = _subsection_errors(traj_c2w, gt, cumdist, list(lengths), frame_of)
    colors = {lengths[0]: 'tab:blue', lengths[1]: 'tab:red', lengths[2]: 'tab:green'}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    summary = []
    for L in lengths:
        sf, loc_pct, ang_pm, avg_loc, avg_ang = res[L]
        ax1.plot(sf, loc_pct, color=colors[L], linewidth=1.0,
                 label=f'L={L}  (avg {avg_loc:.2f}%)')
        ax2.plot(sf, ang_pm, color=colors[L], linewidth=1.0,
                 label=f'L={L}  (avg {avg_ang:.4f} deg/m)')
        summary.append(f"  {tag} L={L}: avg loc={avg_loc:.3f}% , avg ang={avg_ang:.5f} deg/m")
    ax1.set_title(f'Relative {tag} estimation error over sub-sections - location')
    ax1.set_ylabel('location error [%] (m/m)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax2.set_title(f'Relative {tag} estimation error over sub-sections - angle')
    ax2.set_xlabel('sub-section start frame')
    ax2.set_ylabel('angle error [deg/m]')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    print("\n".join(summary))
    return savefig(fig, f'fig_relative_subsections_{tag.lower()}.png'), res


# --------------------------------------------------------------------------- #
#  SECTION 3c  -  Bundle-window details (per-window BA rerun, cached)
# --------------------------------------------------------------------------- #
def generate_bundle_details(db, poses_raw, K, P, Q, cache='ex8_ba_details.pkl'):
    """Re-run BA on every keyframe window, recording optimization error metrics
    and projection-error-vs-distance data. Cached to `cache`."""
    cached = load_cache(cache)
    if cached is not None:
        print(f"  loaded cached bundle details from {cache}")
        return cached

    print("  running per-window bundle adjustment (one-time, please wait)...")
    from collections import defaultdict
    current = 0
    kf_start = []
    mean_factor_init, mean_factor_opt = [], []
    med_proj_init, med_proj_opt = [], []
    dist_err_opt = defaultdict(list)   # distance-from-ref -> [pixel residual norms]

    n = len(poses_raw)
    while current < n - 5:
        end = get_keyframe(poses_raw, current)
        window = list(range(current, end + 1))
        graph, initial, calib = build_bundle_graph(db, poses_raw, window, K, P, Q)
        if graph.size() == 0:
            current = end
            continue
        result = gtsam.LevenbergMarquardtOptimizer(graph, initial).optimize()
        nf = graph.size()
        kf_start.append(current)
        mean_factor_init.append(graph.error(initial) / nf)
        mean_factor_opt.append(graph.error(result) / nf)

        pin, popt = [], []
        for i in range(nf):
            fac = graph.at(i)
            if type(fac) is gtsam.GenericStereoFactor3D:
                r_i = fac.unwhitenedError(initial)
                r_o = fac.unwhitenedError(result)
                pin.append(float(np.linalg.norm(r_i)))
                popt.append(float(np.linalg.norm(r_o)))
                fr = gtsam.symbolIndex(fac.keys()[0])
                dist_err_opt[fr - current].append(float(np.linalg.norm(r_o)))
        med_proj_init.append(float(np.median(pin)) if pin else 0.0)
        med_proj_opt.append(float(np.median(popt)) if popt else 0.0)
        if len(kf_start) % 50 == 0:
            print(f"    processed {len(kf_start)} windows (frame {current})")
        current = end

    # collapse distance->errors into median curve
    dists = sorted(dist_err_opt.keys())
    med_curve = [float(np.median(dist_err_opt[d])) for d in dists]
    details = dict(kf_start=kf_start,
                   mean_factor_init=mean_factor_init, mean_factor_opt=mean_factor_opt,
                   med_proj_init=med_proj_init, med_proj_opt=med_proj_opt,
                   proj_dist=dists, proj_dist_median=med_curve)
    dump_cache(details, cache)
    print(f"  cached bundle details to {cache}")
    return details


def fig_bundle_mean_factor_error(details):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(details['kf_start'], details['mean_factor_init'], color='tab:orange',
            linewidth=1.0, label='Initial error')
    ax.plot(details['kf_start'], details['mean_factor_opt'], color='tab:blue',
            linewidth=1.0, label='Optimized error')
    ax.set_title('Mean factor error per bundle window (before / after optimization)')
    ax.set_xlabel('bundle starting at frame idx')
    ax.set_ylabel('mean factor error (total / #factors)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    return savefig(fig, 'fig_bundle_mean_factor_error.png')


def fig_bundle_median_projection_error(details):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(details['kf_start'], details['med_proj_init'], color='tab:orange',
            linewidth=1.0, label='Initial (before opt)')
    ax.plot(details['kf_start'], details['med_proj_opt'], color='tab:blue',
            linewidth=1.0, label='Optimized (after opt)')
    ax.set_title('Median projection error per bundle window (before / after optimization)')
    ax.set_xlabel('bundle starting at frame idx')
    ax.set_ylabel('median projection error [px]')
    ax.grid(True, alpha=0.3)
    ax.legend()
    return savefig(fig, 'fig_bundle_median_projection_error.png')


def fig_bundle_projection_vs_distance(details):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(details['proj_dist'], details['proj_dist_median'], color='tab:red',
            marker='o', markersize=3, linewidth=1.0)
    ax.set_title('Bundle: median projection error vs distance from reference frame')
    ax.set_xlabel('distance from reference (frames)')
    ax.set_ylabel('median projection error [px]')
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)
    return savefig(fig, 'fig_bundle_projection_vs_distance.png')


# --------------------------------------------------------------------------- #
#  SECTION 3d  -  PnP projection error vs distance (subset of tracks, median)
# --------------------------------------------------------------------------- #
def fig_pnp_projection_vs_distance(db, gt, P, Q, min_len=10, max_tracks=500, seed=42):
    """Median reprojection error of track links vs frames from the triangulation
    (reference) frame, using GT poses. Averaged over a random subset of long tracks."""
    from collections import defaultdict
    rng = np.random.default_rng(seed)
    long_tracks = [t for t, fr in db.trackId_to_frames.items() if len(fr) >= min_len]
    if len(long_tracks) > max_tracks:
        long_tracks = list(rng.choice(long_tracks, size=max_tracks, replace=False))
    print(f"  PnP proj-vs-distance over {len(long_tracks)} tracks (len>={min_len})")

    gt_raw = load_kitti_poses(GT_PATH)  # world-to-camera 4x4 (P,Q act on these)
    dist_err_l = defaultdict(list)
    dist_err_r = defaultdict(list)

    for t in long_tracks:
        frames = db.frames(t)
        ref = frames[-1]                      # triangulate at last frame of the track
        link_ref = db.link(ref, t)
        pl = np.array([[link_ref.x_left], [link_ref.y]], float)
        pr = np.array([[link_ref.x_right], [link_ref.y]], float)
        X_local = cv2.triangulatePoints(P, Q, pl, pr)
        X_local = X_local[:3] / X_local[3]
        X_world = np.linalg.inv(gt_raw[ref]) @ np.vstack([X_local, [[1.0]]])
        for f in frames:
            X_cam = gt_raw[f] @ X_world
            projL = P @ X_cam
            projR = Q @ X_cam
            lx, ly = projL[0, 0] / projL[2, 0], projL[1, 0] / projL[2, 0]
            rx, ry = projR[0, 0] / projR[2, 0], projR[1, 0] / projR[2, 0]
            link = db.link(f, t)
            d = ref - f
            dist_err_l[d].append(np.hypot(lx - link.x_left, ly - link.y))
            dist_err_r[d].append(np.hypot(rx - link.x_right, ry - link.y))

    dists = sorted(dist_err_l.keys())
    med_l = [np.median(dist_err_l[d]) for d in dists]
    med_r = [np.median(dist_err_r[d]) for d in dists]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(dists, med_l, color='tab:blue', label='Left', linewidth=1.2)
    ax.plot(dists, med_r, color='tab:orange', label='Right', linewidth=1.2)
    ax.set_title(f'PnP: median projection error vs distance from reference '
                 f'({len(long_tracks)} tracks, len>={min_len})')
    ax.set_xlabel('distance from reference (frames)')
    ax.set_ylabel('median projection error [px]')
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return savefig(fig, 'fig_pnp_projection_vs_distance.png')


# --------------------------------------------------------------------------- #
#  SECTION 3e  -  Loop-closure pipeline (cached) + downstream figures
# --------------------------------------------------------------------------- #
def run_loop_closure(db, constraints, optimized_poses, original_graph, K, P, Q,
                     cache='ex8_lc_results.pkl'):
    """Run q7 loop-closure detection/validation/optimization, capturing per-LC
    match statistics. Cached to `cache`."""
    cached = load_cache(cache)
    if cached is not None:
        print(f"  loaded cached loop-closure results from {cache}")
        return cached

    print("  detecting loop-closure candidates (q7_1)...")
    candidates = q7_1(constraints, optimized_poses)
    print(f"  {len(candidates)} candidates")

    print("  validating + estimating relative poses (q7_2/q7_3)...")
    successful, best_match = process_loop_closures(db, candidates, K, P, Q)
    print(f"  {len(successful)} successful loop closures")

    # per-LC match statistics
    lc_stats = []
    for lc in successful:
        ci = gtsam.Symbol(lc['c_i_key']).index()
        cn = gtsam.Symbol(lc['c_n_key']).index()
        n_temporal = len(left_matches(db.features(ci), db.features(cn)))
        _, supporters = q7_2(db, ci, cn, K, P, Q)
        n_sup = len(supporters)
        lc_stats.append(dict(c_i=ci, c_n=cn, n_matches=n_temporal,
                             n_inliers=n_sup,
                             inlier_pct=100.0 * n_sup / n_temporal if n_temporal else 0.0))

    print("  re-optimising pose graph with loop closures (q7_4)...")
    updated_graph, final_traj = q7_4(original_graph, optimized_poses, successful)

    # best_match holds cv2.DMatch objects (not picklable) -> keep only the indices
    best_match_idx = (best_match[0], best_match[1]) if best_match else None
    result = dict(lc_stats=lc_stats,
                  final_traj=final_traj,
                  updated_graph=updated_graph,
                  best_match_idx=best_match_idx,
                  n_successful=len(successful))
    dump_cache(result, cache)
    print(f"  cached loop-closure results to {cache}")
    return result


def fig_trajectory_overlay(est, initial_values, optimized_poses, final_traj, gt, keyframes):
    """Combined bird's-eye trajectory: PnP, Bundle, Pose-Graph(noLC), +LC, GT."""
    def xz(M):
        c = center(M)
        return c[0], c[2]

    # PnP: all frames
    pnp = np.array([xz(M) for M in est])
    # GT: all frames
    gtxz = np.array([xz(M) for M in gt])
    # Bundle: chained BA odometry (keyframes)
    bun = values_to_c2w_dict(initial_values)
    bunp = np.array([[bun[k][0, 3], bun[k][2, 3]] for k in sorted(bun)])
    # Pose graph without LC
    pg = values_to_c2w_dict(optimized_poses)
    pgp = np.array([[pg[k][0, 3], pg[k][2, 3]] for k in sorted(pg)])
    # Pose graph with LC
    lc = values_to_c2w_dict(final_traj)
    lcp = np.array([[lc[k][0, 3], lc[k][2, 3]] for k in sorted(lc)])

    fig, ax = plt.subplots(figsize=(11, 10))
    ax.plot(gtxz[:, 0], gtxz[:, 1], color='green', linewidth=2.5, label='Ground Truth')
    ax.plot(pnp[:, 0], pnp[:, 1], color='gray', linewidth=1.0, alpha=0.8, label='PnP')
    ax.plot(pgp[:, 0], pgp[:, 1], color='tab:blue', linewidth=1.4, label='Pose Graph (no LC)')
    # Bundle odometry ~coincides with Pose-Graph-no-LC -> draw dashed on top so it shows
    ax.plot(bunp[:, 0], bunp[:, 1], color='tab:orange', linewidth=1.1, linestyle='--',
            label='Bundle (≈ Pose Graph no LC)')
    ax.plot(lcp[:, 0], lcp[:, 1], color='tab:red', linewidth=1.6, label='Pose Graph + Loop Closure')
    ax.scatter([0], [0], c='black', marker='*', s=180, zorder=5, label='Start')
    ax.set_title('Trajectory (top-down): PnP vs Bundle vs Pose Graph vs Ground Truth')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Z [m]')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    return savefig(fig, 'fig_trajectory_overlay.png')


def fig_absolute_posegraph_error(values, gt, tag, fname):
    """Absolute error (x/y/z/norm + angle) for a pose-graph Values vs GT, at keyframes."""
    d = values_to_c2w_dict(values)
    frames = sorted(d)
    xs, ys, zs, norms, angs = [], [], [], [], []
    for f in frames:
        dx, dy, dz, nrm, ang = abs_pose_error(d[f], gt[f])
        xs.append(abs(dx)); ys.append(abs(dy)); zs.append(abs(dz))
        norms.append(nrm); angs.append(ang)
    prefix = 'Bundle' if tag == 'Bundle' else f'Pose Graph ({tag})'
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    ax1.plot(frames, xs, label='X error', color='tab:blue', linewidth=1.0)
    ax1.plot(frames, ys, label='Y error', color='tab:orange', linewidth=1.0)
    ax1.plot(frames, zs, label='Z error', color='tab:green', linewidth=1.0)
    ax1.plot(frames, norms, label='Total norm', color='tab:red', linewidth=1.4)
    ax1.set_title(f'Absolute {prefix} estimation error - location')
    ax1.set_ylabel('location error [m]')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax2.plot(frames, angs, color='tab:purple', linewidth=1.0)
    ax2.set_title(f'Absolute {prefix} estimation error - angle')
    ax2.set_xlabel('keyframe (frame index)')
    ax2.set_ylabel('angle error [deg]')
    ax2.grid(True, alpha=0.3)
    return savefig(fig, fname)


def fig_loop_closure_match_stats(lc_stats):
    if not lc_stats:
        print("  no successful loop closures -> skipping match-stats figure")
        return None
    idx = [s['c_n'] for s in lc_stats]
    n_matches = [s['n_matches'] for s in lc_stats]
    inlier_pct = [s['inlier_pct'] for s in lc_stats]
    order = np.argsort(idx)
    idx = np.array(idx)[order]
    n_matches = np.array(n_matches)[order]
    inlier_pct = np.array(inlier_pct)[order]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    ax1.bar(range(len(idx)), n_matches, color='tab:blue')
    ax1.set_title('Number of matches per successful loop-closure frame')
    ax1.set_ylabel('# matches')
    ax1.set_xticks(range(len(idx)))
    ax1.set_xticklabels(idx, rotation=90, fontsize=7)
    ax1.grid(axis='y', alpha=0.3)
    ax2.bar(range(len(idx)), inlier_pct, color='tab:orange')
    ax2.set_title('Inlier percentage per successful loop-closure frame')
    ax2.set_xlabel('loop-closure frame (c_n index)')
    ax2.set_ylabel('inliers [%]')
    ax2.set_xticks(range(len(idx)))
    ax2.set_xticklabels(idx, rotation=90, fontsize=7)
    ax2.grid(axis='y', alpha=0.3)
    return savefig(fig, 'fig_loop_closure_match_stats.png')


def fig_uncertainty_vs_keyframe(graph_no, values_no, graph_lc, values_lc, lc_frames):
    """Location & angle uncertainty (determinant of covariance sub-block) vs keyframe,
    for pose graph with and without loop closure, on a log scale."""
    print("  computing marginals (no LC)...")
    m_no = gtsam.Marginals(graph_no, values_no)
    print("  computing marginals (with LC)...")
    m_lc = gtsam.Marginals(graph_lc, values_lc)

    keys = sorted([k for k in values_no.keys() if gtsam.symbolChr(k) == ord('x')],
                  key=lambda k: gtsam.symbolIndex(k))
    idx = [gtsam.symbolIndex(k) for k in keys]
    loc_no, loc_lc, ang_no, ang_lc = [], [], [], []
    for k in keys:
        c_no = m_no.marginalCovariance(k)
        c_lc = m_lc.marginalCovariance(k)
        ang_no.append(np.linalg.det(c_no[:3, :3]))
        loc_no.append(np.linalg.det(c_no[3:, 3:]))
        ang_lc.append(np.linalg.det(c_lc[:3, :3]))
        loc_lc.append(np.linalg.det(c_lc[3:, 3:]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    ax1.plot(idx, loc_no, color='tab:red', label='no LC', linewidth=1.2)
    ax1.plot(idx, loc_lc, color='tab:green', label='with LC', linewidth=1.2)
    for lf in lc_frames:
        ax1.axvline(lf, color='blue', alpha=0.15)
    ax1.set_yscale('log')
    ax1.set_ylim(bottom=1e-12)  # crop the anchored first-keyframe spike
    ax1.set_title('Location uncertainty vs keyframe (det of translation covariance)')
    ax1.set_ylabel('location uncertainty (log)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax2.plot(idx, ang_no, color='tab:red', label='no LC', linewidth=1.2)
    ax2.plot(idx, ang_lc, color='tab:green', label='with LC', linewidth=1.2)
    for lf in lc_frames:
        ax2.axvline(lf, color='blue', alpha=0.15)
    ax2.set_yscale('log')
    ax2.set_ylim(bottom=1e-20)  # crop the anchored first-keyframe spike
    ax2.set_title('Angle uncertainty vs keyframe (det of rotation covariance)')
    ax2.set_xlabel('keyframe (frame index)')
    ax2.set_ylabel('angle uncertainty (log)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    return savefig(fig, 'fig_uncertainty_vs_keyframe.png')


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    K, M1, M2 = read_cameras()
    P, Q = K @ M1, K @ M2

    data = load_all()
    db = data['db']
    est = data['est']
    gt = data['gt']
    keyframes = data['keyframes']
    cumdist = gt_cumulative_distance(gt)
    all_frames = list(range(len(est)))

    print("\n--- tracking statistics & histograms ---")
    stats_tracking(db)
    fig_matches_per_frame(db)
    fig_connectivity(db)
    fig_inlier_percentage(db)
    fig_track_length_histogram(db)

    print("\n--- absolute PnP error ---")
    fig_absolute_pnp_error(est, gt)

    print("\n--- relative errors ---")
    fig_relative_consecutive_keyframes(est, gt, data['constraints'], keyframes)
    fig_relative_subsections(est, gt, all_frames, cumdist, 'PnP')

    # ---- Bundle window details (cached heavy step) ----
    print("\n--- bundle window details ---")
    est_raw = list(np.load(EST_POSES_FILE, allow_pickle=True))  # 3x4 world-to-camera for BA
    details = generate_bundle_details(db, est_raw, K, P, Q)
    fig_bundle_mean_factor_error(details)
    fig_bundle_median_projection_error(details)
    fig_bundle_projection_vs_distance(details)

    print("\n--- projection error vs distance (PnP) ---")
    fig_pnp_projection_vs_distance(db, gt, P, Q)

    # ---- Bundle trajectory: absolute + relative sub-sections ----
    print("\n--- bundle absolute / relative errors ---")
    fig_absolute_posegraph_error(data['initial_values'], gt, 'Bundle',
                                 'fig_absolute_bundle_error.png')
    bun_dict = values_to_c2w_dict(data['initial_values'])
    bun_frames = sorted(bun_dict)
    bun_traj = [bun_dict[f] for f in bun_frames]
    fig_relative_subsections(bun_traj, gt, bun_frames, cumdist, 'Bundle')

    # ---- Loop closure pipeline (cached heavy step) ----
    print("\n--- loop closure pipeline ---")
    lc = run_loop_closure(db, data['constraints'], data['optimized_poses'],
                          data['original_graph'], K, P, Q)
    lc_frames = [s['c_n'] for s in lc['lc_stats']]

    print("\n--- trajectory overlay + pose-graph errors + LC stats + uncertainty ---")
    fig_trajectory_overlay(est, data['initial_values'], data['optimized_poses'],
                           lc['final_traj'], gt, keyframes)
    fig_absolute_posegraph_error(data['optimized_poses'], gt, 'no LC',
                                 'fig_absolute_posegraph_noLC_error.png')
    fig_absolute_posegraph_error(lc['final_traj'], gt, 'with LC',
                                 'fig_absolute_posegraph_withLC_error.png')
    fig_loop_closure_match_stats(lc['lc_stats'])
    fig_uncertainty_vs_keyframe(data['original_graph'], data['optimized_poses'],
                                lc['updated_graph'], lc['final_traj'], lc_frames)

    print("\nAll figures done.")
