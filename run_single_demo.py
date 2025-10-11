from estimater import *
from datareader import *
import argparse
import os
import trimesh
import numpy as np
import cv2
import imageio
import logging
import copy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/single_data/mesh/textured_simple.obj')
    parser.add_argument('--rgb_path', type=str, default=f'{code_dir}/single_data/rgb.png', help='Path to input RGB image')
    parser.add_argument('--depth_path', type=str, default=f'{code_dir}/single_data/depth.png', help='Path to input depth image (same size as RGB)')
    parser.add_argument('--mask_path', type=str, default=f'{code_dir}/single_data/mask.png', help='Path to mask file path')
    parser.add_argument('--K', type=str, default=f'{code_dir}/single_data/cam_K.txt', help='Camera intrinsic matrix txt file path')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    args = parser.parse_args()

    # === 初始化 ===
    set_logging_format()
    set_seed(0)
    os.makedirs(args.debug_dir, exist_ok=True)

    mesh = trimesh.load(args.mesh_file)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=args.debug_dir,
        debug=args.debug,
        glctx=glctx
    )
    logging.info("Estimator initialized")

    color = imageio.imread(args.rgb_path)[..., :3]
    H, W = color.shape[:2]
    color = cv2.resize(color, (W, H), interpolation=cv2.INTER_NEAREST)

    plot_img = copy.deepcopy(color)
    # cv2.imshow('Initial RGB image', color[..., ::-1])
    # cv2.waitKey(0)

    depth = cv2.imread(args.depth_path, -1) / 1e3  # mm→m
    depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
    depth[(depth < 0.001) | (depth >= np.inf)] = 0

    mask = cv2.imread(args.mask_path, -1)
    if len(mask.shape) == 3:
        for c in range(3):
            if mask[..., c].sum() > 0:
                mask = mask[..., c]
                break
    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)
    K = np.loadtxt(args.K).reshape(3, 3)

    # === 执行位姿估计 ===
    pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

    # np.savetxt(f'{args.debug_dir}/pose_result.txt', pose.reshape(4, 4))
    # logging.info("Pose estimation complete, saved to pose_result.txt")

    # === 可视化结果 ===
    # center_pose = pose @ np.linalg.inv(to_origin)
    # Directly use the estimated pose without to_origin. 
    # This is because the to_origin transformation may not be accurate for some objects.
    center_pose = pose
    vis = draw_posed_3d_box(K, img=plot_img, ob_in_cam=center_pose, bbox=bbox)
    vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
    cv2.imshow('Pose Estimation', vis[..., ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    xyz_map = depth2xyzmap(depth, K)
    valid = depth>=0.001
    pcd = toOpen3dCloud(xyz_map[valid], color[valid])
    os.makedirs(f'{code_dir}/single_data/results/', exist_ok=True)

    np.savetxt(f'{code_dir}/single_data/results/pose_result.txt', pose.reshape(4, 4))
    o3d.io.write_point_cloud(f'{code_dir}/single_data/results/pcd.pcd', pcd)
    imageio.imwrite(f'{code_dir}/single_data/results/pose_vis.png', vis)
    logging.info(f"Visualization saved to {code_dir}/single_data/results/pose_vis.png")
