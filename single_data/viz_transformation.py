import numpy as np
import open3d as o3d

pcd = o3d.io.read_point_cloud("./results/pcd.pcd")

# F_o: object frame
# F_c: camera frame
# pose: transformation from F_c to F_o
pose = np.loadtxt("./results/pose_result.txt")

camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.5)

obj_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.2)
obj_frame.transform(pose)
o3d.visualization.draw_geometries([pcd, camera_frame, obj_frame])
