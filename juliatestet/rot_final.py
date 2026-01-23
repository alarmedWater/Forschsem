import numpy as np
import open3d as o3d

# Lade Punktwolken
pcl0 = o3d.io.read_point_cloud(r'C:\Users\julia\Documents\Strawberry_Estimation\SuperellipseStrawberry\Data\FS_filtered3\view0.ply')
pcl1 = o3d.io.read_point_cloud(r'C:\Users\julia\Documents\Strawberry_Estimation\SuperellipseStrawberry\Data\FS_filtered3\view1.ply')
pcl2 = o3d.io.read_point_cloud(r'C:\Users\julia\Documents\Strawberry_Estimation\SuperellipseStrawberry\Data\FS_filtered3\view2.ply')

# Skalierungsfaktor (falls nötig, von step units zu Metern)
SCALE_FACTOR = 1

# Roboter-Posen (TCP im Roboter-Basis-Koordinatensystem)
# Position 0
x0 = -185.153e-3
y0 = -60.253e-3
z0 = -248.061e-3
rx0 = -63.286  # Grad
ry0 = -55.563
rz0 = -127.337

# Position 1
x1 = -144.867e-3
y1 = 0.000e-3
z1 = -283.586e-3
rx1 = 180
ry1 = -90
rz1 = 0

# Position 2
x2 = -192.338e-3
y2 = 36.919e-3
z2 = -243.173e-3
rx2 = 71.161
ry2 = -50.263
rz2 = 104.430

def get_hand_eye_transform():
    """
    Transformation von Kamera-Koordinaten zu TCP-Koordinaten
    
    Intel RealSense:          TCP (X nach unten):
    X → rechts                X → unten
    Y → unten                 Y → links  
    Z → vorne                 Z → vorne
    
    Also:
    X_tcp = Y_cam (unten)
    Y_tcp = -X_cam (von rechts zu links)
    Z_tcp = Z_cam (vorne bleibt vorne)
    """
    R_cam_to_tcp = np.array([
        [ 0.0, -1.0,  0.0],  # X_tcp = -X_cam → nach links wird zu Y_tcp
        [ 1.0,  0.0,  0.0],  # Y_tcp = X_cam → nach rechts
        [ 0.0,  0.0,  1.0]   # Z_tcp = Z_cam
    ])
    
    # Falls die Kamera einen Offset zum TCP hat, hier einfügen:
    # t_cam_to_tcp = np.array([offset_x, offset_y, offset_z])
    t_cam_to_tcp = np.array([0.0, 0.0, 0.0])
    
    # 4x4 Transformationsmatrix
    T_cam_to_tcp = np.eye(4)
    T_cam_to_tcp[:3, :3] = R_cam_to_tcp
    T_cam_to_tcp[:3, 3] = t_cam_to_tcp
    
    return T_cam_to_tcp

def euler_to_rotation_matrix(rx, ry, rz, degrees=True):
    """
    Euler-Winkel zu Rotationsmatrix (ZYX-Konvention, extrinsisch)
    Dies ist die gängigste Konvention bei Industrierobotern
    """
    if degrees:
        rx = np.radians(rx)
        ry = np.radians(ry)
        rz = np.radians(rz)
    
    # Rotation um X-Achse
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    
    # Rotation um Y-Achse
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    
    # Rotation um Z-Achse
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    
    # Kombination: Rz * Ry * Rx (extrinsisch)
    R = Rz @ Ry @ Rx
    return R

def get_tcp_to_base_transform(x, y, z, rx, ry, rz):
    """
    Transformationsmatrix vom TCP zum Roboter-Basis-Koordinatensystem
    
    WICHTIG: Die Werte werden negiert, da die Roboter-Pose die inverse
    Transformation beschreibt (Basis vom TCP aus gesehen)
    """
    R = euler_to_rotation_matrix(rx, ry, rz)
    t = np.array([x, y, z])
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return T

def transform_pointcloud(pcl, pose_params, scale_factor=1.0):
    """
    Transformiert eine Punktwolke vom Kamera-Koordinatensystem 
    ins Roboter-Basis-Koordinatensystem
    
    Transformationskette:
    Punkte (Kamera) → Hand-Eye → TCP → Roboter-Basis
    """
    x, y, z, rx, ry, rz = pose_params
    
    # 1. Skaliere Punkte
    points = np.asarray(pcl.points) * scale_factor
    
    # 2. Hand-Eye Transformation (Kamera → TCP)
    T_cam_to_tcp = get_hand_eye_transform()
    
    # 3. TCP → Roboter-Basis
    T_tcp_to_base = get_tcp_to_base_transform(x, y, z, rx, ry, rz)
    
    # 4. Gesamttransformation
    T_total = T_tcp_to_base @ T_cam_to_tcp
    
    # 5. Transformiere Punkte
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack((points, ones))
    
    transformed_points = (T_total @ points_homogeneous.T).T
    transformed_points_3d = transformed_points[:, :3]
    
    # 6. Aktualisiere Punktwolke
    pcl_transformed = o3d.geometry.PointCloud()
    pcl_transformed.points = o3d.utility.Vector3dVector(transformed_points_3d)
    
    # Kopiere Farben falls vorhanden
    if pcl.has_colors():
        pcl_transformed.colors = pcl.colors
    
    return pcl_transformed

# Transformiere alle Punktwolken
pose0 = (x0, y0, z0, rx0, ry0, rz0)
pose1 = (x1, y1, z1, rx1, ry1, rz1)
pose2 = (x2, y2, z2, rx2, ry2, rz2)

pcl0_transformed = transform_pointcloud(pcl0, pose0, SCALE_FACTOR)
pcl1_transformed = transform_pointcloud(pcl1, pose1, SCALE_FACTOR)
pcl2_transformed = transform_pointcloud(pcl2, pose2, SCALE_FACTOR)

# Färbe die Punktwolken für bessere Visualisierung
pcl0_transformed.paint_uniform_color([1, 0, 0])  # Rot
pcl1_transformed.paint_uniform_color([0, 1, 0])  # Grün
pcl2_transformed.paint_uniform_color([0, 0, 1])  # Blau

# Speichere transformierte Punktwolken
o3d.io.write_point_cloud('viewrot0.ply', pcl0_transformed)
o3d.io.write_point_cloud('viewrot1.ply', pcl1_transformed)
o3d.io.write_point_cloud('viewrot2.ply', pcl2_transformed)

# Visualisierung
print("Visualisiere alle drei Punktwolken zusammen...")
o3d.visualization.draw_geometries(
    [pcl0_transformed, pcl1_transformed, pcl2_transformed],
    window_name="Alle 3 Views (Rot=View0, Grün=View1, Blau=View2)"
)

# Optional: Merge und speichere kombinierte Punktwolke
merged_pcl = pcl0_transformed + pcl1_transformed + pcl2_transformed
o3d.io.write_point_cloud('merged_strawberry.ply', merged_pcl)
print("Kombinierte Punktwolke gespeichert als 'merged_strawberry.ply'")

# Debug-Informationen
print("\n=== Debug-Informationen ===")
print(f"View 0: {len(pcl0_transformed.points)} Punkte")
print(f"View 1: {len(pcl1_transformed.points)} Punkte")
print(f"View 2: {len(pcl2_transformed.points)} Punkte")
print(f"Merged: {len(merged_pcl.points)} Punkte")

print("\nAbstände zwischen TCP-Positionen:")
d01 = np.linalg.norm(np.array([x1-x0, y1-y0, z1-z0]))
d12 = np.linalg.norm(np.array([x2-x1, y2-y1, z2-z1]))
d02 = np.linalg.norm(np.array([x2-x0, y2-y0, z2-z0]))
print(f"Pos0 → Pos1: {d01*1000:.1f} mm")
print(f"Pos1 → Pos2: {d12*1000:.1f} mm")
print(f"Pos0 → Pos2: {d02*1000:.1f} mm")