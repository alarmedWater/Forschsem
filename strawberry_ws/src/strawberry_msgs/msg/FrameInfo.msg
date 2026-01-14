std_msgs/Header header

# Stable identifiers
int32 frame_index
int32 plant_id
int32 view_id

# Paths (optional for live camera, but useful for folder replay)
string rgb_path
string depth_path

# Pose of camera in world coordinates (if available)
geometry_msgs/Pose camera_pose_world
string world_frame_id

# --- Added for robust synchronization & auditing ---
# Original timestamp produced by the source (camera_folder)
builtin_interfaces/Time source_stamp

# Unique frame identifier (stable across topics/nodes)
string frame_uid

# Allows you to detect mixed pipelines / old bags etc.
uint32 pipeline_version
