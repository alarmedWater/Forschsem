import pyrealsense2 as rs
import numpy as np
import cv2
import os

class RealsenseCamera:
    def __init__(self, width=640, height=480, fps=30):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        # Start streaming
        self.pipeline.start(self.config)

        # Create an align object to align depth frames to color frames
        self.align = rs.align(rs.stream.color)

    def get_frames(self):
        """
        Returns aligned color and depth frames as numpy arrays
        """
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return color_image, depth_image

    def show_stream(self):
        """
        Live stream with option to save snapshot
        Press 's' to save snapshot
        Press 'q' to quit
        """
        snapshot_dir = "snapshots"
        os.makedirs(snapshot_dir, exist_ok=True)
        snapshot_count = 0

        try:
            while True:
                color_image, depth_image = self.get_frames()
                if color_image is None:
                    continue

                # Convert depth to colormap for visualization
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET
                )

                # Stack both images horizontally
                images = np.hstack((color_image, depth_colormap))

                cv2.imshow('Aligned RGB + Depth', images)
                key = cv2.waitKey(1)

                if key & 0xFF == ord('s'):
                    cv2.imwrite(os.path.join(snapshot_dir, f"color_{snapshot_count}.png"), color_image)
                    cv2.imwrite(os.path.join(snapshot_dir, f"depth_{snapshot_count}.png"), depth_image)
                    print(f"Saved snapshot {snapshot_count}")
                    snapshot_count += 1
                elif key & 0xFF == ord('q'):
                    break

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

    def get_snapshot(self, filename_prefix="snapshot"):
        """
        Get a single aligned frame and save it
        """
        color_image, depth_image = self.get_frames()
        if color_image is None:
            print("No frames received!")
            return

        cv2.imwrite(f"{filename_prefix}_color.png", color_image)
        cv2.imwrite(f"{filename_prefix}_depth.png", depth_image)
        print(f"Snapshot saved as {filename_prefix}_color.png and {filename_prefix}_depth.png")

if __name__ == "__main__":
    cam = RealsenseCamera()
    cam.show_stream()
