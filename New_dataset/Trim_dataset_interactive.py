#!/usr/bin/env python3
"""
Interactive Video-based Dataset Trimming Tool

This script allows you to trim HDF5 dataset files by watching the video
and selecting start/end frames interactively.

Usage:
    python Trim_dataset_interactive.py episode.h5

Controls:
    SPACE       - Pause/Resume playback
    LEFT/RIGHT  - Move backward/forward 1 frame
    UP/DOWN     - Move backward/forward 10 frames
    S           - Mark START frame
    E           - Mark END frame
    R           - Reset marks
    Q           - Save trimmed dataset and quit
    ESC         - Cancel and quit
"""

import h5py
import cv2
import numpy as np
import sys
from pathlib import Path


class InteractiveVideoTrimmer:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.current_frame = 0
        self.start_frame = None
        self.end_frame = None
        self.playing = True
        self.fps = 15

        # Load data
        print(f"📂 Loading dataset: {self.file_path.name}")
        self.load_data()

    def load_data(self):
        """Load video frames and metadata from HDF5 file."""
        with h5py.File(self.file_path, 'r') as f:
            # Load images
            images_grp = f['observations/images']
            self.cam_keys = sorted(list(images_grp.keys()))
            self.total_frames = len(f['action'])

            print(f"📷 Cameras: {self.cam_keys}")
            print(f"⏱️  Total frames: {self.total_frames} (~{self.total_frames/self.fps:.1f} sec)")
            print(f"📥 Loading and decoding video frames to memory...")

            # Load all frames to memory for smooth playback
            self.video_streams = {}
            for cam in self.cam_keys:
                compressed_frames = images_grp[cam][:]

                # Check if JPEG compressed
                first_frame = compressed_frames[0]
                decoded_first = cv2.imdecode(first_frame, cv2.IMREAD_COLOR)

                if decoded_first is not None:
                    # JPEG compressed
                    frames = []
                    for i, comp_frame in enumerate(compressed_frames):
                        if i % 50 == 0:
                            print(f"  {cam}: {i}/{self.total_frames} frames decoded...")
                        decoded = cv2.imdecode(comp_frame, cv2.IMREAD_COLOR)
                        frames.append(decoded)
                    self.video_streams[cam] = frames
                else:
                    # Raw pixels - convert to list
                    self.video_streams[cam] = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                               for frame in compressed_frames]

                print(f"  ✅ {cam}: {len(self.video_streams[cam])} frames loaded")

            # Load action data for display
            self.actions = f['action'][:]
            self.qpos = f['observations/qpos'][:]

        print(f"✅ Data loaded successfully\n")

    def get_combined_frame(self, frame_idx):
        """Get combined frame from all cameras."""
        frames = []
        max_height = 0

        for cam in self.cam_keys:
            frame = self.video_streams[cam][frame_idx].copy()

            # Add camera label
            cv2.putText(frame, f"{cam}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            frames.append(frame)
            max_height = max(max_height, frame.shape[0])

        # Resize frames to same height if needed
        resized_frames = []
        for frame in frames:
            if frame.shape[0] != max_height:
                aspect_ratio = frame.shape[1] / frame.shape[0]
                new_width = int(max_height * aspect_ratio)
                frame = cv2.resize(frame, (new_width, max_height))
            resized_frames.append(frame)

        # Combine horizontally
        combined = np.hstack(resized_frames)
        return combined

    def add_info_overlay(self, frame, frame_idx):
        """Add information overlay to frame."""
        h, w = frame.shape[:2]
        info_height = 180

        # Create info panel
        info_panel = np.zeros((info_height, w, 3), dtype=np.uint8)

        # Current frame info
        curr_act = self.actions[frame_idx]
        curr_q = self.qpos[frame_idx]

        y_pos = 25
        line_height = 25

        # Frame counter
        cv2.putText(info_panel, f"Frame: {frame_idx}/{self.total_frames-1} ({frame_idx/self.fps:.2f}s)",
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += line_height

        # Start/End markers
        start_text = f"START: {self.start_frame}" if self.start_frame is not None else "START: Not set"
        end_text = f"END: {self.end_frame}" if self.end_frame is not None else "END: Not set"
        marker_color = (0, 255, 0) if self.start_frame is not None and self.end_frame is not None else (100, 100, 100)
        cv2.putText(info_panel, f"{start_text}  |  {end_text}",
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, marker_color, 2)
        y_pos += line_height

        # Action data
        cv2.putText(info_panel, f"Action(Move): [{curr_act[0]:.2f}, {curr_act[1]:.2f}, {curr_act[2]:.2f}]",
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        y_pos += line_height

        cv2.putText(info_panel, f"Action(Rot):  [{curr_act[3]:.2f}, {curr_act[4]:.2f}, {curr_act[5]:.2f}]",
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        y_pos += line_height

        # Controls
        cv2.putText(info_panel, "Controls: SPACE=Pause | ←→=Frame | ↑↓=10frames | S=Start | E=End | Q=Save | ESC=Cancel",
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += line_height

        # Status
        status = "PLAYING" if self.playing else "PAUSED"
        status_color = (0, 255, 0) if self.playing else (0, 165, 255)
        cv2.putText(info_panel, f"Status: {status}",
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # Combine frame and info panel
        combined = np.vstack([frame, info_panel])

        # Add visual markers for start/end frames
        if self.start_frame is not None and frame_idx == self.start_frame:
            cv2.rectangle(combined, (0, 0), (combined.shape[1], combined.shape[0]),
                         (0, 255, 0), 8)
            cv2.putText(combined, "START FRAME", (combined.shape[1]//2 - 100, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        if self.end_frame is not None and frame_idx == self.end_frame:
            cv2.rectangle(combined, (0, 0), (combined.shape[1], combined.shape[0]),
                         (0, 0, 255), 8)
            cv2.putText(combined, "END FRAME", (combined.shape[1]//2 - 80, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        return combined

    def run(self):
        """Run interactive video player."""
        window_name = f"Dataset Trimmer - {self.file_path.name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("\n🎬 Starting interactive video player...")
        print("\nControls:")
        print("  SPACE       - Pause/Resume")
        print("  ← →         - Move 1 frame")
        print("  ↑ ↓         - Move 10 frames")
        print("  S           - Mark START frame")
        print("  E           - Mark END frame")
        print("  R           - Reset marks")
        print("  Q           - Save and quit")
        print("  ESC         - Cancel\n")

        delay = int(1000 / self.fps)  # milliseconds per frame

        while True:
            # Get and display frame
            combined_frame = self.get_combined_frame(self.current_frame)
            display_frame = self.add_info_overlay(combined_frame, self.current_frame)

            # Resize if too large
            if display_frame.shape[1] > 1920:
                scale = 1920 / display_frame.shape[1]
                new_width = 1920
                new_height = int(display_frame.shape[0] * scale)
                display_frame = cv2.resize(display_frame, (new_width, new_height))

            cv2.imshow(window_name, display_frame)

            # Handle input
            key = cv2.waitKey(delay if self.playing else 0) & 0xFF

            if key == 27:  # ESC
                print("\n❌ Cancelled")
                cv2.destroyAllWindows()
                return False

            elif key == ord(' '):  # SPACE - pause/play
                self.playing = not self.playing

            elif key == 83 or key == 82:  # RIGHT arrow (or special key code)
                self.current_frame = min(self.current_frame + 1, self.total_frames - 1)
                self.playing = False

            elif key == 81 or key == 84:  # LEFT arrow
                self.current_frame = max(self.current_frame - 1, 0)
                self.playing = False

            elif key == 82 or key == 85:  # UP arrow
                self.current_frame = max(self.current_frame - 10, 0)
                self.playing = False

            elif key == 84 or key == 86:  # DOWN arrow
                self.current_frame = min(self.current_frame + 10, self.total_frames - 1)
                self.playing = False

            elif key == ord('s') or key == ord('S'):  # Mark start
                self.start_frame = self.current_frame
                print(f"✅ START frame set to: {self.start_frame}")

            elif key == ord('e') or key == ord('E'):  # Mark end
                self.end_frame = self.current_frame
                print(f"✅ END frame set to: {self.end_frame}")

            elif key == ord('r') or key == ord('R'):  # Reset marks
                self.start_frame = None
                self.end_frame = None
                print("🔄 Marks reset")

            elif key == ord('q') or key == ord('Q'):  # Save and quit
                if self.start_frame is None or self.end_frame is None:
                    print("\n⚠️  Please set both START and END frames first!")
                    continue

                if self.start_frame >= self.end_frame:
                    print("\n⚠️  START frame must be before END frame!")
                    continue

                cv2.destroyAllWindows()
                return True

            # Auto advance if playing
            if self.playing:
                self.current_frame += 1
                if self.current_frame >= self.total_frames:
                    self.current_frame = 0  # Loop

    def save_trimmed_dataset(self, output_path=None):
        """Save trimmed dataset to new file."""
        if output_path is None:
            output_name = f"{self.file_path.stem}_trimmed_{self.start_frame}_{self.end_frame}.h5"
            output_path = self.file_path.parent / output_name

        print(f"\n🔧 Saving trimmed dataset...")
        print(f"  Frames: {self.start_frame} to {self.end_frame} (total: {self.end_frame - self.start_frame})")
        print(f"  Output: {output_path}")

        with h5py.File(self.file_path, 'r') as f_in:
            with h5py.File(output_path, 'w') as f_out:
                # Copy action
                action_data = f_in['action'][self.start_frame:self.end_frame]
                f_out.create_dataset('action', data=action_data)

                # Copy timestamp
                if 'timestamp' in f_in:
                    timestamp_data = f_in['timestamp'][self.start_frame:self.end_frame]
                    f_out.create_dataset('timestamp', data=timestamp_data)

                # Create observations group
                obs_grp = f_out.create_group('observations')

                # Copy qpos
                qpos_data = f_in['observations/qpos'][self.start_frame:self.end_frame]
                obs_grp.create_dataset('qpos', data=qpos_data)

                # Copy ee_pose
                if 'observations/ee_pose' in f_in:
                    ee_pose_data = f_in['observations/ee_pose'][self.start_frame:self.end_frame]
                    obs_grp.create_dataset('ee_pose', data=ee_pose_data)

                # Copy images
                images_grp_in = f_in['observations/images']
                images_grp_out = obs_grp.create_group('images')

                for cam in self.cam_keys:
                    frames = images_grp_in[cam][self.start_frame:self.end_frame]
                    images_grp_out.create_dataset(
                        cam,
                        data=frames,
                        dtype=h5py.special_dtype(vlen=np.uint8)
                    )
                    print(f"  ✓ {cam}: {len(frames)} frames")

                # Copy sensor data if exists
                if 'observations/sensor' in f_in:
                    sensor_grp_in = f_in['observations/sensor']
                    sensor_grp_out = obs_grp.create_group('sensor')

                    if 'force' in sensor_grp_in:
                        force_data = sensor_grp_in['force'][self.start_frame:self.end_frame]
                        sensor_grp_out.create_dataset('force', data=force_data)

                    if 'aline' in sensor_grp_in:
                        aline_data = sensor_grp_in['aline'][self.start_frame:self.end_frame]
                        sensor_grp_out.create_dataset('aline', data=aline_data)

        # Show file sizes
        input_size_mb = self.file_path.stat().st_size / (1024 * 1024)
        output_size_mb = output_path.stat().st_size / (1024 * 1024)

        print(f"\n✅ Trimmed dataset saved successfully!")
        print(f"\n📊 Summary:")
        print(f"  Original: {self.file_path.name}")
        print(f"    - Frames: {self.total_frames}")
        print(f"    - Size: {input_size_mb:.2f} MB")
        print(f"  Trimmed: {output_path.name}")
        print(f"    - Frames: {self.end_frame - self.start_frame}")
        print(f"    - Size: {output_size_mb:.2f} MB ({output_size_mb/input_size_mb*100:.1f}%)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python Trim_dataset_interactive.py <episode.h5>")
        sys.exit(1)

    file_path = Path(sys.argv[1])

    if not file_path.exists():
        print(f"❌ Error: File not found: {file_path}")
        sys.exit(1)

    if not file_path.suffix == '.h5':
        print(f"❌ Error: File must be an HDF5 file (.h5)")
        sys.exit(1)

    # Create trimmer and run
    trimmer = InteractiveVideoTrimmer(file_path)

    if trimmer.run():
        # User pressed Q to save
        trimmer.save_trimmed_dataset()

    print("\n👋 Done!")


if __name__ == "__main__":
    main()
