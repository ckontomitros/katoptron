from celery import shared_task
from django.core.files.base import ContentFile
from django.utils import timezone
import os
import json
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from tslearn.metrics import dtw, dtw_path
from mediapipe.framework.formats import landmark_pb2


@shared_task
def process_comparison(comparison_id):
    """Background task to process video comparison"""
    from .models import Comparison

    comparison = Comparison.objects.get(id=comparison_id)
    comparison.status = 'processing'
    comparison.save()

    try:
        # Initialize pose comparator
        comparator = PoseComparator()

        # Extract poses from both videos
        video1_path = comparison.video1.path
        video2_path = comparison.video2.path

        landmarks1 = comparator.extract_pose_landmarks(video1_path)
        landmarks2 = comparator.extract_pose_landmarks(video2_path)

        # Calculate differences
        results = comparator.calculate_pose_differences(landmarks1, landmarks2)

        # Save results to files
        output_dir = f'media/results/{comparison_id}'
        os.makedirs(output_dir, exist_ok=True)

        # Save JSON results
        json_path = os.path.join(output_dir, 'pose_differences_complete.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        comparison.result_json.name = f'results/{comparison_id}/pose_differences_complete.json'

        # Save frame differences CSV
        if results['frame_differences']:
            frames_data = []
            for fd in results['frame_differences']:
                if fd['has_pose_data']:
                    row = {
                        'aligned_frame1': fd['aligned_frame1'],
                        'aligned_frame2': fd['aligned_frame2'],
                        'total_difference': fd['total_difference'],
                        'total_angle_difference': fd['total_angle_difference']
                    }
                    sorted_landmarks = sorted(fd['landmark_differences'],
                                              key=lambda x: x['distance'], reverse=True)
                    for i, lm in enumerate(sorted_landmarks[:5]):
                        row[f'top_{i + 1}_landmark'] = lm['landmark_name']
                        row[f'top_{i + 1}_distance'] = lm['distance']

                    for ad in fd['angle_differences']:
                        row[f"{ad['joint_name']}_angle_diff"] = ad['angle_difference']

                    frames_data.append(row)

            if frames_data:
                df = pd.DataFrame(frames_data)
                csv_path = os.path.join(output_dir, 'frame_differences.csv')
                df.to_csv(csv_path, index=False)
                comparison.result_csv.name = f'results/{comparison_id}/frame_differences.csv'

        # Save landmark statistics CSV
        if results['landmark_statistics']:
            landmark_data = []
            for landmark_name, stats in results['landmark_statistics'].items():
                row = {'landmark_name': landmark_name}
                row.update(stats)
                landmark_data.append(row)

            df_landmarks = pd.DataFrame(landmark_data)
            landmark_csv_path = os.path.join(output_dir, 'landmark_statistics.csv')
            df_landmarks.to_csv(landmark_csv_path, index=False)
            comparison.landmark_csv.name = f'results/{comparison_id}/landmark_statistics.csv'

        # Save joint statistics CSV
        if results['joint_statistics']:
            joint_data = []
            for joint_name, stats in results['joint_statistics'].items():
                row = {'joint_name': joint_name}
                row.update(stats)
                joint_data.append(row)

            df_joints = pd.DataFrame(joint_data)
            joint_csv_path = os.path.join(output_dir, 'joint_statistics.csv')
            df_joints.to_csv(joint_csv_path, index=False)
            comparison.joint_csv.name = f'results/{comparison_id}/joint_statistics.csv'

        # Save summary
        summary_path = os.path.join(output_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write("POSE COMPARISON SUMMARY\n")
            f.write("=" * 50 + "\n\n")

            stats = results['statistics']
            if 'error' not in stats:
                f.write(f"Total aligned frame pairs: {stats.get('total_aligned_pairs', 'N/A')}\n")
                f.write(f"\nPOSITIONAL DIFFERENCES:\n")
                f.write(f"Average total difference: {stats.get('avg_total_difference', 'N/A'):.4f}\n")
                f.write(f"DTW Position Distance: {stats.get('dtw_position_distance', 'N/A'):.4f}\n")
                f.write(f"\nANGULAR DIFFERENCES:\n")
                f.write(f"Average angle difference: {stats.get('avg_angle_difference', 'N/A'):.2f}°\n")
                f.write(f"DTW Angle Distance: {stats.get('dtw_angle_distance', 'N/A'):.4f}\n")

        comparison.summary_txt.name = f'results/{comparison_id}/summary.txt'

        # Generate comparison video
        video_output_path = os.path.join(output_dir, 'comparison.mp4')
        comparator.generate_comparison_video(video1_path, video2_path, landmarks1,
                                             landmarks2, results, video_output_path)
        comparison.comparison_video.name = f'results/{comparison_id}/comparison.mp4'

        # Update statistics
        stats = results['statistics']
        if 'error' not in stats:
            comparison.avg_position_diff = stats['avg_total_difference']
            comparison.avg_angle_diff = stats['avg_angle_difference']
            comparison.dtw_position_distance = stats['dtw_position_distance']
            comparison.dtw_angle_distance = stats['dtw_angle_distance']
            comparison.total_aligned_pairs = stats['total_aligned_pairs']

        comparison.status = 'completed'
        comparison.completed_at = timezone.now()
        comparison.save()

    except Exception as e:
        comparison.status = 'failed'
        comparison.error_message = str(e)
        comparison.save()
        raise


class PoseComparator:
    """Embedded pose comparison class"""

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose_config = {
            'static_image_mode': False,
            'model_complexity': model_complexity,
            'enable_segmentation': False,
            'min_detection_confidence': min_detection_confidence,
            'min_tracking_confidence': min_tracking_confidence
        }

        self.landmark_names = [
            'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer', 'left_ear',
            'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder',
            'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist',
            'right_wrist', 'left_pinky', 'right_pinky', 'left_index',
            'right_index', 'left_thumb', 'right_thumb', 'left_hip',
            'right_hip', 'left_knee', 'right_knee', 'left_ankle',
            'right_ankle', 'left_heel', 'right_heel', 'left_foot_index',
            'right_foot_index'
        ]

        self.joint_angles = {
            13: [11, 15],  # left_elbow
            14: [12, 16],  # right_elbow
            25: [23, 27],  # left_knee
            26: [24, 28],  # right_knee
            11: [13, 23],  # left_shoulder
            12: [14, 24]  # right_shoulder
        }

    def extract_pose_landmarks(self, video_path: str) -> List[Dict]:
        """Extract pose landmarks from video"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        landmarks_data = []

        with self.mp_pose.Pose(**self.pose_config) as pose:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)

                frame_data = {
                    'frame_number': frame_idx,
                    'timestamp': frame_idx / fps,
                    'landmarks': None
                }

                if results.pose_landmarks:
                    landmarks = []
                    for landmark in results.pose_landmarks.landmark:
                        landmarks.append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z,
                            'visibility': landmark.visibility
                        })
                    frame_data['landmarks'] = landmarks

                landmarks_data.append(frame_data)
                frame_idx += 1

        cap.release()
        return landmarks_data

    def calculate_euclidean_distance(self, point1: Dict, point2: Dict) -> float:
        dx = point1['x'] - point2['x']
        dy = point1['y'] - point2['y']
        dz = point1['z'] - point2['z']
        return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    def normalize_landmarks(self, landmarks: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        normalized_seq = []
        original_indices = []
        for f, frame in enumerate(landmarks):
            if frame['landmarks'] is None:
                continue
            lms = frame['landmarks']
            ref_hip = np.mean([[lms[23]['x'], lms[23]['y'], lms[23]['z']],
                               [lms[24]['x'], lms[24]['y'], lms[24]['z']]], axis=0)
            centered = np.array([[lm['x'] - ref_hip[0], lm['y'] - ref_hip[1],
                                  lm['z'] - ref_hip[2]] for lm in lms])
            neck = np.array([lms[12]['x'], lms[12]['y'], lms[12]['z']])
            torso_len = np.linalg.norm(neck - ref_hip) + 1e-8
            normalized = centered / torso_len
            normalized_seq.append(normalized.flatten())
            original_indices.append(f)
        return np.array(normalized_seq), np.array(original_indices)

    def extract_angle_sequences(self, landmarks: List[Dict]) -> np.ndarray:
        angle_seq = []
        for frame in landmarks:
            if frame['landmarks'] is None:
                continue
            lms = frame['landmarks']
            angles = []
            for joint_idx, adj in self.joint_angles.items():
                def calc_angle(p1, p2, p3):
                    v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y'], p1['z'] - p2['z']])
                    v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y'], p3['z'] - p2['z']])
                    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                    return np.arccos(np.clip(cos, -1, 1)) * 180 / np.pi

                try:
                    angle = calc_angle(lms[adj[0]], lms[joint_idx], lms[adj[1]])
                except:
                    angle = 0.0
                angles.append(angle)
            angle_seq.append(angles)
        return np.array(angle_seq)

    def calculate_pose_differences(self, landmarks1: List[Dict], landmarks2: List[Dict]) -> Dict:
        pos_seq1, idx1 = self.normalize_landmarks(landmarks1)
        pos_seq2, idx2 = self.normalize_landmarks(landmarks2)
        angle_seq1 = self.extract_angle_sequences(landmarks1)
        angle_seq2 = self.extract_angle_sequences(landmarks2)

        if len(pos_seq1) == 0 or len(pos_seq2) == 0:
            return {
                'frame_differences': [],
                'statistics': {'error': 'No valid pose data found'},
                'landmark_statistics': {},
                'joint_statistics': {},
                'dtw_path': []
            }

        pos_dtw_dist = dtw(pos_seq1, pos_seq2)
        pos_path = dtw_path(pos_seq1, pos_seq2)[0]
        angle_dtw_dist = dtw(angle_seq1, angle_seq2)

        frame_differences = []
        landmark_diffs_all = []
        angle_diffs_all = []

        for seq_i, seq_j in pos_path:
            orig_i = idx1[seq_i]
            orig_j = idx2[seq_j]
            frame1 = landmarks1[orig_i]
            frame2 = landmarks2[orig_j]

            if frame1['landmarks'] is None or frame2['landmarks'] is None:
                continue

            landmark_diffs = []
            total_pos_diff = 0.0
            for k, (lm1, lm2) in enumerate(zip(frame1['landmarks'], frame2['landmarks'])):
                dist = self.calculate_euclidean_distance(lm1, lm2)
                landmark_diffs.append({
                    'landmark_index': k,
                    'landmark_name': self.landmark_names[k],
                    'distance': dist
                })
                total_pos_diff += dist
            landmark_diffs_all.extend(landmark_diffs)

            angle_diffs = []
            total_angle_diff = 0.0
            for joint_idx, adj in self.joint_angles.items():
                angle_diff = self.calculate_angular_difference(
                    frame1['landmarks'], frame2['landmarks'], joint_idx, adj)
                angle_diffs.append({
                    'joint_index': joint_idx,
                    'joint_name': self.landmark_names[joint_idx],
                    'angle_difference': angle_diff
                })
                total_angle_diff += angle_diff
            angle_diffs_all.extend(angle_diffs)

            frame_differences.append({
                'aligned_frame1': int(orig_i),
                'aligned_frame2': int(orig_j),
                'total_difference': float(total_pos_diff),
                'total_angle_difference': float(total_angle_diff),
                'landmark_differences': landmark_diffs,
                'angle_differences': angle_diffs,
                'has_pose_data': True
            })

        if frame_differences:
            total_diffs = [fd['total_difference'] for fd in frame_differences]
            total_angle_diffs = [fd['total_angle_difference'] for fd in frame_differences]
            statistics = {
                'total_aligned_pairs': len(frame_differences),
                'avg_total_difference': float(np.mean(total_diffs)),
                'max_total_difference': float(np.max(total_diffs)),
                'min_total_difference': float(np.min(total_diffs)),
                'std_total_difference': float(np.std(total_diffs)),
                'avg_angle_difference': float(np.mean(total_angle_diffs)),
                'max_angle_difference': float(np.max(total_angle_diffs)),
                'min_angle_difference': float(np.min(total_angle_diffs)),
                'std_angle_difference': float(np.std(total_angle_diffs)),
                'dtw_position_distance': float(pos_dtw_dist),
                'dtw_angle_distance': float(angle_dtw_dist)
            }

            landmark_stats = {}
            for name in self.landmark_names:
                diffs = [ld['distance'] for ld in landmark_diffs_all if ld['landmark_name'] == name]
                if diffs:
                    landmark_stats[name] = {
                        'avg_difference': float(np.mean(diffs)),
                        'max_difference': float(np.max(diffs)),
                        'min_difference': float(np.min(diffs)),
                        'std_difference': float(np.std(diffs))
                    }

            joint_stats = {}
            unique_joints = set(ad['joint_name'] for ad in angle_diffs_all)
            for name in unique_joints:
                diffs = [ad['angle_difference'] for ad in angle_diffs_all if ad['joint_name'] == name]
                if diffs:
                    joint_stats[name] = {
                        'avg_angle_difference': float(np.mean(diffs)),
                        'max_angle_difference': float(np.max(diffs)),
                        'min_angle_difference': float(np.min(diffs)),
                        'std_angle_difference': float(np.std(diffs))
                    }
        else:
            statistics = {'error': 'No aligned poses found'}
            landmark_stats = {}
            joint_stats = {}

        return {
            'frame_differences': frame_differences,
            'statistics': statistics,
            'landmark_statistics': landmark_stats,
            'joint_statistics': joint_stats,
            'dtw_path': [[int(i), int(j)] for i, j in pos_path]
        }

    def calculate_angular_difference(self, landmarks1: List[Dict], landmarks2: List[Dict],
                                     joint_idx: int, adjacent_indices: List[int]) -> float:
        def calculate_angle(p1, p2, p3):
            v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y'], p1['z'] - p2['z']])
            v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y'], p3['z'] - p2['z']])
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            return np.arccos(cos_angle) * 180.0 / np.pi

        if len(adjacent_indices) < 2:
            return 0.0

        try:
            angle1 = calculate_angle(landmarks1[adjacent_indices[0]],
                                     landmarks1[joint_idx],
                                     landmarks1[adjacent_indices[1]])
            angle2 = calculate_angle(landmarks2[adjacent_indices[0]],
                                     landmarks2[joint_idx],
                                     landmarks2[adjacent_indices[1]])
            return abs(angle1 - angle2)
        except:
            return 0.0

    def generate_comparison_video(self, video1_path: str, video2_path: str,
                                  landmarks1: List[Dict], landmarks2: List[Dict],
                                  results: Dict, output_path: str = 'comparison.mp4'):
        if 'error' in results['statistics']:
            return

        cap2 = cv2.VideoCapture(video2_path)
        fps = cap2.get(cv2.CAP_PROP_FPS)
        width = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Use H264 codec for better browser compatibility
        # Try different codecs in order of preference
        fourcc_options = [
            cv2.VideoWriter_fourcc(*'avc1'),  # H.264
            cv2.VideoWriter_fourcc(*'H264'),  # H.264 alternative
            cv2.VideoWriter_fourcc(*'X264'),  # H.264 alternative
            cv2.VideoWriter_fourcc(*'mp4v'),  # MPEG-4 fallback
        ]

        writer = None
        for fourcc in fourcc_options:
            temp_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if temp_writer.isOpened():
                writer = temp_writer
                break
            temp_writer.release()

        if writer is None:
            raise RuntimeError("Could not initialize video writer with any codec")

        pos_path = results['dtw_path']
        _, idx1 = self.normalize_landmarks(landmarks1)
        _, idx2 = self.normalize_landmarks(landmarks2)
        frame_differences = results['frame_differences']

        for k, (seq_i, seq_j) in enumerate(pos_path):
            orig_i = idx1[seq_i]
            orig_j = idx2[seq_j]

            cap2.set(cv2.CAP_PROP_POS_FRAMES, orig_j)
            ret, frame = cap2.read()
            if not ret:
                continue

            lm1 = landmarks1[orig_i]['landmarks']
            lm2 = landmarks2[orig_j]['landmarks']

            def to_landmark_list(lms):
                landmark_list = landmark_pb2.NormalizedLandmarkList()
                for lm in lms:
                    landmark_list.landmark.add(x=lm['x'], y=lm['y'], z=lm['z'],
                                               visibility=lm['visibility'])
                return landmark_list

            lm_list1 = to_landmark_list(lm1)
            lm_list2 = to_landmark_list(lm2)

            self.mp_drawing.draw_landmarks(
                frame, lm_list2, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

            self.mp_drawing.draw_landmarks(
                frame, lm_list1, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

            fd = frame_differences[k]
            threshold = 0.05
            for ld in fd['landmark_differences']:
                if ld['distance'] > threshold:
                    lm = lm2[ld['landmark_index']]
                    x = int(lm['x'] * width)
                    y = int(lm['y'] * height)
                    cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)

            cv2.putText(frame, f"Diff: {fd['total_difference']:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            writer.write(frame)

        cap2.release()
        writer.release()
