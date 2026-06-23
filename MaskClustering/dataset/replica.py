import open3d as o3d
import numpy as np
import os
import cv2

REPLICA_LABELS = (
    "basket", "bed", "bench", "bin", "blanket", "blinds", "book", "bottle",
    "box", "bowl", "camera", "cabinet", "candle", "chair", "clock", "cloth",
    "comforter", "cushion", "desk", "desk-organizer", "door", "indoor-plant",
    "lamp", "monitor", "nightstand", "panel", "picture", "pillar", "pillow",
    "pipe", "plant-stand", "plate", "pot", "sculpture", "shelf", "sofa",
    "stool", "switch", "table", "tablet", "tissue-paper", "tv-screen",
    "tv-stand", "vase", "vent", "wall-plug", "window", "rug",
)

REPLICA_IDS = (
    3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 26, 29,
    34, 35, 37, 44, 47, 52, 54, 56, 59, 60, 61, 62, 63, 64, 65, 70, 71, 76,
    78, 79, 80, 82, 83, 87, 88, 91, 92, 95, 97, 98,
)


class ReplicaDataset:

    def __init__(self, seq_name, mask_dir_name=None) -> None:
        self.seq_name = seq_name
        self.root = f'./data/replica/{seq_name}'
        self.rgb_dir = f'{self.root}/color'
        self.depth_dir = f'{self.root}/depth'
        if mask_dir_name is None:
            mask_dir_name = os.environ.get('MASK_DIR_NAME', 'mask')
        self.segmentation_dir = f'{self.root}/output/{mask_dir_name}'
        self.object_dict_dir = f'{self.root}/output/object'
        self.point_cloud_path = f'{self.root}/{seq_name}_mesh.ply'
        self.mesh_path = self.point_cloud_path
        self.extrinsics_dir = f'{self.root}/poses'

        self.depth_scale = 6553.5
        self.image_size = (640, 360)

    def get_frame_list(self, stride):
        image_list = os.listdir(self.rgb_dir)
        image_list = sorted(image_list, key=lambda x: int(x.split('.')[0]))
        end = int(image_list[-1].split('.')[0]) + 1
        frame_id_list = np.arange(0, end, stride)
        return list(frame_id_list)

    def get_intrinsics(self, frame_id):
        intrinsic_path = f'{self.root}/intrinsics.txt'
        intrinsics = np.loadtxt(intrinsic_path)
        intrinisc_cam_parameters = o3d.camera.PinholeCameraIntrinsic()
        intrinisc_cam_parameters.set_intrinsics(
            self.image_size[0], self.image_size[1],
            intrinsics[0, 0], intrinsics[1, 1],
            intrinsics[0, 2], intrinsics[1, 2],
        )
        return intrinisc_cam_parameters

    def get_extrinsic(self, frame_id):
        pose_path = os.path.join(self.extrinsics_dir, str(frame_id) + '.txt')
        pose = np.loadtxt(pose_path)
        return pose

    def get_depth(self, frame_id):
        depth_path = os.path.join(self.depth_dir, str(frame_id) + '.png')
        depth = cv2.imread(depth_path, -1)
        depth = depth / self.depth_scale
        depth = depth.astype(np.float32)
        return depth

    def get_rgb(self, frame_id, change_color=True):
        rgb_path = os.path.join(self.rgb_dir, str(frame_id) + '.jpg')
        rgb = cv2.imread(rgb_path)
        if change_color:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return rgb

    def get_segmentation(self, frame_id, align_with_depth=False):
        segmentation_path = os.path.join(self.segmentation_dir, f'{frame_id}.png')
        if not os.path.exists(segmentation_path):
            assert False, f"Segmentation not found: {segmentation_path}"
        segmentation = cv2.imread(segmentation_path, cv2.IMREAD_UNCHANGED)
        return segmentation

    def get_frame_path(self, frame_id):
        rgb_path = os.path.join(self.rgb_dir, str(frame_id) + '.jpg')
        segmentation_path = os.path.join(self.segmentation_dir, f'{frame_id}.png')
        return rgb_path, segmentation_path

    def get_label_features(self):
        label_features_dict = np.load(f'data/text_features/replica.npy', allow_pickle=True).item()
        return label_features_dict

    def get_scene_points(self):
        mesh = o3d.io.read_point_cloud(self.point_cloud_path)
        vertices = np.asarray(mesh.points)
        return vertices

    def get_label_id(self):
        self.class_id = REPLICA_IDS
        self.class_label = REPLICA_LABELS
        self.label2id = {}
        self.id2label = {}
        for label, id in zip(self.class_label, self.class_id):
            self.label2id[label] = id
            self.id2label[id] = label
        return self.label2id, self.id2label
