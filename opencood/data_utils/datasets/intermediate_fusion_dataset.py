"""
Dataset class for intermediate fusion
"""

import math
import logging
from collections import OrderedDict

import numpy as np
import torch

import opencood.data_utils.datasets
import opencood.data_utils.post_processor as post_processor
from opencood.utils import box_utils
from opencood.data_utils.datasets import basedataset
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.utils.pcd_utils import mask_points_by_range, mask_ego_points, shuffle_points, downsample_lidar_minimum

logger = logging.getLogger("cavise.OpenCOOD.opencood.data_utils.datasets.intermediate_fusion_dataset")


class IntermediateFusionDataset(basedataset.BaseDataset):
    """
    This class is for intermediate fusion where each vehicle transmit the
    deep features to ego.
    """

    def __init__(self, params, visualize, train=True, message_handler=None):
        super(IntermediateFusionDataset, self).__init__(params, visualize, train)

        # if project first, cav's lidar will first be projected to
        # the ego's coordinate frame. otherwise, the feature will be
        # projected instead.
        self.proj_first = True
        if "proj_first" in params["fusion"]["args"] and not params["fusion"]["args"]["proj_first"]:
            self.proj_first = False

        # whether there is a time delay between the time that cav project
        # lidar to ego and the ego receive the delivered feature
        if "cur_ego_pose_flag" in params["fusion"]["args"]:
            self.cur_ego_pose_flag = params["fusion"]["args"]["cur_ego_pose_flag"]
        else:
            self.cur_ego_pose_flag = True

        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = post_processor.build_postprocessor(params["postprocess"], train)

        self.message_handler = message_handler
        self.module_name = "OpenCOOD.IntermediateFusionDataset"

    @staticmethod
    def __wrap_ndarray(ndarray):
        return {"data": ndarray.tobytes(), "shape": ndarray.shape, "dtype": str(ndarray.dtype)}

    def extract_data(self, idx):
        base_data_dict = self.retrieve_base_data(idx, cur_ego_pose_flag=self.cur_ego_pose_flag)
        _, ego_lidar_pose = self.__find_ego_vehicle(base_data_dict)

        if self.message_handler is not None:
            for cav_id, selected_cav_base in base_data_dict.items():
                selected_cav_processed = self.get_item_single_car(selected_cav_base, ego_lidar_pose)

                with self.message_handler.handle_opencda_message(cav_id, self.module_name) as msg:
                    msg["infra"] = {"name": "infra", "label": "LABEL_OPTIONAL", "type": "int64", "data": 1 if "rsu" in cav_id else 0}

                    msg["velocity"] = {"name": "velocity", "label": "LABEL_OPTIONAL", "type": "float", "data": selected_cav_processed["velocity"]}

                    msg["time_delay"] = {
                        "name": "time_delay",
                        "label": "LABEL_OPTIONAL",
                        "type": "float",
                        "data": float(selected_cav_base["time_delay"]),
                    }

                    msg["object_ids"] = {
                        "name": "object_ids",
                        "label": "LABEL_REPEATED",
                        "type": "int64",
                        "data": selected_cav_processed["object_ids"],
                    }

                    msg["object_bbx_center"] = {
                        "name": "object_bbx_center",
                        "label": "LABEL_OPTIONAL",
                        "type": "NDArray",
                        "data": self.__wrap_ndarray(selected_cav_processed["object_bbx_center"]),
                    }

                    msg["spatial_correction_matrix"] = {
                        "name": "spatial_correction_matrix",
                        "label": "LABEL_OPTIONAL",
                        "type": "NDArray",
                        "data": self.__wrap_ndarray(selected_cav_base["params"]["spatial_correction_matrix"]),
                    }

                    msg["voxel_num_points"] = {
                        "name": "voxel_num_points",
                        "label": "LABEL_OPTIONAL",
                        "type": "NDArray",
                        "data": self.__wrap_ndarray(selected_cav_processed["processed_features"]["voxel_num_points"]),
                    }

                    msg["voxel_features"] = {
                        "name": "voxel_features",
                        "label": "LABEL_OPTIONAL",
                        "type": "NDArray",
                        "data": self.__wrap_ndarray(selected_cav_processed["processed_features"]["voxel_features"]),
                    }

                    msg["voxel_coords"] = {
                        "name": "voxel_coords",
                        "label": "LABEL_OPTIONAL",
                        "type": "NDArray",
                        "data": self.__wrap_ndarray(selected_cav_processed["processed_features"]["voxel_coords"]),
                    }

                    if self.visualize:
                        msg["projected_lidar"] = {
                            "name": "projected_lidar",
                            "label": "LABEL_OPTIONAL",
                            "type": "NDArray",
                            "data": self.__wrap_ndarray(selected_cav_processed["projected_lidar"]),
                        }
                    else:
                        msg["projected_lidar_raw"] = None

    def __find_ego_vehicle(self, base_data_dict):
        ego_id = -1
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content["ego"]:
                ego_id = cav_id
                ego_lidar_pose = cav_content["params"]["lidar_pose"]
                break

        assert cav_id == list(base_data_dict.keys())[0], "The first element in the OrderedDict must be ego"
        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        return ego_id, ego_lidar_pose

    def __prepare_object_stack(self, object_stack, object_id_stack):
        # exclude all repetitive objects
        unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]

        # make sure bounding boxes across all frames have the same number
        object_bbx_center = np.zeros((self.params["postprocess"]["max_num"], 7))
        mask = np.zeros(self.params["postprocess"]["max_num"])
        object_bbx_center[: object_stack.shape[0], :] = object_stack
        mask[: object_stack.shape[0]] = 1

        filtered_object_ids = [object_id_stack[i] for i in unique_indices]

        return object_bbx_center, mask, filtered_object_ids

    def __pad_spatial_matrix(self, matrix_list):
        matrix_list = np.stack(matrix_list)
        padding_eye = np.tile(np.eye(4)[None], (self.max_cav - len(matrix_list), 1, 1))
        return np.concatenate([matrix_list, padding_eye], axis=0)

    def __pad_to_max(self, lst, pad_value):
        return lst + (self.max_cav - len(lst)) * [pad_value]

    def __process_with_messages(self, ego_id, ego_lidar_pose, base_data_dict):
        processed_features = []
        object_stack = []
        object_id_stack = []
        velocity = []
        time_delay = []
        infra = []
        spatial_correction_matrix = []
        projected_lidar_stack = [] if self.visualize else None

        ego_cav_base = base_data_dict.get(ego_id)
        ego_cav_processed = self.get_item_single_car(ego_cav_base, ego_lidar_pose)

        infra.append(1 if "rsu" in ego_id else 0)
        velocity.append(ego_cav_processed["velocity"])
        time_delay.append(float(ego_cav_base["time_delay"]))
        object_id_stack += ego_cav_processed["object_ids"]
        object_stack.append(ego_cav_processed["object_bbx_center"])
        spatial_correction_matrix.append(ego_cav_base["params"]["spatial_correction_matrix"])
        processed_features.append(ego_cav_processed["processed_features"])
        if self.visualize:
            projected_lidar_stack.append(ego_cav_processed["projected_lidar"])

        if ego_id in self.message_handler.current_message_artery:
            for cav_id, _ in base_data_dict.items():
                if cav_id in self.message_handler.current_message_artery[ego_id]:
                    with self.message_handler.handle_artery_message(ego_id, cav_id, self.module_name) as msg:
                        infra.append(msg["infra"])
                        velocity.append(msg["velocity"])
                        time_delay.append(msg["time_delay"])
                        object_id_stack += msg["object_ids"]

                        bbx = np.frombuffer(msg["object_bbx_center"]["data"], np.dtype(msg["object_bbx_center"]["dtype"]))
                        bbx = bbx.reshape(msg["object_bbx_center"]["shape"])
                        object_stack.append(bbx)

                        scm = np.frombuffer(msg["spatial_correction_matrix"]["data"], np.dtype(msg["spatial_correction_matrix"]["dtype"]))
                        scm = scm.reshape(msg["spatial_correction_matrix"]["shape"])
                        spatial_correction_matrix.append(scm)

                        voxel_num_points = np.frombuffer(msg["voxel_num_points"]["data"], np.dtype(msg["voxel_num_points"]["dtype"]))
                        voxel_num_points = voxel_num_points.reshape(msg["voxel_num_points"]["shape"])

                        voxel_features = np.frombuffer(msg["voxel_features"]["data"], np.dtype(msg["voxel_features"]["dtype"]))
                        voxel_features = voxel_features.reshape(msg["voxel_features"]["shape"])

                        voxel_coords = np.frombuffer(msg["voxel_coords"]["data"], np.dtype(msg["voxel_coords"]["dtype"]))
                        voxel_coords = voxel_coords.reshape(msg["voxel_coords"]["shape"])

                        processed_features.append(
                            {"voxel_num_points": voxel_num_points, "voxel_features": voxel_features, "voxel_coords": voxel_coords}
                        )

                        if self.visualize:
                            projected = np.frombuffer(msg["projected_lidar"]["data"], np.dtype(msg["projected_lidar"]["dtype"]))
                            projected = projected.reshape(msg["projected_lidar"]["shape"])
                            projected_lidar_stack.append(projected)

        return {
            "processed_features": processed_features,
            "object_stack": object_stack,
            "object_id_stack": object_id_stack,
            "velocity": velocity,
            "time_delay": time_delay,
            "infra": infra,
            "spatial_correction_matrix": spatial_correction_matrix,
            "projected_lidar_stack": projected_lidar_stack or [],
        }

    def __process_without_messages(self, ego_lidar_pose, base_data_dict):
        processed_features = []
        object_stack = []
        object_id_stack = []
        velocity = []
        time_delay = []
        infra = []
        spatial_correction_matrix = []
        projected_lidar_stack = [] if self.visualize else None

        for cav_id, selected_cav_base in base_data_dict.items():
            dx = selected_cav_base["params"]["lidar_pose"][0] - ego_lidar_pose[0]
            dy = selected_cav_base["params"]["lidar_pose"][1] - ego_lidar_pose[1]
            distance = math.hypot(dx, dy)

            if distance > opencood.data_utils.datasets.COM_RANGE:
                continue

            selected_cav_processed = self.get_item_single_car(selected_cav_base, ego_lidar_pose)

            infra.append(1 if "rsu" in cav_id else 0)
            velocity.append(selected_cav_processed["velocity"])
            time_delay.append(float(selected_cav_base["time_delay"]))
            object_id_stack += selected_cav_processed["object_ids"]
            object_stack.append(selected_cav_processed["object_bbx_center"])
            spatial_correction_matrix.append(selected_cav_base["params"]["spatial_correction_matrix"])
            processed_features.append(selected_cav_processed["processed_features"])

            if self.visualize:
                projected_lidar_stack.append(selected_cav_processed["projected_lidar"])

        return {
            "processed_features": processed_features,
            "object_stack": object_stack,
            "object_id_stack": object_id_stack,
            "velocity": velocity,
            "time_delay": time_delay,
            "infra": infra,
            "spatial_correction_matrix": spatial_correction_matrix,
            "projected_lidar_stack": projected_lidar_stack or [],
        }

    def __getitem__(self, idx):
        base_data_dict = self.retrieve_base_data(idx, cur_ego_pose_flag=self.cur_ego_pose_flag)
        processed_data_dict = OrderedDict()
        processed_data_dict["ego"] = {}

        ego_id, ego_lidar_pose = self.__find_ego_vehicle(base_data_dict)
        pairwise_t_matrix = self.get_pairwise_transformation(base_data_dict, self.max_cav)

        if self.message_handler is not None:
            data = self.__process_with_messages(ego_id, ego_lidar_pose, base_data_dict)
        else:
            data = self.__process_without_messages(ego_lidar_pose, base_data_dict)

        object_bbx_center, mask, object_ids = self.__prepare_object_stack(data["object_stack"], data["object_id_stack"])

        merged_feature_dict = self.merge_features_to_dict(data["processed_features"])
        anchor_box = self.post_processor.generate_anchor_box()
        label_dict = self.post_processor.generate_label(gt_box_center=object_bbx_center, anchors=anchor_box, mask=mask)

        spatial_correction_matrix = self.__pad_spatial_matrix(data["spatial_correction_matrix"])
        velocity = self.__pad_to_max(data["velocity"], 0.0)
        time_delay = self.__pad_to_max(data["time_delay"], 0.0)
        infra = self.__pad_to_max(data["infra"], 0.0)

        processed_data_dict["ego"].update(
            {
                "object_bbx_center": object_bbx_center,
                "object_bbx_mask": mask,
                "object_ids": object_ids,
                "anchor_box": anchor_box,
                "processed_lidar": merged_feature_dict,
                "label_dict": label_dict,
                "cav_num": len(data["processed_features"]),
                "velocity": velocity,
                "time_delay": time_delay,
                "infra": infra,
                "spatial_correction_matrix": spatial_correction_matrix,
                "pairwise_t_matrix": pairwise_t_matrix,
            }
        )

        if self.visualize:
            processed_data_dict["ego"]["origin_lidar"] = np.vstack(data["projected_lidar_stack"])

        return processed_data_dict

    def get_item_single_car(self, selected_cav_base, ego_pose):
        """
        Project the lidar and bbx to ego space first, and then do clipping.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
        ego_pose : list
            The ego vehicle lidar pose under world coordinate.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = {}

        # calculate the transformation matrix
        transformation_matrix = selected_cav_base["params"]["transformation_matrix"]

        # retrieve objects under ego coordinates
        object_bbx_center, object_bbx_mask, object_ids = self.post_processor.generate_object_center([selected_cav_base], ego_pose)

        # filter lidar
        lidar_np = selected_cav_base["lidar_np"]
        lidar_np = shuffle_points(lidar_np)
        # remove points that hit itself
        lidar_np = mask_ego_points(lidar_np)
        # project the lidar to ego space
        if self.proj_first:
            lidar_np[:, :3] = box_utils.project_points_by_matrix_torch(lidar_np[:, :3], transformation_matrix)
        lidar_np = mask_points_by_range(lidar_np, self.params["preprocess"]["cav_lidar_range"])
        processed_lidar = self.pre_processor.preprocess(lidar_np)

        # velocity
        velocity = selected_cav_base["params"]["ego_speed"]
        # normalize veloccity by average speed 30 km/h
        velocity = velocity / 30

        selected_cav_processed.update(
            {
                "object_bbx_center": object_bbx_center[object_bbx_mask == 1],
                "object_ids": object_ids,
                "projected_lidar": lidar_np,
                "processed_features": processed_lidar,
                "velocity": velocity,
            }
        )

        return selected_cav_processed

    @staticmethod
    def merge_features_to_dict(processed_feature_list):
        """
        Merge the preprocessed features from different cavs to the same
        dictionary.

        Parameters
        ----------
        processed_feature_list : list
            A list of dictionary containing all processed features from
            different cavs.

        Returns
        -------
        merged_feature_dict: dict
            key: feature names, value: list of features.
        """

        merged_feature_dict = OrderedDict()

        for i in range(len(processed_feature_list)):
            for feature_name, feature in processed_feature_list[i].items():
                if feature_name not in merged_feature_dict:
                    merged_feature_dict[feature_name] = []
                if isinstance(feature, list):
                    merged_feature_dict[feature_name] += feature
                else:
                    merged_feature_dict[feature_name].append(feature)

        return merged_feature_dict

    def collate_batch_train(self, batch):
        # Intermediate fusion is different the other two
        output_dict = {"ego": {}}

        object_bbx_center = []
        object_bbx_mask = []
        object_ids = []
        processed_lidar_list = []
        # used to record different scenario
        record_len = []
        label_dict_list = []

        # used for PriorEncoding for models
        velocity = []
        time_delay = []
        infra = []

        # pairwise transformation matrix
        pairwise_t_matrix_list = []

        # used for correcting the spatial transformation between delayed timestamp
        # and current timestamp
        spatial_correction_matrix_list = []

        if self.visualize:
            origin_lidar = []

        for i in range(len(batch)):
            ego_dict = batch[i]["ego"]
            object_bbx_center.append(ego_dict["object_bbx_center"])
            object_bbx_mask.append(ego_dict["object_bbx_mask"])
            object_ids.append(ego_dict["object_ids"])

            processed_lidar_list.append(ego_dict["processed_lidar"])
            record_len.append(ego_dict["cav_num"])
            label_dict_list.append(ego_dict["label_dict"])
            pairwise_t_matrix_list.append(ego_dict["pairwise_t_matrix"])

            velocity.append(ego_dict["velocity"])
            time_delay.append(ego_dict["time_delay"])
            infra.append(ego_dict["infra"])
            spatial_correction_matrix_list.append(ego_dict["spatial_correction_matrix"])

            if self.visualize:
                origin_lidar.append(ego_dict["origin_lidar"])
        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        # example: {'voxel_features':[np.array([1,2,3]]),
        # np.array([3,5,6]), ...]}
        merged_feature_dict = self.merge_features_to_dict(processed_lidar_list)
        processed_lidar_torch_dict = self.pre_processor.collate_batch(merged_feature_dict)
        # [2, 3, 4, ..., M], M <= max_cav
        record_len = torch.from_numpy(np.array(record_len, dtype=int))
        label_torch_dict = self.post_processor.collate_batch(label_dict_list)

        # (B, max_cav)
        velocity = torch.from_numpy(np.array(velocity))
        time_delay = torch.from_numpy(np.array(time_delay))
        infra = torch.from_numpy(np.array(infra))
        spatial_correction_matrix_list = torch.from_numpy(np.array(spatial_correction_matrix_list))
        # (B, max_cav, 3)
        prior_encoding = torch.stack([velocity, time_delay, infra], dim=-1).float()
        # (B, max_cav)
        pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))

        # object id is only used during inference, where batch size is 1.
        # so here we only get the first element.
        output_dict["ego"].update(
            {
                "object_bbx_center": object_bbx_center,
                "object_bbx_mask": object_bbx_mask,
                "processed_lidar": processed_lidar_torch_dict,
                "record_len": record_len,
                "label_dict": label_torch_dict,
                "object_ids": object_ids[0],
                "prior_encoding": prior_encoding,
                "spatial_correction_matrix": spatial_correction_matrix_list,
                "pairwise_t_matrix": pairwise_t_matrix,
            }
        )

        if self.visualize:
            origin_lidar = np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict["ego"].update({"origin_lidar": origin_lidar})

        return output_dict

    def collate_batch_test(self, batch):
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        output_dict = self.collate_batch_train(batch)

        # check if anchor box in the batch
        if batch[0]["ego"]["anchor_box"] is not None:
            output_dict["ego"].update({"anchor_box": torch.from_numpy(np.array(batch[0]["ego"]["anchor_box"]))})

        # save the transformation matrix (4, 4) to ego vehicle
        transformation_matrix_torch = torch.from_numpy(np.identity(4)).float()
        output_dict["ego"].update({"transformation_matrix": transformation_matrix_torch})

        return output_dict

    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        """
        pred_box_tensor, pred_score = self.post_processor.post_process(data_dict, output_dict)
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        return pred_box_tensor, pred_score, gt_box_tensor

    def get_pairwise_transformation(self, base_data_dict, max_cav):
        """
        Get pair-wise transformation matrix accross different agents.

        Parameters
        ----------
        base_data_dict : dict
            Key : cav id, item: transformation matrix to ego, lidar points.

        max_cav : int
            The maximum number of cav, default 5

        Return
        ------
        pairwise_t_matrix : np.array
            The pairwise transformation matrix across each cav.
            shape: (L, L, 4, 4)
        """
        pairwise_t_matrix = np.zeros((max_cav, max_cav, 4, 4))

        if self.proj_first:
            # if lidar projected to ego first, then the pairwise matrix
            # becomes identity
            pairwise_t_matrix[:, :] = np.identity(4)
        else:
            t_list = []

            # save all transformation matrix in a list in order first.
            for cav_id, cav_content in base_data_dict.items():
                t_list.append(cav_content["params"]["transformation_matrix"])

            for i in range(len(t_list)):
                for j in range(len(t_list)):
                    # identity matrix to self
                    if i == j:
                        t_matrix = np.eye(4)
                        pairwise_t_matrix[i, j] = t_matrix
                        continue
                    # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                    t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                    pairwise_t_matrix[i, j] = t_matrix

        return pairwise_t_matrix
