import os
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path

import cv2 as cv
import kornia
import numpy as np
import torch
from torchvision import transforms

from pupil_labs.egocentric_video_mapper.efficient_loftr.src.loftr import LoFTR as eLoFTR
from pupil_labs.egocentric_video_mapper.efficient_loftr.src.loftr import (
    full_default_cfg,
    opt_default_cfg,
    reparameter,
)


class ImageMatcher(ABC):
    def __init__(self, gpu_num=None):
        if gpu_num is None:
            self._device = torch.device("cpu")
        else:
            self._device = torch.device(
                f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu"
            )
        self._transform = transforms.Compose([transforms.ToTensor()])

    @abstractmethod
    def get_correspondences(
        self, src_image, dst_image, src_patch_corners=None, dst_patch_corners=None
    ):
        """Get correspondences between source and destination images. If patch corners are provided, the correspondences are only calculated in the region delimited by them.

        Args:
            src_image (ndarray): Source image.
            dst_image (ndarray): Destination image.
            src_patch_corners (ndarray, optional): Array, of shape (4,2), containing the corner coordinates of the region of interest in the image for match searching. If None is given, the whole image is used in the match search. Defaults to None.
            dst_patch_corners (ndarray, optional):  Array, of shape (4,2), containing the corner coordinates of the region of interest in the image for match searching. If None is given, the whole image is used in the match search. Defaults to None.

        """
        return

    def _get_image_patch(self, image, patch_corners):
        x_min, y_min = min(patch_corners[:, 0]), min(patch_corners[:, 1])
        x_max, y_max = max(patch_corners[:, 0]), max(patch_corners[:, 1])
        image_patch = image[y_min:y_max, x_min:x_max, :]
        return image_patch

    def _rescale_correspondences(self, correspondences, src_ratio, dst_ratio):
        correspondences["keypoints0"] = correspondences["keypoints0"] * src_ratio
        correspondences["keypoints1"] = correspondences["keypoints1"] * dst_ratio
        return correspondences

    def _preprocess_image(self, image, patch_corners, gray_scale=True):
        image = (
            self._get_image_patch(image, patch_corners)
            if patch_corners is not None
            else image.copy()
        )

        if gray_scale:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        scaled_image = cv.resize(
            image, (image.shape[1] // 60 * 32, image.shape[0] // 60 * 32)
        )
        ratio_scaled2image = (
            image.shape[1] / scaled_image.shape[1],
            image.shape[0] / scaled_image.shape[0],
        )
        scaled_image = self._transform(scaled_image)
        scaled_image = torch.unsqueeze(scaled_image, dim=0)
        return scaled_image, ratio_scaled2image


class LOFTRImageMatcher(ImageMatcher):
    def __init__(self, location, gpu_num=None):
        """
        This class is a wrapper for the LOFTR model from Kornia library. It is used to find correspondences between two images. To read about the model, please refer to https://zju3dv.github.io/loftr/
        Args:
            location (str):Pretrained weights of LOFTR in ['indoor', 'outdoor'].'outdoor' is trained on MegaDepth dataset and 'indoor' on ScanNet dataset.
            gpu_num (int, optional): The GPU device number to use, if None it uses CPU. Defaults to None.
        """
        super().__init__(gpu_num)
        self.image_matcher = kornia.feature.LoFTR(pretrained=location).to(self._device)

    def get_correspondences(
        self, src_image, dst_image, src_patch_corners=None, dst_patch_corners=None
    ):
        src_tensor, src_scaled2original = self._preprocess_image(
            src_image, src_patch_corners
        )
        dst_tensor, dst_scaled2original = self._preprocess_image(
            dst_image, dst_patch_corners
        )

        input_dict = {
            "image0": src_tensor.to(self._device),
            "image1": dst_tensor.to(self._device),
        }

        with torch.inference_mode():
            correspondences = self.image_matcher(input_dict)

        for k in correspondences.keys():
            correspondences[k] = correspondences[k].cpu().numpy()

        correspondences = self._rescale_correspondences(
            correspondences, src_scaled2original, dst_scaled2original
        )

        if src_patch_corners is not None:
            correspondences["keypoints0"] = (
                correspondences["keypoints0"] + src_patch_corners[0]
            )
        if dst_patch_corners is not None:
            correspondences["keypoints1"] = (
                correspondences["keypoints1"] + dst_patch_corners[0]
            )

        return correspondences


class DISKLightGlueImageMatcher(ImageMatcher):
    def __init__(self, num_features=None, gpu_num=None):
        """
        This class is a wrapper for the LightGlue algorithm with DISK features from Kornia library. It is used to find correspondences between two images. To read about both algorithms, please refer to DISK: Learning local features with policy gradient (https://github.com/cvlab-epfl/disk) and to LightGlue: Local Feature Matching at Light Speed (https://github.com/cvg/LightGlue?tab=readme-ov-file)
        Args:
            num_features (int): The maximum number of keypoints in DISK to detect. If None, all keypoints are returned.
            gpu_num (int, optional): The GPU device number to use, if None it uses CPU. Defaults to None
        """
        super().__init__(gpu_num)
        self.num_features = num_features

        self.feature_extractor = kornia.feature.DISK.from_pretrained("depth").to(
            self._device
        )
        self.feature_matcher = kornia.feature.LightGlue("disk").eval().to(self._device)

    def get_correspondences(
        self, src_image, dst_image, src_patch_corners=None, dst_patch_corners=None
    ):
        src_tensor, src_scaled2original = self._preprocess_image(
            src_image, src_patch_corners, gray_scale=False
        )

        dst_tensor, dst_scaled2original = self._preprocess_image(
            dst_image, dst_patch_corners, gray_scale=False
        )  # 1xCxHxW

        with torch.inference_mode():
            src_tensor = src_tensor.to(self._device)
            dst_tensor = dst_tensor.to(self._device)

            features0 = self.feature_extractor(
                src_tensor, self.num_features, pad_if_not_divisible=True
            )[0]
            features1 = self.feature_extractor(
                dst_tensor, self.num_features, pad_if_not_divisible=True
            )[0]

            image_src = {
                "keypoints": features0.keypoints[None],
                "descriptors": features0.descriptors[None],
                "image_size": torch.tensor(src_tensor.shape[-2:][::-1])
                .view(1, 2)
                .to(self._device),
            }
            image_dst = {
                "keypoints": features1.keypoints[None],
                "descriptors": features1.descriptors[None],
                "image_size": torch.tensor(dst_tensor.shape[-2:][::-1])
                .view(1, 2)
                .to(self._device),
            }

            out = self.feature_matcher({"image0": image_src, "image1": image_dst})

        scores = out["scores"][0].cpu().numpy()
        idxs = out["matches"][0].cpu().numpy()
        kp0 = features0.keypoints.cpu().numpy()
        kp1 = features1.keypoints.cpu().numpy()
        correspondences = {
            "keypoints0": kp0[idxs[:, 0]],
            "keypoints1": kp1[idxs[:, 1]],
            "confidence": scores,
        }

        correspondences = self._rescale_correspondences(
            correspondences, src_scaled2original, dst_scaled2original
        )

        if src_patch_corners is not None:
            correspondences["keypoints0"] = (
                correspondences["keypoints0"] + src_patch_corners[0]
            )

        if dst_patch_corners is not None:
            correspondences["keypoints1"] = (
                correspondences["keypoints1"] + dst_patch_corners[0]
            )

        return correspondences


class DeDoDeLightGlueImageMatcher(ImageMatcher):
    def __init__(self, num_features=10000, gpu_num=None):
        """
        This class is a wrapper for the LightGlue algorithm with DeDoDe features from Kornia library. It is used to find correspondences between two images. To read about both algorithms, please refer to DeDoDe: Detect, Don't Describe — Describe, Don't Detect for Local Feature Matching (https://github.com/Parskatt/DeDoDe) and to LightGlue: Local Feature Matching at Light Speed (https://github.com/cvg/LightGlue?tab=readme-ov-file)
        Args:
            num_features (int): The maximum number of keypoints in DeDoDe to detect. Defaults to 10000.
            gpu_num (int, optional): The GPU device number to use, if None it uses CPU. Defaults to None.
        """
        super().__init__(gpu_num)
        self.num_features = num_features

        self.feature_extractor = kornia.feature.DeDoDe.from_pretrained(
            detector_weights="L-upright", descriptor_weights="B-upright"
        ).to(self._device)
        self.feature_matcher = (
            kornia.feature.LightGlue("dedodeb").eval().to(self._device)
        )

    def get_correspondences(
        self, src_image, dst_image, src_patch_corners=None, dst_patch_corners=None
    ):
        src_tensor, src_scaled2original = self._preprocess_image(
            src_image, src_patch_corners, gray_scale=False
        )
        dst_tensor, dst_scaled2original = self._preprocess_image(
            dst_image, dst_patch_corners, gray_scale=False
        )  # 1xcxhxw

        with torch.inference_mode():
            src_tensor = src_tensor.to(self._device)
            dst_tensor = dst_tensor.to(self._device)

            keypoints0, _, descriptors0 = self.feature_extractor(
                src_tensor, self.num_features
            )
            keypoints1, _, descriptors1 = self.feature_extractor(
                dst_tensor, self.num_features
            )

            image_src = {
                "keypoints": keypoints0,
                "descriptors": descriptors0,
                "image_size": torch.tensor(src_tensor.shape[-2:][::-1])
                .view(1, 2)
                .to(self._device),
            }
            image_dst = {
                "keypoints": keypoints1,
                "descriptors": descriptors1,
                "image_size": torch.tensor(dst_tensor.shape[-2:][::-1])
                .view(1, 2)
                .to(self._device),
            }

            out = self.feature_matcher({"image0": image_src, "image1": image_dst})

        scores = out["scores"][0].cpu().numpy()
        idxs = out["matches"][0].cpu().numpy()
        kp0 = keypoints0.cpu().numpy().reshape(-1, 2)
        kp1 = keypoints1.cpu().numpy().reshape(-1, 2)
        correspondences = {
            "keypoints0": kp0[idxs[:, 0]],
            "keypoints1": kp1[idxs[:, 1]],
            "confidence": scores,
        }

        correspondences = self._rescale_correspondences(
            correspondences, src_scaled2original, dst_scaled2original
        )

        if src_patch_corners is not None:
            correspondences["keypoints0"] = (
                correspondences["keypoints0"] + src_patch_corners[0]
            )

        if dst_patch_corners is not None:
            correspondences["keypoints1"] = (
                correspondences["keypoints1"] + dst_patch_corners[0]
            )

        return correspondences


class EfficientLoFTRImageMatcher(ImageMatcher):
    def __init__(self, model_type="opt", gpu_num=None):
        """
        This class is a wrapper for the Efficient LoFTR model. To read about the model, please refer to https://zju3dv.github.io/loftr/
        Args:
            model_type (str, optional): Model type in ['full', 'opt']. Use 'full' for best quality, 'opt' for best efficiency. Defaults to "opt".
            gpu_num (int, optional): The GPU device number to use, if None it uses CPU. Defaults to None.
        """
        super().__init__(gpu_num)
        if model_type == "full":
            _default_cfg = deepcopy(full_default_cfg)
        elif model_type == "opt":
            _default_cfg = deepcopy(opt_default_cfg)

        self.image_matcher = eLoFTR(config=_default_cfg)
        try:
            self.image_matcher.load_state_dict(
                torch.load(
                    Path(__file__).parent
                    / "efficient_loftr/weights/eloftr_outdoor.ckpt",
                    map_location=self._device,
                )["state_dict"]
            )
        # If the weights are not found in the current directory, it tries to find them in user-set environment variable
        except:
            try:
                EFFICIENT_LOFTR_WEIGHTS = os.environ.get("efficentloftr_weights")
                self.image_matcher.load_state_dict(
                    torch.load(EFFICIENT_LOFTR_WEIGHTS, map_location=self._device)[
                        "state_dict"
                    ]
                )
            except:
                print("Explore")
                self.image_matcher.load_state_dict(
                    torch.load(
                        "/content/egocentric_video_mapper/src/pupil_labs/egocentric_video_mapper/efficient_loftr/weights/eloftr_outdoor.ckpt",
                        map_location=self._device,
                    )["state_dict"]
                )
        self.image_matcher = reparameter(self.image_matcher)
        self.image_matcher = self.image_matcher.eval().to(self._device)

    def get_correspondences(
        self, src_image, dst_image, src_patch_corners=None, dst_patch_corners=None
    ):
        src_tensor, src_scaled2original = self._preprocess_image(
            src_image, src_patch_corners
        )
        dst_tensor, dst_scaled2original = self._preprocess_image(
            dst_image, dst_patch_corners
        )

        batch = {
            "image0": src_tensor.to(self._device),
            "image1": dst_tensor.to(self._device),
        }

        with torch.inference_mode():
            self.image_matcher(batch)

        correspondences = {
            "keypoints0": batch["mkpts0_f"].cpu().numpy(),
            "keypoints1": batch["mkpts1_f"].cpu().numpy(),
            "confidence": batch["mconf"].cpu().numpy(),
        }
        correspondences = self._rescale_correspondences(
            correspondences, src_scaled2original, dst_scaled2original
        )
        if src_patch_corners is not None:
            correspondences["keypoints0"] = (
                correspondences["keypoints0"] + src_patch_corners[0]
            )

        if dst_patch_corners is not None:
            correspondences["keypoints1"] = (
                correspondences["keypoints1"] + dst_patch_corners[0]
            )

        return correspondences


def get_matcher(image_matcher, image_matcher_parameters):
    if image_matcher.lower() == "loftr":
        return LOFTRImageMatcher(**image_matcher_parameters)
    elif image_matcher.lower() == "disk_lightglue":
        return DISKLightGlueImageMatcher(**image_matcher_parameters)
    elif image_matcher.lower() == "efficient_loftr":
        return EfficientLoFTRImageMatcher(**image_matcher_parameters)
    elif image_matcher.lower() == "dedode_lightglue":
        return DeDoDeLightGlueImageMatcher(**image_matcher_parameters)
    else:
        raise ValueError("Invalid image matcher", image_matcher)
