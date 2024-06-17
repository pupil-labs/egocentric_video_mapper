import cv2 as cv
import numpy as np
import kornia
import torch
from torchvision import transforms
from abc import ABC, abstractmethod


class ImageMatcher(ABC):
    @abstractmethod
    def get_correspondences(self, image1, image2):
        return

    def _get_image_patch(self, image, patch_corners):
        x_min, y_min = min(patch_corners[:, 0]), min(patch_corners[:, 1])
        x_max, y_max = max(patch_corners[:, 0]), max(patch_corners[:, 1])
        image_patch = image[y_min:y_max, x_min:x_max, :]
        return image_patch


class LOFTRImageMatcher(ImageMatcher):
    def __init__(self, location, gpu_num=None):
        if gpu_num is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(
                f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu"
            )
        self.image_matcher = kornia.feature.LoFTR(pretrained=location).to(self.device)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def get_correspondences(
        self, src_image, dst_image, src_patch_corners=None
    ):  # add dst corner,
        dst_tensor, dst_scaled2original = self._preprocess_image(dst_image)
        src_image = (
            self._get_image_patch(src_image, src_patch_corners)
            if src_patch_corners is not None
            else src_image.copy()
        )
        src_tensor, src_scaled2original = self._preprocess_image(src_image)
        input_dict = {"image0": src_tensor, "image1": dst_tensor}
        for k in input_dict.keys():
            input_dict[k] = input_dict[k].to(self.device)
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
        return correspondences

    def _preprocess_image(self, image):
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        scaled_image = cv.resize(
            image, (round(540 * image.shape[1] / image.shape[0]), 540)
        )
        ratio_scaled2image = (
            image.shape[1] / scaled_image.shape[1],
            image.shape[0] / scaled_image.shape[0],
        )
        scaled_image = self.transform(scaled_image)
        scaled_image = torch.unsqueeze(scaled_image, dim=0)
        return scaled_image, ratio_scaled2image

    def _get_image_patch(self, image, patch_corners):
        x_min, y_min = min(patch_corners[:, 0]), min(patch_corners[:, 1])
        x_max, y_max = max(patch_corners[:, 0]), max(patch_corners[:, 1])
        image_patch = image[y_min:y_max, x_min:x_max, :]
        return image_patch

    def _rescale_correspondences(self, correspondences, src_ratio, dst_ratio):
        correspondences["keypoints0"] = correspondences["keypoints0"] * src_ratio
        correspondences["keypoints1"] = correspondences["keypoints1"] * dst_ratio
        return correspondences


class DISK_LightGlueImageMatcher(ImageMatcher):
    def __init__(self, num_features, gpu_num=None):
        self.num_features = num_features
        if gpu_num is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(
                f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu"
            )
        self.feature_extractor = kornia.feature.DISK.from_pretrained("depth").to(
            self.device
        )
        self.feature_matcher = kornia.feature.LightGlue("disk").eval().to(self.device)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def get_correspondences(
        self, src_image, dst_image, src_patch_corners=None
    ):  # add dst corner,
        dst_tensor = self._preprocess_image(dst_image.copy())  # 1xcxhxw
        src_image = (
            self._get_image_patch(src_image, src_patch_corners)
            if src_patch_corners is not None
            else src_image.copy()
        )
        src_tensor = self._preprocess_image(src_image)

        with torch.inference_mode():
            src_tensor = src_tensor.to(self.device)
            dst_tensor = dst_tensor.to(self.device)
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
                .to(self.device),
            }
            image_dst = {
                "keypoints": features1.keypoints[None],
                "descriptors": features1.descriptors[None],
                "image_size": torch.tensor(dst_tensor.shape[-2:][::-1])
                .view(1, 2)
                .to(self.device),
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
        if src_patch_corners is not None:
            correspondences["keypoints0"] = (
                correspondences["keypoints0"] + src_patch_corners[0]
            )
        return correspondences

    def _preprocess_image(self, image):
        # image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # scaled_image = cv.resize(image, (round(540*image.shape[1]/image.shape[0]), 540))
        # ratio_scaled2image = (image.shape[1]/scaled_image.shape[1] ,
        #                     image.shape[0]/scaled_image.shape[0])
        scaled_image = self.transform(image)
        scaled_image = torch.unsqueeze(scaled_image, dim=0)
        return scaled_image  # , ratio_scaled2image

    def _process_lightglue_output(self, out):
        scores = out["scores"][0].cpu().numpy()
        idx = out["matches"][0].cpu().numpy()


class ImageMatcherFactory:
    def get_matcher(self, image_matcher, image_matcher_parameters):
        if image_matcher.upper() == "LOFTR":
            return LOFTRImageMatcher(**image_matcher_parameters)
        # does all the ifs, switchcases and param settings specific to each matcher to instantiate the desired image matcher
        if image_matcher.upper() == "DISK_LIGHTGLUE":
            return DISK_LightGlueImageMatcher(**image_matcher_parameters)
