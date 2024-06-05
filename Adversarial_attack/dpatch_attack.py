# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements a variation of the adversarial patch attack `DPatch` for object detectors.
It follows Lee & Kolter (2019) in using sign gradients with expectations over transformations.
The particular transformations supported in this implementation are cropping, rotations by multiples of 90 degrees,
and changes in the brightness of the image.

| Paper link (original DPatch): https://arxiv.org/abs/1806.02299v4
| Paper link (physical-world patch from Lee & Kolter): https://arxiv.org/abs/1906.11897
"""
import logging
import math
import os
import sys
import random
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import torch
import imutils
import numpy as np
from tqdm.auto import trange
import matplotlib.pyplot as plt
from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.object_detection.object_detector import ObjectDetectorMixin
from art import config
import cv2
from torch import nn
from Adversarial_detection.object_extractor_detector import Prototypes,oe_detector_demo_super_store,Pointrend_instance_segmentation
from Adversarial_detection.scene_processing_detector import style_transfer_helper
from style_transfer import style_img, get_style_transfer_models


if TYPE_CHECKING:
    from art.utils import OBJECT_DETECTOR_TYPE

logger = logging.getLogger(__name__)


class RobustDPatch(EvasionAttack):
    """
    Implementation of a particular variation of the DPatch attack.
    It follows Lee & Kolter (2019) in using sign gradients with expectations over transformations.
    The particular transformations supported in this implementation are cropping, rotations by multiples of 90 degrees,
    and changes in the brightness of the image.

    | Paper link (original DPatch): https://arxiv.org/abs/1806.02299v4
    | Paper link (physical-world patch from Lee & Kolter): https://arxiv.org/abs/1906.11897
    """

    attack_params = EvasionAttack.attack_params + [
        "patch_shape",
        "learning_rate",
        "max_iter",
        "batch_size",
        "patch_location",
        "crop_range",
        "brightness_range",
        "rotation_weights",
        "blur_scale",
        "noise_scale",
        "adaptive_attack",
        "style_transfer_model_path",
        "sample_size",
        "targeted",
        "verbose",
        "patch_save_dir"
    ]

    _estimator_requirements = (
    BaseEstimator, LossGradientsMixin, ObjectDetectorMixin)

    def __init__(
            self,
            estimator: "OBJECT_DETECTOR_TYPE",
            patch_shape: Tuple[int, int, int] = (40, 40, 3),
            patch_location: Tuple[int, int] = (0, 0),
            crop_range: Tuple[int, int] = (0, 0),
            brightness_range: Tuple[float, float] = (1.0, 1.0),
            rotation_weights: Union[Tuple[float, float, float, float], Tuple[
                int, int, int, int]] = (1, 0, 0, 0),
            blur_scale: int = 0,
            noise_scale: float = 0.25,
            adaptive_attack: bool = False,
            adaptive_type :str = "",
            prototypes_path:str = "",
            style_transfer_model_path:str = "",
            sample_size: int = 1,
            learning_rate: float = 5.0,
            max_iter: int = 500,
            batch_size: int = 16,
            targeted: bool = False,
            verbose: bool = True,
            attack_losses: Tuple[str, ...] = (
                    "loss_classifier",
                    "loss_box_reg",
                    "loss_objectness",
                    "loss_rpn_box_reg",
            ),
            patch_save_dir=None):
        """
        Create an instance of the :class:`.RobustDPatch`.

        :param estimator: A trained object detector.
        :param patch_shape: The shape of the adversarial patch as a tuple of shape (height, width, nb_channels).
        :param patch_location: The location of the adversarial patch as a tuple of shape (upper left x, upper left y).
        :param crop_range: By how much the images may be cropped as a tuple of shape (height, width).
        :param brightness_range: Range for randomly adjusting the brightness of the image.
        :param rotation_weights: Sampling weights for random image rotations by (0, 90, 180, 270) degrees clockwise.
        :param: blur_range: Range for randomly adjusting the blurness of the image.
        :param: random_noise_range: Range for randomly adjusting the random noise of the image.
        :param: adaptive_attack: Use as an adaptive attack.
        :param: style_transfer_model_path: model for style transfer technique.
        :param sample_size: Number of samples to be used in expectations over transformation.
        :param learning_rate: The learning rate of the optimization.
        :param max_iter: The number of optimization steps.
        :param batch_size: The size of the training batch.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).
        :param verbose: Show progress bars.
        """

        super().__init__(estimator=estimator)

        self.attack_losses = attack_losses
        self.patch_shape = patch_shape
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.patch_path = patch_save_dir
        if self.estimator.clip_values is None:
            self._patch = np.zeros(shape=patch_shape,
                                   dtype=config.ART_NUMPY_DTYPE)
        else:
            self._patch = (
                    np.random.randint(0, 255, size=patch_shape)
                    / 255
                    * (self.estimator.clip_values[1] -
                       self.estimator.clip_values[0])
                    + self.estimator.clip_values[0]
            ).astype(config.ART_NUMPY_DTYPE)
        self.verbose = verbose
        self.patch_location = patch_location
        self.crop_range = crop_range
        self.brightness_range = brightness_range
        self.rotation_weights = rotation_weights
        self.blur_scale = blur_scale
        self.noise_scale = noise_scale
        self.sample_size = sample_size
        self._targeted = targeted
        self.loss_history = []
        self.open_reduce_on_plateau = 15
        self.best_loss = 100
        self.adaptive_attack = adaptive_attack
        self.model_util_path = "../Models/Util_models"
        if self.adaptive_attack:
            sth = style_transfer_helper()
            self.style_transfer_model = sth.get_style_transfer_models(style_transfer_model_path)
            self.pointrend_model = Pointrend_instance_segmentation(self.model_util_path)
            self.prototypes = Prototypes(prototypes_path,self.model_util_path)
            self.prototypes.process_prototypes()
            self.adaptive_type = adaptive_type
        self._check_params()

    def generate(self, x: np.ndarray,
                 y: Optional[List[Dict[str, np.ndarray]]] = None,
                 **kwargs) -> np.ndarray:
        """
        Generate RobustDPatch.

        :param x: Sample images.
        :param y: Target labels for object detector.
        :return: Adversarial patch.
        """
        channel_index = 1 if self.estimator.channels_first else x.ndim - 1
        if x.shape[channel_index] != self.patch_shape[channel_index - 1]:
            raise ValueError(
                "The color channel index of the images and the patch have to be identical.")
        if y is None and self.targeted:
            raise ValueError(
                "The targeted version of RobustDPatch attack requires target labels provided to `y`.")
        if y is not None and not self.targeted:
            raise ValueError(
                "The RobustDPatch attack does not use target labels.")
        if x.ndim != 4:  # pragma: no cover
            raise ValueError(
                "The adversarial patch can only be applied to images.")

        # Check whether patch fits into the cropped images:
        if self.estimator.channels_first:
            image_height, image_width = x.shape[2:4]
        else:
            image_height, image_width = x.shape[1:3]

        if not self.estimator.native_label_is_pytorch_format and y is not None:
            from art.estimators.object_detection.utils import convert_tf_to_pt

            y = convert_tf_to_pt(y=y, height=x.shape[1], width=x.shape[2])

        if y is not None:
            for i_image in range(x.shape[0]):
                y_i = y[i_image]["boxes"]
                for i_box in range(y_i.shape[0]):
                    x_1, y_1, x_2, y_2 = y_i[i_box]
                    if (  # pragma: no cover
                            x_1 < self.crop_range[1]
                            or y_1 < self.crop_range[0]
                            or x_2 > image_width - self.crop_range[1] + 1
                            or y_2 > image_height - self.crop_range[0] + 1
                    ):
                        raise ValueError(
                            "Cropping is intersecting with at least one box, reduce `crop_range`.")

        if (  # pragma: no cover
                self.patch_location[0] + self.patch_shape[0] > image_height -
                self.crop_range[0]
                or self.patch_location[1] + self.patch_shape[1] > image_width -
                self.crop_range[1]
        ):
            raise ValueError(
                "The patch (partially) lies outside the cropped image.")

        for i_step in trange(self.max_iter, desc="RobustDPatch iteration",
                             disable=not self.verbose):
            if i_step == 0 or (i_step + 1) % 100 == 0:
                logger.info("Training Step: %i", i_step + 1)

            num_batches = math.ceil(x.shape[0] / self.batch_size)
            patch_gradients_old = np.zeros_like(self._patch)
            loss = 0
            for e_step in range(self.sample_size):

                if e_step == 0 or (e_step + 1) % 100 == 0:
                    logger.info("EOT Step: %i", e_step + 1)

                for i_batch in range(num_batches):
                    i_batch_start = i_batch * self.batch_size
                    i_batch_end = min((i_batch + 1) * self.batch_size,
                                      x.shape[0])

                    if y is None:
                        y_batch = y
                    else:
                        y_batch = y[i_batch_start:i_batch_end]

                    # Sample and apply the random transformations:
                    patched_images, patch_target, transforms = self._augment_images_with_patch(
                        x[i_batch_start:i_batch_end], y_batch, self._patch,
                        channels_first=self.estimator.channels_first,step = e_step
                    )

                    loss +=self.get_loss(patched_images,patch_target)
                    gradients = self.estimator.loss_gradient(
                        x=patched_images,
                        y=patch_target,
                        standardise_output=True,
                        #adaptive_attack = self.adaptive_attack
                    )

                    gradients = self._untransform_gradients(
                        gradients, transforms,
                        channels_first=self.estimator.channels_first,
                        step=e_step
                    )

                    patch_gradients = patch_gradients_old + np.sum(gradients,
                                                                   axis=0)
                    logger.debug(
                        "Gradient percentage diff: %f)",
                        np.mean(np.sign(patch_gradients) != np.sign(
                            patch_gradients_old)),
                    )

                    patch_gradients_old = patch_gradients

            self.loss_history.append(loss)
            if loss < self.best_loss:
                self.best_loss = loss
                self.save_patch(self._patch, (
                    self._patch.shape[0] / 100,
                    self._patch.shape[1] / 100),
                                output_path=self.patch_path,type="best")
                self.save_patch(self._patch, (10, 7),
                                output_path=self.patch_path,type="best")
            if self.reduce_on_plateau():
                self.learning_rate = self.learning_rate/2
            self._patch = self._patch + np.sign(patch_gradients) * (
                        1 - 2 * int(self.targeted)) * self.learning_rate
            # self._patch = self._patch - (np.sign(patch_gradients) * (int(
            #     self.targeted)) * self.learning_rate)

            if self.estimator.clip_values is not None:
                self._patch = np.clip(
                    self._patch,
                    a_min=self.estimator.clip_values[0],
                    a_max=self.estimator.clip_values[1],
                )

            if i_step %10 == 0:
                self.plot_losses(self.patch_path)
                if i_step % 100 == 0:
                    self.save_patch(self._patch, (
                        self._patch.shape[0] / 100,
                        self._patch.shape[1] / 100),
                               output_path=self.patch_path)
                    self.save_patch(self._patch, (10, 7), output_path=self.patch_path)

        return self._patch

    def save_patch(self,patch, patch_size, output_path="output",type = "last"):
        fig = plt.figure(figsize=(patch_size))
        plt.axis("off")
        plt.imshow(patch.astype(np.uint8), interpolation="nearest")
        # plt.show()
        size = ""
        if patch_size[0] > 2:
            size = "big"
        plt.savefig(f'{output_path}/patch_{type}_{size}.png', dpi=fig.dpi)
        with open(f'{output_path}/{type}_patch.npy', 'wb') as f:
            np.save(f, patch)

    def get_loss(self,x,y):
        for i in range(x.shape[0]):

            x_i = x[[i]]
            y_i = [y[i]]

            output, inputs_t, image_tensor_list_grad = self.estimator._get_losses(
                x=x_i, y=y_i)

            # Compute the gradient and return
            loss = None
            for loss_name in self.attack_losses:
                if loss is None:
                    loss = output[loss_name]
                else:
                    loss = loss + output[loss_name]
            if self.adaptive_attack and (self.adaptive_type=="Object extraction" or self.adaptive_type=="Ensemble"):
                matching_points_probability_vector = self.compute_sift_loss(
                    x_i,y_i)
                if matching_points_probability_vector != None and len(matching_points_probability_vector) > 0:
                    sift_loss = self.compute_matching_point_loss(matching_points_probability_vector,y_i)
                    loss +=sift_loss

        return float(loss.cpu().detach().numpy())

    def reduce_on_plateau(self):
        if len(self.loss_history)>self.open_reduce_on_plateau:
            if self.loss_history[-1]>self.loss_history[-15]:
                if self.loss_history[-2]>self.loss_history[-15]:
                    print("Reduce on plateau")
                    self.open_reduce_on_plateau = len(self.loss_history)+20
                    return True
        return False

    def get_dict(self):
        return {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0,
            10: 0,
            11: 0,
            12: 0,
            13: 0,
            14: 0,
            15: 0,
            16: 0,
            17: 0,
            18: 0,
            19: 0,
            20: 0
        }
    def get_items_dict(self):
        return {
            'Agnesi Polenta':1,
            'Almond Milk':2,
            'Snyders':3,
            'Calvin Klein':4,
            'Dr Pepper':5,
            'Flour':6,
            'Groats':7,
            'Jack Daniels': 8,
            'Nespresso': 9,
            'Oil':10,
            'Paco Rabanne':11,
            'Pixel4':12,
            'Samsung_s20':13,
            'Greek Olives':14,
            'Curry Spice':15,
            'Chablis Wine':16,
            'Lindor':17,
            'Piling Sabon':18,
            'Tea':19,
            'Versace':20,
        }

    def compute_sift_loss(self,frames,target):
        frame_oe_detector = np.squeeze(frames, axis=0)
        frame_oe_detector = np.copy(frame_oe_detector)
        frame_oe_detector = np.uint8(frame_oe_detector)
        frame_oe_detector = cv2.cvtColor(frame_oe_detector, cv2.COLOR_BGR2RGB)
        bbox_list = torch.tensor(target[0]['boxes'][0])
        object_extraction_pred = oe_detector_demo_super_store(
                self.prototypes, self.pointrend_model,
                frame_oe_detector, bbox_list, 'gt')
        return(object_extraction_pred)

    def compute_matching_point_loss(self,matching_points_probability_vector,target):
        map_dict = self.get_items_dict()
        pred_dict = self.get_dict()
        for pred in matching_points_probability_vector.keys():
            pred_dict[map_dict[pred]] += matching_points_probability_vector[pred]
        prediction = torch.tensor(list(pred_dict.values()))
        prediction = torch.unsqueeze(prediction, axis=0)
        prediction = torch.log(prediction + 1e-20)
        ground_truth = torch.tensor(target[0]['labels']) -1
        loss = nn.NLLLoss()
        output = loss(prediction, ground_truth)
        output = output.cpu().numpy().astype(int)
        return output/(min(prediction[0])*-1)


    def plot_losses(self,output_path):
        """
        plot losses of the model epochs.
        :param diz_ep: required. dictionary of epochs results.
        :return: Save the plot in the output folder.
        """
        fig = plt.figure(figsize=(10, 8))
        plt.plot(self.loss_history, label='Attack loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        # plt.grid()
        plt.legend()
        plt.title('Attack loss')
        # plt.show()
        plt.savefig(f'{output_path}/loss.png', dpi=fig.dpi)

    def _augment_images_with_patch(
            self, x: np.ndarray, y: Optional[List[Dict[str, np.ndarray]]],
            patch: np.ndarray, channels_first: bool,step:int
    ) -> Tuple[
        np.ndarray, List[Dict[str, np.ndarray]], Dict[str, Union[int, float]]]:
        """
        Augment images with patch.

        :param x: Sample images.
        :param y: Target labels.
        :param patch: The patch to be applied.
        :param channels_first: Set channels first or last.
        """

        transformations: Dict[str, Union[float, int]] = dict()
        x_copy = x.copy()
        patch_copy = patch.copy()
        x_patch = x.copy()

        if channels_first:
            x_copy = np.transpose(x_copy, (0, 2, 3, 1))
            x_patch = np.transpose(x_patch, (0, 2, 3, 1))
            patch_copy = np.transpose(patch_copy, (1, 2, 0))

        # Apply patch:
        self.patch_location = self.get_patch_location_faster_rcnn_format(
            y[0]['boxes'][0],patch_copy)
        x_1, y_1 = self.patch_location
        x_2, y_2 = x_1 + patch_copy.shape[0], y_1 + patch_copy.shape[1]
        try:
            x_patch[:, x_1:x_2, y_1:y_2, :] = patch_copy
        except:
            # print("Patch location failed...")
            self.patch_location = [200,200]
            x_1, y_1 = self.patch_location
            x_2, y_2 = x_1 + patch_copy.shape[0], y_1 + patch_copy.shape[1]
            x_patch[:, x_1:x_2, y_1:y_2, :] = patch_copy


        # 1) crop images:
        crop_x = random.randint(0, self.crop_range[0])
        crop_y = random.randint(0, self.crop_range[1])
        x_1, y_1 = crop_x, crop_y
        x_2, y_2 = x_copy.shape[1] - crop_x + 1, x_copy.shape[2] - crop_y + 1
        x_copy = x_copy[:, x_1:x_2, y_1:y_2, :]
        x_patch = x_patch[:, x_1:x_2, y_1:y_2, :]

        transformations.update({"crop_x": crop_x, "crop_y": crop_y})

        # 2) rotate images:
        rot90 = random.choices([0, 1, 2, 3], weights=self.rotation_weights)[0]

        x_copy = np.rot90(x_copy, rot90, (1, 2))
        x_patch = np.rot90(x_patch, rot90, (1, 2))

        transformations.update({"rot90": rot90})

        if y is not None:

            y_copy: List[Dict[str, np.ndarray]] = list()

            for i_image in range(x_copy.shape[0]):
                y_b = y[i_image]["boxes"].copy()
                image_width = x.shape[2]
                image_height = x.shape[1]
                x_1_arr = y_b[:, 0]
                y_1_arr = y_b[:, 1]
                x_2_arr = y_b[:, 2]
                y_2_arr = y_b[:, 3]
                box_width = x_2_arr - x_1_arr
                box_height = y_2_arr - y_1_arr

                if rot90 == 0:
                    x_1_new = x_1_arr
                    y_1_new = y_1_arr
                    x_2_new = x_2_arr
                    y_2_new = y_2_arr

                if rot90 == 1:
                    x_1_new = y_1_arr
                    y_1_new = image_width - x_1_arr - box_width
                    x_2_new = y_1_arr + box_height
                    y_2_new = image_width - x_1_arr

                if rot90 == 2:
                    x_1_new = image_width - x_2_arr
                    y_1_new = image_height - y_2_arr
                    x_2_new = x_1_new + box_width
                    y_2_new = y_1_new + box_height

                if rot90 == 3:
                    x_1_new = image_height - y_1_arr - box_height
                    y_1_new = x_1_arr
                    x_2_new = image_height - y_1_arr
                    y_2_new = x_1_arr + box_width

                y_i = dict()
                y_i["boxes"] = np.zeros_like(y[i_image]["boxes"])
                y_i["boxes"][:, 0] = x_1_new
                y_i["boxes"][:, 1] = y_1_new
                y_i["boxes"][:, 2] = x_2_new
                y_i["boxes"][:, 3] = y_2_new

                y_i["labels"] = y[i_image]["labels"]
                y_i["scores"] = y[i_image]["scores"]

                y_copy.append(y_i)

        # 3) adjust brightness:
        brightness = random.uniform(*self.brightness_range)
        x_copy = np.round(
            brightness * x_copy / self.learning_rate) * self.learning_rate
        x_patch = np.round(
            brightness * x_patch / self.learning_rate) * self.learning_rate

        transformations.update({"brightness": brightness})


        # Adaptive attack settings
        if self.adaptive_attack and (self.adaptive_type=="Scene Manipulation" or self.adaptive_type=="Ensemble"):
            x_copy = np.squeeze(x_copy, axis=0)
            x_patch = np.squeeze(x_patch, axis=0)
            if step==0:
                # 4) adjust blur
                try:
                    x_copy = cv2.blur(x_copy, (self.blur_scale,
                                               self.blur_scale))
                    x_patch = cv2.blur(x_patch, (self.blur_scale, self.blur_scale))
                    transformations.update({"blur": self.blur_scale})
                except:
                    print("Blur failed")

            if step==1:
                # 5) random noise
                noise = (np.random.random_sample(x_copy.shape) - 0.5) * 255
                x_copy = (x_copy + self.noise_scale * noise).astype(np.uint8)
                x_patch = (x_patch + self.noise_scale * noise).astype(np.uint8)
                transformations.update({"random noise": self.noise_scale})
            if step==2:
                # 6) style transfer
                x_copy = x_copy.astype(np.uint8)
                x_patch = x_patch.astype(np.uint8)
                sth = style_transfer_helper()
                x_copy= sth.style_img(
                    x_copy, x_copy, self.style_transfer_model)
                x_patch = sth.style_img(
                    x_patch, x_patch, self.style_transfer_model)

            x_copy = np.expand_dims(x_copy,axis=0)
            x_patch = np.expand_dims(x_patch,axis=0)

        logger.debug("Transformations: %s", str(transformations))

        patch_target: List[Dict[str, np.ndarray]] = list()

        if self.targeted:
            predictions = y_copy
        else:
            predictions = self.estimator.predict(x=x_copy,
                                                 standardise_output=True)

        for i_image in range(x_copy.shape[0]):
            target_dict = dict()
            target_dict["boxes"] = predictions[i_image]["boxes"]
            target_dict["labels"] = predictions[i_image]["labels"]
            target_dict["scores"] = predictions[i_image]["scores"]
            # add here matching point predictions

            patch_target.append(target_dict)

        if channels_first:
            x_patch = np.transpose(x_patch, (0, 3, 1, 2))

        return x_patch, patch_target, transformations

    def _untransform_gradients(
            self,
            gradients: np.ndarray,
            transforms: Dict[str, Union[int, float]],
            channels_first: bool,
            step: int,
    ) -> np.ndarray:
        """
        Revert transformation on gradients.

        :param gradients: The gradients to be reverse transformed.
        :param transforms: The transformations in forward direction.
        :param channels_first: Set channels first or last.
        """

        if channels_first:
            gradients = np.transpose(gradients, (0, 2, 3, 1))

        # Account for brightness adjustment:
        gradients = transforms["brightness"] * gradients

        # Undo rotations:
        rot90 = (4 - transforms["rot90"]) % 4
        gradients = np.rot90(gradients, rot90, (1, 2))

        # Account for cropping when considering the upper left point of the patch:
        x_1 = self.patch_location[0] - int(transforms["crop_x"])
        y_1 = self.patch_location[1] - int(transforms["crop_y"])
        x_2 = x_1 + self.patch_shape[0]
        y_2 = y_1 + self.patch_shape[1]
        gradients = gradients[:, x_1:x_2, y_1:y_2, :]
        if channels_first:
            gradients = np.transpose(gradients, (0, 3, 1, 2))

        return gradients

    def apply_patch(self, x: np.ndarray,
                    patch_external: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply the adversarial patch to images.

        :param x: Images to be patched.
        :param patch_external: External patch to apply to images `x`. If None the attacks patch will be applied.
        :return: The patched images.
        """

        x_patch = x.copy()

        if patch_external is not None:
            patch_local = patch_external.copy()
        else:
            patch_local = self._patch.copy()

        if self.estimator.channels_first:
            x_patch = np.transpose(x_patch, (0, 2, 3, 1))
            patch_local = np.transpose(patch_local, (1, 2, 0))

        # Apply patch:
        x_1, y_1 = self.patch_location
        x_2, y_2 = x_1 + patch_local.shape[0], y_1 + patch_local.shape[1]

        if x_2 > x_patch.shape[1] or y_2 > x_patch.shape[2]:  # pragma: no cover
            raise ValueError("The patch (partially) lies outside the image.")

        x_patch[:, x_1:x_2, y_1:y_2, :] = patch_local

        if self.estimator.channels_first:
            x_patch = np.transpose(x_patch, (0, 3, 1, 2))

        return x_patch

    def get_patch_location_faster_rcnn_format(self,bbox, patch, image_width=640,
                                              image_height=360):

        object_width = bbox[2] - bbox[0]
        object_height = bbox[3] - bbox[1]
        if (bbox[1] + object_height) < patch.shape[1]:
            patch = imutils.resize(patch, width=int(object_height * 0.5))

        object_x_middle = (bbox[2] + bbox[0]) / 2
        object_y_middle = (bbox[3] + bbox[1]) / 2

        x_patch_location = object_x_middle - (patch.shape[0] / 2)
        y_patch_location = object_y_middle - (patch.shape[1] / 2)
        return (int(y_patch_location),int(x_patch_location),)


    def _check_params(self) -> None:
        if not isinstance(self.patch_shape, (tuple, list)) or not all(
                isinstance(s, int) for s in self.patch_shape):
            raise ValueError(
                "The patch shape must be either a tuple or list of integers.")
        if len(self.patch_shape) != 3:
            raise ValueError("The length of patch shape must be 3.")

        if not isinstance(self.learning_rate, float):
            raise ValueError("The learning rate must be of type float.")
        if self.learning_rate <= 0.0:
            raise ValueError("The learning rate must be greater than 0.0.")

        if not isinstance(self.max_iter, int):
            raise ValueError(
                "The number of optimization steps must be of type int.")
        if self.max_iter <= 0:
            raise ValueError(
                "The number of optimization steps must be greater than 0.")

        if not isinstance(self.batch_size, int):
            raise ValueError("The batch size must be of type int.")
        if self.batch_size <= 0:
            raise ValueError("The batch size must be greater than 0.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")

        if not isinstance(self.patch_location, (tuple, list)) or not all(
                isinstance(s, int) for s in self.patch_location
        ):
            raise ValueError(
                "The patch location must be either a tuple or list of integers.")
        if len(self.patch_location) != 2:
            raise ValueError("The length of patch location must be 2.")

        if not isinstance(self.crop_range, (tuple, list)) or not all(
                isinstance(s, int) for s in self.crop_range):
            raise ValueError(
                "The crop range must be either a tuple or list of integers.")
        if len(self.crop_range) != 2:
            raise ValueError("The length of crop range must be 2.")

        if self.crop_range[0] > self.crop_range[1]:
            raise ValueError(
                "The first element of the crop range must be less or equal to the second one.")

        if self.patch_location[0] < self.crop_range[0] or self.patch_location[
            1] < self.crop_range[1]:
            raise ValueError(
                "The patch location must be outside the crop range.")

        if not isinstance(self.brightness_range, (tuple, list)) or not all(
                isinstance(s, float) for s in self.brightness_range
        ):
            raise ValueError(
                "The brightness range must be either a tuple or list of floats.")
        if len(self.brightness_range) != 2:
            raise ValueError("The length of brightness range must be 2.")

        if self.brightness_range[0] < 0.0:
            raise ValueError("The brightness range must be >= 0.0.")

        if self.brightness_range[0] > self.brightness_range[1]:
            raise ValueError(
                "The first element of the brightness range must be less or equal to the second one.")

        if not isinstance(self.rotation_weights, (tuple, list)) or not all(
                isinstance(s, (float, int)) for s in self.rotation_weights
        ):
            raise ValueError(
                "The rotation sampling weights must be provided as tuple or list of float or int values.")
        if len(self.rotation_weights) != 4:
            raise ValueError(
                "The number of rotation sampling weights must be 4.")

        if not all(s >= 0.0 for s in self.rotation_weights):
            raise ValueError(
                "The rotation sampling weights must be non-negative.")

        if all(s == 0.0 for s in self.rotation_weights):
            raise ValueError(
                "At least one of the rotation sampling weights must be strictly greater than zero.")

        if not isinstance(self.sample_size, int):
            raise ValueError("The EOT sample size must be of type int.")
        if self.sample_size <= 0:
            raise ValueError("The EOT sample size must be greater than 0.")

        if not isinstance(self.targeted, bool):
            raise ValueError("The argument `targeted` has to be of type bool.")
