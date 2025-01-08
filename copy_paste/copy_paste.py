"""
Copy-Paste Augmentation


Todo: Allow a second dataset to be the source of the images to be pasted.

"""
import os
import cv2
import random
import numpy as np
import albumentations as A
from copy import deepcopy
from skimage.filters import gaussian

import pdb  # HERE!!


def image_copy_paste(img, paste_img, alpha, blend=True, sigma=1):
    """
    Paste an image onto another image with an alpha mask.

    Parameters
    ----------
    img : (H, W, 3) ndarray
        The image to paste onto.
    paste_img : (H, W, 3) ndarray
        The image to paste.
    alpha : (H, W) ndarray or None
        Alpha mask for the paste image. If alpha mask is None, returns img unaltered.
    blend : bool, optional
        If True, apply a Gaussian filter to the alpha mask before pasting.
        Defaults to True.
    sigma : float, optional
        The standard deviation of the Gaussian filter to apply to the alpha mask.
        Defaults to 1.

    Returns
    -------
    img : (H, W, 3) ndarray
        The pasted image.
    """
    if alpha is not None:
        if blend:
            alpha = gaussian(alpha, sigma=sigma, preserve_range=True)

        img_dtype = img.dtype
        alpha = alpha[..., None]
        img = paste_img * alpha + img * (1 - alpha)
        img = img.astype(img_dtype)

    return img

def mask_copy_paste(mask, paste_mask, alpha):
    """
    Paste a mask onto another mask with an optional alpha mask.

    Parameters
    ----------
    mask : (H, W) ndarray
        The mask to paste onto.
    paste_mask : (H, W) ndarray
        The mask to paste.
    alpha : (H, W) ndarray or None
        Optional alpha mask for the paste mask.

    Returns
    -------
    mask : (H, W) ndarray
        The pasted mask.
    """
    raise NotImplementedError


def masks_copy_paste(masks, paste_masks, alpha):
    """
    Paste a list of masks onto another list of masks with an optional alpha mask.

    Parameters
    ----------
    masks : list of (H, W) ndarrays
        The list of masks to paste onto.
    paste_masks : list of (H, W) ndarrays
        The list of masks to paste.
    alpha : (H, W) ndarray or None
        Alpha mask for the paste masks. If alpha mask is None, returns masks unaltered.

    Returns
    -------
    masks : list of (H, W) ndarrays
        The pasted masks.
    """
    if alpha is not None:
        # eliminate pixels that will be pasted over
        masks = [
            np.logical_and(mask, np.logical_xor(mask, alpha)).astype(np.uint8) for mask in masks
        ]
        masks.extend(paste_masks)

    return masks


def extract_bboxes(masks) -> list:
    """
    Extract bounding boxes from a list of masks.

    Parameters
    ----------
    masks : list of (H, W) ndarrays
        The list of masks from which to extract bounding boxes.

    Returns
    -------
    bboxes : list of (y1, x1, y2, x2)
        The bounding boxes extracted from the masks, in normalized coordinates.
    """
    bboxes = []
    # allow for case of no masks
    if len(masks) == 0:
        return bboxes

    h, w = masks[0].shape
    for mask in masks:
        yindices = np.where(np.any(mask, axis=0))[0]
        xindices = np.where(np.any(mask, axis=1))[0]
        if yindices.shape[0]:
            y1, y2 = yindices[[0, -1]]
            x1, x2 = xindices[[0, -1]]
            y2 += 1
            x2 += 1
            y1 /= w
            y2 /= w
            x1 /= h
            x2 /= h
        else:
            y1, x1, y2, x2 = 0., 0., 0., 0.

        # guaratees values are float
        bboxes.append([float(x) for x in (y1, x1, y2, x2)])

    return bboxes

def bboxes_copy_paste(bboxes, paste_bboxes, masks, paste_masks, alpha, key) -> np.ndarray:
    if key == 'paste_bboxes':
        return bboxes
    elif paste_bboxes is not None:
        masks = masks_copy_paste(masks, paste_masks=[], alpha=alpha)
        adjusted_bboxes = extract_bboxes(masks)

        mask_indices: list[int] = [int(box[-1]) for box in bboxes]
        # only keep the bounding boxes for objects listed in bboxes
        adjusted_bboxes = [adjusted_bboxes[idx] for idx in mask_indices]
        #append bbox tails (classes, etc.)
        assert len(bboxes) == len(adjusted_bboxes)
        # add at the end of each line in adjusted_bboxes the last 2 values in bboxes
        adjusted_bboxes = [list(bbox) + tail[4:].tolist() for bbox, tail in zip(adjusted_bboxes, bboxes)]

        #adjust paste_bboxes mask indices to avoid overlap
        if len(masks) > 0:
            max_mask_index = len(masks)
        else:
            max_mask_index = 0

        paste_mask_indices: list[int] = [max_mask_index + ix for ix in range(len(paste_bboxes))]
        paste_bboxes: list = [pbox[:-1].tolist() + [pmi,] for pbox, pmi in zip(paste_bboxes, paste_mask_indices)]
        adjusted_paste_bboxes = extract_bboxes(paste_masks)
        adjusted_paste_bboxes = [apbox + tail[4:] for apbox, tail in zip(adjusted_paste_bboxes, paste_bboxes)]

        bboxes = adjusted_bboxes + adjusted_paste_bboxes
        bboxes = np.array(bboxes)

    return bboxes

def keypoints_copy_paste(keypoints, paste_keypoints, alpha):
    #remove occluded keypoints
    if alpha is not None:
        visible_keypoints = []
        for kp in keypoints:
            x, y = kp[:2]
            tail = kp[2:]
            if alpha[int(y), int(x)] == 0:
                visible_keypoints.append(kp)

        keypoints = visible_keypoints + paste_keypoints

    return keypoints

class CopyPaste(A.DualTransform):
    def __init__(
        self,
        blend=True,
        sigma=3,
        pct_objects_paste=0.1,
        max_paste_objects=None,
        p=0.5,
        always_apply=False
    ):
        super().__init__(always_apply, p)

        self.blend = blend
        self.sigma = sigma
        self.pct_objects_paste = pct_objects_paste
        self.max_paste_objects = max_paste_objects
        self.p = p
        self.always_apply = always_apply

    @staticmethod
    def get_class_fullname():
        return 'copypaste.CopyPaste'

    @property
    def targets_as_params(self):
        return [
            "masks",
            "paste_image",
            #"paste_mask",  # TODO: not implemented
            "paste_masks",
            "paste_bboxes",
            #"paste_keypoints"
        ]

    def get_params_dependent_on_targets(self, params):
        image = params["paste_image"]
        masks = None
        if "paste_mask" in params:
            #handle a single segmentation mask with
            #multiple targets
            #nothing for now.
            raise NotImplementedError
        elif "paste_masks" in params:
            masks = params["paste_masks"]

        assert(masks is not None), "Masks cannot be None!"

        bboxes = params.get("paste_bboxes", None)
        keypoints = params.get("paste_keypoints", None)

        # number of objects: n_bboxes <= n_masks because of automatic removal
        n_objects = len(bboxes) if bboxes is not None else len(masks)

        # paste all objects if no restrictions
        n_select = n_objects
        if self.pct_objects_paste:
            n_select = int(n_select * self.pct_objects_paste)
        if self.max_paste_objects:
            n_select = min(n_select, self.max_paste_objects)

        # no objects condition
        if n_select == 0:
            return {
                "param_masks": params["masks"],
                "paste_img": None,
                "alpha": None,
                "paste_mask": None,
                "paste_masks": None,
                "paste_bboxes": None,
                "paste_keypoints": None,
                "objs_to_paste": []
            }

        #select objects
        objs_to_paste = np.random.choice(
            range(0, n_objects), size=n_select, replace=False
        )

        # take the bboxes
        if bboxes is not None and len(bboxes) > 0:
            bboxes = [bboxes[idx] for idx in objs_to_paste]
            #the last label in bboxes is the index of corresponding mask
            mask_indices = [int(bbox[-1]) for bbox in bboxes]

        # create alpha by combining all the objects into a single binary mask
        masks = [masks[idx] for idx in mask_indices]

        alpha = masks[0] > 0
        for mask in masks[1:]:
            alpha += mask > 0
        return {
            "param_masks": params["masks"],
            "paste_img": image,
            "alpha": alpha,
            "paste_mask": None,
            "paste_masks": masks,
            "paste_bboxes": bboxes,
            "paste_keypoints": keypoints
        }

    @property
    def ignore_kwargs(self):
        return [
            "paste_image",
            "paste_mask",
            "paste_masks"
        ]

    def _get_target_function(self, key):
        """ copied from https://vfdev-5-albumentations.readthedocs.io/en/docs_pytorch_fix/_modules/albumentations/core/transforms_interface.html """
        transform_key = key
        if key in self._additional_targets:
            transform_key = self._additional_targets.get(key, None)

        target_function = self.targets.get(transform_key, lambda x, **p: x)
        return target_function

    @property
    def target_dependence(self):
        # no extra dependency based on targets
        return {}

    def apply_with_params(self, params, force_apply=False, **kwargs):  # skipcq: PYL-W0613
        """
            Apply transformation functions to specified arguments using provided parameters.
            This is the function called by self.copy_paste(**img_data, **paste_img_data) in `copy_paste_annotations`.

            Parameters
            ----------
            params : dict
                Parameters for transformations. If None, the function returns the input arguments unaltered.
            force_apply : bool, optional
                Flag to force application of transformations. Currently unused in the function. Defaults to False.
            **kwargs : dict
                Keyword arguments representing data to transform. Keys refer to the type of data
                (e.g., 'masks', 'bboxes', 'keypoints', 'image') and values are the actual data.

            Returns
            -------
            dict
                Transformed data with the same keys as input kwargs. If a key is not transformed,
                its value will be None in the returned dictionary.
        """
        if params is None:
            return kwargs
        params = self.update_params(params, **kwargs)
        res = {}
        for key, arg in kwargs.items():
            if arg is not None and key not in self.ignore_kwargs:
                target_function = self._get_target_function(key)
                # if key is 'masks', target_function is apply_to_masks
                #        is 'bboxes', target_function is apply_to_bboxes
                #        is 'keypoints', target_function is apply_to_keypoints
                #        is 'image', target_function is apply
                target_dependencies = {k: kwargs[k] for k in self.target_dependence.get(key, [])}  # actually this lines is useless. it is only an empty dict.
                target_dependencies['key'] = key
                res[key] = target_function(arg, **dict(params, **target_dependencies))
            else:
                res[key] = None
        return res

    def apply(self, img, paste_img, alpha, **params):
        return image_copy_paste(
            img, paste_img, alpha, blend=self.blend, sigma=self.sigma
        )

    def apply_to_mask(self, mask, paste_mask, alpha, **params):
        return mask_copy_paste(mask, paste_mask, alpha)

    def apply_to_masks(self, masks, paste_masks, alpha, **params):
        return masks_copy_paste(masks, paste_masks, alpha)

    def apply_to_bboxes(self, bboxes, paste_bboxes, param_masks, paste_masks, alpha, key, **params):
        return bboxes_copy_paste(bboxes, paste_bboxes, param_masks, paste_masks, alpha, key)

    def apply_to_keypoints(self, keypoints, paste_keypoints, alpha, **params):
        raise NotImplementedError
        #return keypoints_copy_paste(keypoints, paste_keypoints, alpha)

    def get_transform_init_args_names(self):
        return (
            "blend",
            "sigma",
            "pct_objects_paste",
            "max_paste_objects"
        )

def copy_paste_annotation(dataset_class):
    """
    Modifies a dataset class so that it supports Copy-Paste augmentation.

    Copy-Paste augmentation is a kind of augmentation that copies objects from one image
    and pastes them onto another image. It is useful for tasks such as object detection,
    segmentation, and tracking.

    this should be used as a function annotation for a dataset class, sse example below.

    This function splits the transforms defined on the dataset into three parts:
    pre_copy, copy_paste, and post_copy. The pre_copy transforms are applied to both the
    original and paste images. The copy_paste transform is applied to the paste image.
    The post_copy transforms are applied to the resulting image.

    Notice that the annotation assumes that the original __getitem__ function from the dataset was
    actually written as `def load_example(self, index)` because the annotation will
    overwrite the __getitem__ function.

    Args:
        dataset_class (class): The class of the dataset to be modified.

    Returns:
        class: The modified dataset class.

    Example:

        @copy_paste_annotation
        class CocoDetectionCP(CocoDetection):
            ...
    """
    def _split_transforms(self):
        split_index = None
        for ix, tf in enumerate(list(self.transforms.transforms)):
            if tf.get_class_fullname() == 'copypaste.CopyPaste':
                split_index = ix

        if split_index is not None:
            tfs = list(self.transforms.transforms)
            pre_copy = tfs[:split_index]
            copy_paste = tfs[split_index]
            post_copy = tfs[split_index+1:]

            #replicate the other augmentation parameters
            bbox_params = None
            keypoint_params = None
            paste_additional_targets = {}
            if 'bboxes' in self.transforms.processors:
                bbox_params = self.transforms.processors['bboxes'].params
                paste_additional_targets['paste_bboxes'] = 'bboxes'
                if self.transforms.processors['bboxes'].params.label_fields:
                    msg = "Copy-paste does not support bbox label_fields! "
                    msg += "Expected bbox format is (a, b, c, d, label_field)"
                    raise Exception(msg)
            if 'keypoints' in self.transforms.processors:
                keypoint_params = self.transforms.processors['keypoints'].params
                paste_additional_targets['paste_keypoints'] = 'keypoints'
                if keypoint_params.label_fields:
                    raise Exception('Copy-paste does not support keypoint label fields!')

            if self.transforms.additional_targets:
                raise Exception('Copy-paste does not support additional_targets!')

            #recreate transforms
            self.transforms = A.Compose(pre_copy, bbox_params, keypoint_params, additional_targets=None)
            self.post_transforms = A.Compose(post_copy, bbox_params, keypoint_params, additional_targets=None)
            # in general, bbox_params is albumentations.core.bbox_utils.BboxProcessor and
            # keypoint_params is albumentations.core.keypoint_utils.KeypointProcessor or None (e.g. for COCO dataset)
            # paste_additional_targets = {'paste_bboxes': 'bboxes'} to transform also the paste_image's bboxes
            self.copy_paste = A.Compose(
                [copy_paste], bbox_params, keypoint_params, additional_targets=paste_additional_targets
            )
        else:
            self.copy_paste = None
            self.post_transforms = None

    def __getitem__(self, idx):
        #split transforms if it hasn't been done already
        if not hasattr(self, 'post_transforms'):
            self._split_transforms()
        img_data = self.__load_example__(idx)
        if self.copy_paste is not None:
            paste_idx = random.randint(0, self.__len__() - 1)
            paste_img_data = self.__load_example__(paste_idx)
            for k in list(paste_img_data.keys()):
                paste_img_data['paste_' + k] = paste_img_data[k]
                del paste_img_data[k]

            img_data = self.copy_paste(**img_data, **paste_img_data)
            # remove from paste_img_data the dict_keys ['paste_image', 'paste_masks', 'paste_bboxes']
            for k in ['paste_image', 'paste_masks', 'paste_bboxes']:
                if k in img_data:
                    del img_data[k]
            img_data = self.post_transforms(**img_data)
            img_data['paste_index'] = paste_idx

        return img_data

    setattr(dataset_class, '__load_example__', dataset_class.__getitem__)  # rename the old function to __load_example__
    setattr(dataset_class, '_split_transforms', _split_transforms)
    setattr(dataset_class, '__getitem__', __getitem__)  # overwrite __getitem__

    return dataset_class
