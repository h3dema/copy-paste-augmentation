import os
import zipfile
import requests
import cv2
import urllib
from torchvision.datasets import CocoDetection

from copy_paste import copy_paste_annotation


min_keypoints_per_image = 10

# ref. https://cocodataset.org/#download

data_urls: dict[str, str] = {
    'train2014': 'http://images.cocodataset.org/zips/train2014.zip',
    'val2014': 'http://images.cocodataset.org/zips/val2014.zip',
    'test2014': 'http://images.cocodataset.org/zips/test2014.zip',
    'train2017': 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
    }

annot_urls: dict[str, str] = {
    'train2014': 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
    'val2014': 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
    'test2014': 'http://images.cocodataset.org/annotations/image_info_test2014.zip',
    }

def download_dataset(
    download_dir: str = 'datasets/coco/',  # Path to the download directory
    dataset_name: str = 'train2014',
    overwrite: bool = False,
) -> str:
    """
    Downloads and extracts the COCO dataset.

    Parameters
    ----------
    download_dir : str, optional
        Path to the directory where the dataset should be downloaded and extracted.
        Defaults to 'datasets/coco/'.
    dataset_name: str, optional
        Name of the dataset to download. Valid names are the keys from `data_urls`.
        Defaults to 'train2014'
    overwrite : bool, optional
        If True, overwrites the existing dataset file if it exists. Defaults to False.

    Returns
    -------
        Path to the directories where the dataset and the annotations were extracted.

    Notes
    -----
    Downloads the COCO 2014 dataset and annotations as a zip file and extracts its contents
    to the specified directory.
    """

    assert dataset_name in data_urls.keys(), "Dataset name must be one of {}".format(data_urls.keys())

    def download(download_dir, url, zip_fname,
                 overwrite=False,
                 chunk_size: int = 1024*1024,
                 ):
        """
        Downloads the zip file from the specified URL to the specified directory.

        Parameters
        ----------
        download_dir : str
            Path to the download directory.
        url : str
            URL to the dataset.
        zip_fname : str
            Name of the zip file to be saved.
        overwrite : bool, optional
            If True, overwrites the existing dataset file if it exists. Defaults to False.
        chunk_size: int, optional
            chunk_size is the number of bytes iter_content reads into memory
            Defaults to 1024 * 1024 (i.e. 1 MB)
        """
        response = requests.get(url, stream=True)
        fname = os.path.join(download_dir, zip_fname)
        if overwrite or not os.path.exists(fname):
            # Download the zip file
            with open(fname, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)

    def unzip(download_dir, zip_fname):
        """
        Unzips the dataset to the specified directory.

        Parameters
        ----------
        download_dir : str
            Path to the download directory.
        zip_fname : str
            Name of the zip file.
        """
        # Unzip the dataset
        with zipfile.ZipFile(os.path.join(download_dir, zip_fname), 'r') as z:
            z.extractall(download_dir)

    os.makedirs(download_dir, exist_ok=True)

    # URL to the COCO dataset
    url = data_urls[dataset_name]
    zip_fname = urllib.parse.urlparse(url).path.split('/')[-1]
    download(download_dir, url, zip_fname, overwrite=overwrite)
    unzip(download_dir, zip_fname)

    # URL to the COCO dataset annotations
    url = annot_urls[dataset_name]
    zip_fname = urllib.parse.urlparse(url).path.split('/')[-1]
    download(download_dir, url, zip_fname, overwrite=overwrite)
    unzip(download_dir, zip_fname)

    return os.path.join(download_dir, dataset_name), os.path.join(download_dir, 'annotations')


def _count_visible_keypoints(anno):
    """
    Helper function to count the number of keypoints that are visible in an annotation.

    Args:
        anno (list[dict]): Annotation list.

    Returns:
        int: Number of visible keypoints.
    """
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    """
    Check if all bounding box annotations have zero area.

    Parameters
    ----------
    anno : list[dict]
        Annotation list.

    Returns
    -------
    bool
        If all bounding boxes have zero area.
    """
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    """
    Determines if the given annotation contains valid data.

    Parameters
    ----------
    anno : list[dict]
        List of annotations where each annotation is represented as a dictionary.

    Returns
    -------
    bool
        True if the annotation is considered valid, otherwise False.

    Notes
    -----
    - An annotation is invalid if the list is empty.
    - An annotation is invalid if all bounding boxes have close to zero area.
    - If the task does not involve keypoints, a non-empty annotation is considered valid.
    - For keypoint detection tasks, an annotation is valid if it contains at least
      `min_keypoints_per_image` visible keypoints.
    """

    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True

    return False


@copy_paste_annotation
class CocoDetectionCP(CocoDetection):

    def __init__(
        self,
        root,
        annFile,
        transforms
    ):
        """
        Initialize a CocoDetection dataset with copy-paste augmentation.

        Parameters
        ----------
        root : str
            Path to the root directory of the COCO dataset.
        annFile : str
            Path to the annotation file.
        transforms : callable
            A callable that takes in a dict and returns a dict with transformed data.
            Notice that CopyPaste should be in the transforms list.
        """
        super().__init__(
            root, annFile, None, None, transforms
        )

        # filter images without detection annotations
        ids = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = self.coco.loadAnns(ann_ids)
            if has_valid_annotation(anno):
                ids.append(img_id)
        self.ids = ids  # overwrite the original ids

        assert len(self.ids), 'The COCO dataset is empty! I did not find any image with valid annotations.'

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)

        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = cv2.imread(os.path.join(self.root, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # convert all of the target segmentations to masks
        # bboxes are expected to be (y1, x1, y2, x2, category_id)
        masks = []
        bboxes = []
        for ix, obj in enumerate(target):
            masks.append(self.coco.annToMask(obj))
            bboxes.append(obj['bbox'] + [obj['category_id']] + [ix])

        # pack outputs into a dict
        output = {
            'image': image,
            'masks': masks,
            'bboxes': bboxes
        }
        return self.transforms(**output)
