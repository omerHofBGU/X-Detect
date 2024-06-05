import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
from torch.utils.data import DataLoader
from matplotlib.patches import Rectangle
from Object_detection.transformations import ComposeSingle, FunctionWrapperSingle, normalize_01
# from Object_detection.Object_detection_digital_use_case import ObjectDetectionDatasetSingleFromNumpy
from Object_detection.Object_detection_physical_use_case import ObjectDetectionDatasetSingleFromNumpy

"""
A module for handling MSCOCO dataset.
This module uses 'pycocotools' for more information visit 
https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools 
"""

# COCO instances using 91 classes (used in MSCOCO 2017 version)
COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
    "adversarial patch"
]

# COCO instances using 80 classes (used in pycocotools)

COCO_INSTANCE_CATEGORY_NAMES_80_CLASSES = [
    "background",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
    "adversarial patch"
]


class MS_COCO_util():
    """
    Class representing MSCOCO util object. Create an instance of this class to used functionality on MSCOCO dataset.
    """

    def __init__(self,dataset_path,annotations_file_path,ids_list_path):
        """
        Args:
            dataset_path: required. str. path to MSCOCO dataset (folders of images).
            annotations_file_path: req. str. path to MSCOCO annotations file.
            ids_list_path: req. str. path to txt file containing specific image ids to be loaded.
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.annotations_file_path =annotations_file_path
        self.coco = self.create_coco_object()
        self.ids_list = self.set_ids_list(ids_list_path)

    def create_coco_object(self):
        """
        Create an COCO instance using pycocotools.
        Returns: COCO instance.
        """
        return COCO(self.annotations_file_path)

    def set_ids_list(self, ids_list_path):
        """
        Open ids file to ids list.
        Args:
            ids_list_path: req. str. path to txt file containing specific image ids to be loaded.
        Returns: image ids as a list instance.

        """
        if isinstance(ids_list_path, str):
            file1 = open(ids_list_path, 'r')
            ids_list_path = file1.read().splitlines()
            return list(map(int, ids_list_path))
        else:
            return ids_list_path

    def get_imgs_dict_by_id_list(self):
        """
        Get image data as dictionary from list of ids.
        Returns: image data as dictionary.
        """
        imgIds = self.coco.getImgIds(imgIds=self.ids_list)
        imgIds = self.coco.loadImgs(imgIds)
        return imgIds

    def get_img_by_img_dict(self,img_dict):
        """
        Get image as numpy array from an image dict instance.
        Args:
            img_dict: req. dict of information about a specific image.
        Returns: image as numpy array
        """
        return io.imread(img_dict['coco_url'])

    def get_img_annotations_by_img_dict(self, img_dict):
        """
        use COCO instance to load dictionary representation the annotations of a specific image
        (segmentation, bounding box etc.)
        Args:
            img_dict:  req. dict of information about a specific image.

        Returns: Dictionary representation the annotations of a specific image
        """
        annIds = self.coco.getAnnIds(imgIds=img_dict['id'], catIds=[],
                                     iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        return anns

    def plot_image_bbox(self,img_dict,catIds):
        """
        Plot the image with its corresponding bounding boxes of each detected object.
        Args:
            img_dict: req. dict of information about a specific image.
            catIds: category ids.
        Returns: A plot of the image with its corresponding bounding boxes of each detected object.

        """
        I = self.get_img_by_img_dict(img_dict)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(I)
        plt.axis('off')
        annIds = self.coco.getAnnIds(imgIds=img_dict['id'], catIds=catIds,
                                   iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        for i, ann in enumerate(anns):
            plt.gca().add_patch(Rectangle((anns[i]['bbox'][0], anns[i][
                'bbox'][1]), anns[i]['bbox'][2], anns[i]['bbox'][3],
                         edgecolor = 'green',facecolor='none'))
            ax.text(anns[i]['bbox'][0], anns[i]['bbox'][1],
                    COCO_INSTANCE_CATEGORY_NAMES[anns[i]['category_id']],
                                                 style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 5})
        plt.show()

    def label_list_to_indexes(self,label_list):
        """
        Transform labels to indexing using COCO_INSTANCE_CATEGORY_NAMES.
        Args:
            label_list: req. list of labels (strings).

        Returns: list of indexes in COCO format.
        """
        return [self.label_to_index(label) for label in label_list]

    def label_to_index(self,label):
        """
        Transform label to index using COCO_INSTANCE_CATEGORY_NAMES.
        Args:
            label: req. label (str).
        Returns: index in COCO format.

        """
        return COCO_INSTANCE_CATEGORY_NAMES.index(label)

    def index_to_label(self,index):
        """
        Transform index to label using COCO_INSTANCE_CATEGORY_NAMES.
        Args:
            index: req. int.
        Returns: Corresponding label of the given index in COCO format.
        """
        return COCO_INSTANCE_CATEGORY_NAMES[index]

    def display_coco_categories(self):
        """
        Display COCO categories and super-categories
        Returns: COCO categories and super-categories.
        """
        #
        cats = self.coco.loadCats(self.coco.getCatIds())
        nms = [cat['name'] for cat in cats]
        print('COCO categories: \n{}\n'.format(' '.join(nms)))

        nms = set([cat['supercategory'] for cat in cats])
        print('COCO supercategories: \n{}'.format(' '.join(nms)))

    def load_coco_dataset_from_ids_list(self):
        """
        Loads coco dataset from ids list.
        Returns: images dictionary.
        """
        images_dict = self.get_imgs_dict_by_id_list()
        for img_dict in images_dict:
            img_dict['annotations'] = self.get_img_annotations_by_img_dict(img_dict)
            img_dict['image'] = self.get_img_by_img_dict(img_dict)

        return images_dict

    def eval(self,cocoDt,annType,cat_ids):
        """
        Evaluate object detector predictions.
        Args:
            cocoDt: req. coco detections.
            annType: req. str. "bbox" or "segmentations".
            cat_ids: req. list of categories ids.
        Returns: evaluation report produced by pycocotools.
        """
        coco_eval = COCOeval(self.coco,cocoDt,annType)
        imgIds = sorted(self.coco.getImgIds(imgIds=self.ids_list))
        coco_eval.params.imgIds = imgIds
        coco_eval.params.catIds = cat_ids

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    def transformations(self):
        """
        Transformation function for the given dataset (mainly transform from
        numpy to torch and normalize).
        :return: Transformations function.
        """
        return ComposeSingle([
            FunctionWrapperSingle(np.moveaxis, source=-1, destination=0),
            FunctionWrapperSingle(normalize_01)
        ])

    def coco_to_dataloader(self, inputs, batch_size=1):
        """
        Form a pytorch dataloader instance for real time detection.
        :param inputs: req. numpy array representing an image.
        :param batch_size: optinal. tha number of images in a batch (default
        is 1).
        :return: pytorch dataloader instance.
        """

        # transformations
        transforms = self.transformations()

        # create dataset and dataloader
        dataset = ObjectDetectionDatasetSingleFromNumpy(inputs=inputs,
                                                        transform=transforms,
                                                        use_cache=False,
                                                        )

        dataloader_prediction = DataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=0)
        return dataloader_prediction

    def image_dict_to_json(self,images_dict,output_path):
        """
        Transform images dictionary to Json file.
        Args:
            images_dict: req. dict of information about a COCO images.
            output_path: req. str. path to save the json file.

        Returns: Save the json file in the given path.
        """
        images = []
        categories = []
        annotations = []
        for image_dict in images_dict:
            images.append(self.process_image(image_dict,output_path))
            for idx,annotation in enumerate(image_dict['annotations']):
                annotations.append(self.process_annotations(image_dict,annotation))
        data_coco = {}
        data_coco["images"] = images
        data_coco["categories"] = categories
        data_coco["annotations"] = annotations
        json.dump(data_coco, open(f'{output_path}/ms_coco_format.json', "w"), indent=4)

    def process_image(self,image_dict,output_path):
        """
        process image to mscoco format.
        Args:
            image_dict: req. dict of information about a specific COCO image.
            output_path: req. str. path to save the image file.
        Returns: dictionary in mscoco format.
        """
        image = {}
        image["height"] = image_dict['height']
        image["width"] = image_dict['width']
        image["id"] = image_dict['id']
        image["file_name"] = image_dict['file_name']
        output_image = cv2.cvtColor(image_dict['image'],cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{output_path}/{image_dict['id']}.jpeg",output_image)
        return image

    def process_annotations(self,image_dict,curr_annotation):
        """
         process annotations to mscoco format.
        Args:
            image_dict: req. dict of information about a specific COCO image.
            curr_annotation: req. dict of annotations about a specific COCO image.
        Returns: dictionary of annotations in mscoco format.

        """
        annotation = {}
        area = image_dict['width'] * image_dict['height']
        annotation["segmentation"] = []
        annotation["iscrowd"] = curr_annotation['iscrowd']
        annotation["area"] = float(area)
        annotation["image_id"] = curr_annotation['image_id']
        annotation["bbox"] = curr_annotation['bbox']
        annotation["category_id"] = curr_annotation['category_id']
        annotation["id"] = curr_annotation['id']
        return annotation

    def get_ids_from_specific_class(self, classes):
        """
        Function that return instances from specific class.
        Args:
            classes: req. list of COCO classes (strings).
        Returns: COCO image id list selected from the given classes list.
        """
        ids_list = []
        for class_name in classes:
            catIds = self.coco.getCatIds(catNms=class_name)
            imgIds = self.coco.getImgIds(catIds=catIds)
            imgIds = imgIds[:300]
            ids_list.append(imgIds)
        self.ids_list = [item for sublist in ids_list for item in sublist]


def ms_coco_demo(data_dir,annotations_file_path,ids_list_path):
    """
    Demo function that uses ms coco util functionality:
    Upload ids, select ids from specific class, save those ids to json file in mscoco format.
    Args:
        data_dir: required. str. path to MSCOCO dataset (folders of images).
        annotations_file_path: req. str. path to MSCOCO annotations file.
        ids_list_path: req. str. path to txt file containing specific image ids to be loaded.
        dataset_path:
    Returns: Save a json file of and images from COCO dataset.
    """
    data_util = MS_COCO_util(data_dir,annotations_file_path,ids_list_path)
    data_util.get_ids_from_specific_class(["banana","apple","orange","pizza"])
    image_dict = data_util.load_coco_dataset_from_ids_list()
    data_util.image_dict_to_json(image_dict,data_dir+"/ad_yolo_train_data")

# Params example:
# data_dir = '/dt/shabtaia/dt-fujitsu-explainability/MSCOCO'
# annotations_file_path = os.path.join(data_dir,'dataset.json')
# ids_list_path = "/dt/shabtaia/dt-fujitsu-explainability/MSCOCO/Evaluation_data/ids_list/apple/ids_list.txt"
# ms_coco_demo(data_dir,annotations_file_path,ids_list_path)