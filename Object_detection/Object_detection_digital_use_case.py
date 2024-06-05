import os
import torch
from Datasets_util.MSCOCO_util import MS_COCO_util
from art.estimators.object_detection import PyTorchFasterRCNN
from pathlib import Path
import datetime
import cv2
import numpy as np
import imutils
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector,show_result_pyplot

"""
This module is for generating target model predictions on the digital use case. 
"""

class digital_attack_evaluation():

    def __init__(self, dataset_path, annotations_file_path, ids_list_path,patch_path,put_patch_on,
                 use_mmdetection = False,target_model_path = None):
        super().__init__()
        self.data_util = MS_COCO_util(dataset_path, annotations_file_path, ids_list_path)
        self.patch_path = patch_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_mmdetection = False
        if use_mmdetection:
            self.use_mmdetection = True
            self.target_model_path = target_model_path
        self.target_model = self.init_object_detector()
        self.put_patch_on = self.data_util.label_list_to_indexes(put_patch_on)

    def init_object_detector(self):
        """
        Create ART object detector, based on pytorch implementation of faster
        rcnn, more information can be found at  https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/estimators/object_detection.html
        or target model from MM Detection package.
        :return: a pre trained object detector, pytorch model object.
        """
        if not self.use_mmdetection:
            # Create ART object detector
            frcnn = PyTorchFasterRCNN(
                clip_values=(0, 255),channels_first=False, attack_losses=(
                    "loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg")
            )
            frcnn.model.to(self.device)
            frcnn.model.eval()
        else:
            frcnn = init_detector(self.target_model_path[1], self.target_model_path[0], device=self.device)
        return frcnn

    def get_most_suitable_object(self, object_params):
        """
        Function that extract the object index with the biggest bounding box, suited for SuperStore dataset.
        :param object_params: req. list of object bounding boxes.
        :return: The index of the most suitable object
        """
        biggest_bbox_size = 0
        best_idx = 0
        for index,instance in enumerate(object_params):
            curr_bbox_size = (instance[2]-instance[0]) * (instance[3]-instance[1])
            if curr_bbox_size>biggest_bbox_size:
                biggest_bbox_size = curr_bbox_size
                best_idx = index
        return object_params[best_idx][:-1]

    def create_faster_rcnn_result_dict_new(self, result):
        """
        Function that transform the result given by an object detector model taken from MMdetection package
        to Faster R-CNN format.
        :param result: req. list. result given by an object detector model taken from MMdetection
        :return: A dictionary presenting the result in Faster R-CNN format.
        """
        device = self.device
        result_faster_rcnn_format = []
        result_dict = {'boxes': torch.empty(size=(0, 4), device=device),
                       'labels': torch.empty(size=(0,), device=device, dtype=torch.int64),
                       'scores': torch.empty(size=(0,), device=device)}
        class_prob = {}
        for idx, class_instance in enumerate(result):
            if len(class_instance) > 0:
                class_prob[idx] = class_instance[0][4]
        labels,scores,boxes = [],[],[]
        if len(class_prob) > 0:
            for prediction_id in class_prob.keys():
                if class_prob[prediction_id]>0.3:
                    labels.append(prediction_id+6)
                    scores.append(class_prob[prediction_id])
                    boxes.append(self.get_most_suitable_object(result[prediction_id]))
            result_dict['labels'] = labels
            result_dict['scores'] = scores
            result_dict['boxes'] = boxes

        result_faster_rcnn_format.append(result_dict)
        return result_faster_rcnn_format

    def predict(self,frame):
        """
        Function that uses object detection model to detect objects in a given frame.
        :param frame:req. numpy format. An image.
        :return: A prediction of which object is located in the given frame and their bounding boxes.
                In Faster R-CNN format.
        """

        if not self.use_mmdetection:
            with torch.no_grad():
                frames = np.expand_dims(frame,axis=0)
                return self.target_model.predict(frames,batch_size=1)
        else:
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            result = inference_detector(self.target_model, frame)
            # show_result_pyplot(self.target_model, frame, result,score_thr=0.25)
            result = self.create_faster_rcnn_result_dict_new(result)
            return result


    def create_output_folders(self):
        """
        Function for crating output folders.
        :return: Create output folders.
        """
        time = datetime.datetime.now().strftime("%d-%m-%Y_%H;%M")
        output_path = f"output/{time}"
        Path(output_path).mkdir(parents=True, exist_ok=True)
        benign_images_path = f'{output_path}/benign_images'
        Path(benign_images_path).mkdir(parents=True, exist_ok=True)
        patch_images_path = f'{output_path}/patch_images'
        Path(patch_images_path).mkdir(parents=True, exist_ok=True)
        adv_dec_path = f'{output_path}/adversarial_dec'
        Path(adv_dec_path).mkdir(parents=True, exist_ok=True)
        plots = f'{output_path}/plots'
        Path(plots).mkdir(parents=True, exist_ok=True)
        return output_path, benign_images_path, patch_images_path, adv_dec_path, plots

    def load_patch(self,patch_path, patch_as_np=True):
        """
        Load existing patch from path.
        :param patch_path: req. str. The path to the patch.
        :param patch_as_np: req. bool. If the patch saved in numpy format.
        :return: The patch in numpy format.
        """
        if patch_as_np:
            patch = np.load(patch_path)
        else:
            patch = cv2.imread(patch_path)
        return patch

    def get_main_object_index(self,frame):
        """
        Function that locate the main object in the frame.
        :param frame: req. numpy array of the examined image.
        :return: the index of the largest object in the frame.
        """
        largest_index = 0
        biggest_bbox_area = 0
        for idx, detection in enumerate(frame['annotations']):
            if detection['category_id'] in self.put_patch_on:
                bbox_area = detection['bbox'][2] * detection[
                    'bbox'][3]
                if biggest_bbox_area < bbox_area:
                    biggest_bbox_area = bbox_area
                    largest_index = idx
        return largest_index

    def apply_patch(self,image, patch, patch_location):
        """
        Function that place the patch in the frame.
        :param image: req. numpy array of the frame.
        :param patch: req. numpy format of the patch.
        :param patch_location:  req. tuple of 2 ints, representing the patch upper left location by x,y coordinates (x1,y1).
        :return: The image with the patch placed into it.
        """
        x_patch = image.copy()
        patch_local = patch.copy()
        # Apply patch:
        x_1, y_1 = patch_location
        x_2, y_2 = x_1 + patch_local.shape[0], y_1 + patch_local.shape[1]
        if x_2 > x_patch.shape[0] or y_2 > x_patch.shape[1]:  # pragma: no cover
            raise ValueError("The patch (partially) lies outside the image.")
        x_patch[x_1:x_2, y_1:y_2, :] = patch_local
        return x_patch


    def apply_patch_to_image(self,frame):
        """
        Wrapper function the load a patch and place it on a given frame.
        :param frame: req. numpy array of the frame.
        :return: The frame with the patch placed inside of it.
        """
        patch = self.load_patch(self.patch_path)
        index = self.get_main_object_index(frame)
        patch_loc, new_patch = self.compute_patch_location(frame['annotations'][index]['bbox'],patch)
        print(f"smaple id:{frame['id']}")
        patched_image = self.apply_patch(frame['image'], new_patch, patch_loc)
        return patched_image

    def compute_patch_location(self, bbox, patch):
        """
        Function that place the patch in the center of an object.
        :param bbox: req. list. the coordinate of the object (x,y,w,h format).
        :param patch: req. numpy format of the patch.
        :return: The coordinated where the patch would be placed on the object (x1,y1).
        """
        object_left_x = bbox[0]
        object_left_y = bbox[1]
        object_width = bbox[2]
        object_height = bbox[3]
        patch_width = object_width * 0.5
        if patch_width < 300 and patch_width > 60:
            patch = imutils.resize(patch, width=int(patch_width))
        if patch_width < 60:
            patch = imutils.resize(patch, width=60)
        if (object_left_y + object_height) < patch.shape[1]:
            patch = imutils.resize(patch, width=int(object_height * 0.5))

        object_x_middle = object_left_x + (object_width / 2)
        object_y_middle = object_left_y + (object_height / 2)

        x_patch_location = object_x_middle - (patch.shape[0] / 2)
        y_patch_location = object_y_middle - (patch.shape[1] / 2)
        return (int(y_patch_location), int(x_patch_location)), patch

    def save_adversarial_samples(self, images_dicts, patch_images_path):
        """
        Function that saves (write) adversarial samples in jpeg and numpy format.
        :param images_dicts: req. list of dictionary containing the adversarial images and additional
        information regarding them.
        :param patch_images_path: req. str. Path to save the images.
        :return: --
        """
        for image_dict in images_dicts:
            adversarial_sample = image_dict['adversarial_image']
            output_path = os.path.join(
                patch_images_path, f"{str(image_dict['id'])}.npy")
            np.save(output_path,adversarial_sample)
            adversarial_sample = cv2.cvtColor(adversarial_sample, cv2.COLOR_BGR2RGB)
            output_path = os.path.join(
                patch_images_path, f"{str(image_dict['id'])}.jpeg")
            cv2.imwrite(output_path,adversarial_sample)

    def save_benign_samples(self, images_dicts, benign_image_path):
        """
        Function that saves (write) benign samples in numpy format.
        :param images_dicts: req. list of dictionary containing the adversarial images and additional
        information regarding them.
        :param benign_image_path: req. str. Path to save the images.
        :return: --
        """
        for image_dict in images_dicts:
            benign_sample = image_dict['image']
            output_path = os.path.join(
                benign_image_path, f"{str(image_dict['id'])}.npy")
            np.save(output_path,benign_sample)

    def save_prediction(self, images_dicts,adv_dec_path):
        """
        Function that plot the predictions on the original images.
        :param prediction: required. dictionary representing the prediction.
        :param x: required. 3D numpy array. The original image.
        :param x_name: required. string. The image id.
        :return: Plots the predictions on the original images and save it on
        predefined path.
        """
        for image_dict in images_dicts:
            frame = image_dict['adversarial_image']
            preds = image_dict['prediction'][0]
            for i in range(len(preds['boxes'])):
                if float(preds['scores'][i])>0.8:
                    x1 = int(preds['boxes'][i][0])
                    y1 = int(preds['boxes'][i][1])
                    x2 = int(preds['boxes'][i][2])
                    y2 = int(preds['boxes'][i][3])
                    detected_object = cv2.rectangle(frame, (x1, y1), (x2, y2),
                                                    (0, 255, 0), 3)
                    frame = cv2.putText(detected_object, self.data_util.index_to_label(int(preds['labels'][i])),
                                        (x1, y1),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 255, 0), 2)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            filename = os.path.join(adv_dec_path,
                                    f"{str(image_dict['id'])}.jpg")
            cv2.imwrite(filename, frame)

    def evaluate_predictions(self, image_dicts):
        """
        Function that evaluate model predictions.
        :param image_dicts: req. list of dictionary containing the adversarial images and additional
        information regarding them.
        :return: print to the screen the evaluation results.
        """
        coco_detections = self.process_preds_for_eval(image_dicts)
        self.data_util.eval(coco_detections,'bbox',self.put_patch_on)

    def process_preds_for_eval(self,image_dicts):
        """
        Function that process the image dictionary to be suited for evaluation.
        :param image_dicts: req. list of dictionary containing the adversarial images and additional
        information regarding them.
        :return: processed image dictionaries.
        """
        preds_list = []
        for image_dict in tqdm(image_dicts, desc="Processing predictions for evaluation"):
            preds = image_dict['prediction'][0]
            for i in range(len(preds['boxes'])):
                if float(preds['scores'][i]) > 0.8:
                    pred_dict = {}
                    pred_dict['image_id'] = image_dict['id']
                    pred_dict['category_id'] = int(preds['labels'][i])
                    pred_dict['bbox'] = self.transform_bbox_to_yolo_formant(preds['boxes'][i])
                    pred_dict['score'] = preds['scores'][i]
                    preds_list.append(pred_dict)

        return self.data_util.coco.loadRes(preds_list)

    def transform_bbox_to_yolo_formant(self, bbox_list):
        """
        Transform coordinates from (x1,y1,x2,y2) to (x1,y1,w,h).
        :param bbox_list: req. list. bounding box to be transformed.
        :return: transformed bounding box.
        """
        x1 = bbox_list[0]
        y1 = bbox_list[1]
        w = bbox_list[2] - x1
        h = bbox_list[3] - y1
        return [x1,y1,w,h]


    def evaluate_object_detector(self):
        """
        Main function to evaluate object detector model on adversarial images.
        1. Load benign images from MS COCO dataset.
        2. Placed the adversarial patch on the images.
        3. Evaluate the object detection model on the adversarial images.
        :return: Evaluation of the object detection model on the adversarial images.
        """
        output_path, benign_images_path, patch_images_path, adv_dec_path, plots = \
            self.create_output_folders()
        image_dicts = self.data_util.load_coco_dataset_from_ids_list()
        for image in tqdm(image_dicts,desc ="MS coco attack evaluation"):
            image['adversarial_image'] = np.array(self.apply_patch_to_image(image))
            image['prediction'] = self.predict(image['adversarial_image'])
        self.save_adversarial_samples(image_dicts,patch_images_path)
        self.save_benign_samples(image_dicts,benign_images_path)
        self.save_prediction(image_dicts,adv_dec_path)
        self.evaluate_predictions(image_dicts)

def classfication_from_scenes_demo(data_dir,annotations_file_path, ids_list_path,patch_path,
                                          use_mmdetection = False,target_model_path = None):
    """
    Demo function that uses "evaluate_object_detector" function to evaluate object detection model on adversarial images.
    :param data_dir: req. str. path to MS COCO images.
    :param annotations_file_path: req. str. path to MS COCO annotation file.
    :param ids_list_path: req. str. path to ids file which contain subset of exaimned images from MS COCO dataset.
    :param patch_path: req. str. Path where the patch is located.
    :param use_mmdetection: opt. bool. Whether to use object detection model from MM detection package.
    :param target_model_path: opt. str. The path to the target model ( given only if using the object detection
    model from MM detection package)
    :return: Evaluation of the target model.
    """
    patch_target_classes = ['apple', 'banana']
    for patch_target_class in patch_target_classes:
        put_patch_on = ['orange', 'pizza', 'banana', 'apple']
        ids_list_path_target_class = f"{ids_list_path}/{patch_target_class}/ids_list.txt"
        patch_path_target_class = f"{patch_path}/{patch_target_class}_patch/patch.npy"
        put_patch_on.remove(patch_target_class)
        evaluator = digital_attack_evaluation(data_dir, annotations_file_path, ids_list_path_target_class,
                                              patch_path_target_class, put_patch_on, use_mmdetection=use_mmdetection,
                                              target_model_path=target_model_path)
        evaluator.evaluate_object_detector()


