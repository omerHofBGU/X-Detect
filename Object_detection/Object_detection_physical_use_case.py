import os
import json
from typing import List
import numpy as np
import imagezmq
import time as tm
import torch
import cv2
import pathlib
from collections import Counter
from skimage.color import rgba2rgb
from skimage.io import imread
from multiprocessing import Pool
from torch.utils.data import DataLoader,Dataset
from .transformations import ComposeSingle, FunctionWrapperSingle, normalize_01
from pathlib import Path
from pytorch_lightning import seed_everything
import tqdm
from .Faster_RCNN_util import get_faster_rcnn_resnet50_fpn
import shutil
import pandas as pd
import datetime
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector,show_result_pyplot

"""
This module is for generating target model predictions on the physical use case. 
"""

# Super Store dataset class
SUPER_STORE_INSTANCE_CATEGORY_NAMES = {
    1: 'Agnesi Polenta',
    2: 'Almond Milk',
    3: 'Snyders',
    4: 'Calvin Klein',
    5: 'Dr Pepper',
    6: 'Flour',
    7: 'Groats',
    8: 'Jack Daniels',
    9: 'Nespresso',
    10: 'Oil',
    11: 'Paco Rabanne',
    12: 'Pixel4',
    13: 'Samsung_s20',
    14: 'Greek Olives',
    15: 'Curry Spice',
    16: 'Chablis Wine',
    17: 'Lindor',
    18: 'Piling Sabon',
    19: 'Tea',
    20: 'Versace',
    21: 'Adversarial Patch'
}

class evaluate_target_model():

    def __init__(self,target_model_path,number_of_classes,output_path="",
                 mmdetection_model= False,dataset_path=None,custom_classes = False):
        super().__init__()
        self.custom_classes = custom_classes
        # Upload the dataset an transform it to torch dataloader
        self.dataloader = self.upload_dataset(dataset_path)
        # upload the target model from a given path
        self.mmdetection_model = mmdetection_model
        self.target_model = self.upload_model(target_model_path,number_of_classes)
        if output_path!="":
            self.output_path = output_path
            self.output_path = self.output_path+"/Attack evaluation"
            Path(self.output_path).mkdir(parents=True, exist_ok=True)
        self.prediction_dict = {'Patch_id':[], 'Video_path':[], 'Attack_type':[], 'True_class':[], 'Target_class':[],
                                     'Target_model_pred':[], 'Target_attack_succeed':[],'Cheap_attack_succeed':[],
                                     'Untarget_attack_succeed':[]}
        self.evaluation_dict = {'Patch_id':[],'Attack_type':[],'Patch_class':[], 'Target_class_list_rate':[], 'Cheap_list_rate':[], 'Untarget_list_rate':[],
                                     'Target_class_frames_rate':[],'Cheap_frames_rate':[],'Untarget_frames_rate':[]}



    def upload_dataset(self, dataset):
        """
        Wrapper function for uploading Super Store dataset from a given root
        path.
        :param dataset: required. string. root path.
        :return: torch dataloader containing images as numpy array.
        """
        # transformations
        transforms = self.transformations()

        if isinstance(dataset, str):
            test = self.get_images_paths(dataset)
            # create dataset and dataloader
            dataset = ObjectDetectionDatasetSingle(inputs=test,
                                                   transform=transforms,
                                                   use_cache=False)
        else:
            test = dataset
            dataset = ObjectDetectionDatasetSingleFromNumpy(inputs=test,
                                                            transform=transforms,
                                                            use_cache=False)


        dataloader_prediction = DataLoader(dataset=dataset,
                                           batch_size=1,
                                           shuffle=False,
                                           num_workers=0,
                                           collate_fn=self.collate_single)
        return dataloader_prediction

    def get_images_paths(self,dataset_path):
        """
        Function that returns a list of images path from a given root path.
        :param dataset_path: required. string. root path.
        :return: a list of images path from a given root path.
        """
        roots = os.listdir(dataset_path)
        test = []
        if self.custom_classes:
            roots = ['Calvin Klein','Chablis Wine','Jack Daniels',
                     'Lindor','Nespresso','Paco Rabanne','Piling Sabon',
                     'Pixel4','Samsung_s20','Versace']
        for iter,class_name in enumerate(roots):
            root = pathlib.Path(os.path.join(dataset_path, class_name))
            test += self.get_filenames_of_path(root / 'test')
        return test

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

    def get_filenames_of_path(self, path: pathlib.Path, ext: str = "*") -> List[
        pathlib.Path]:
        """
        Returns a list of files in a directory/path. Uses pathlib.
        """
        filenames = [file for file in path.glob(ext) if file.is_file()]
        filenames = filenames[:15]
        assert len(filenames) > 0, f"No files found in path: {path}"
        return filenames

    def upload_model(self,target_model_path,number_of_classes):
        """
        Wrapper function for uploading a object detection model from a given
        path.
        :param target_model_path: required. string. A path to the target model.
        :param number_of_classes: required. int. The number of classes the
        model was trained on.
        :return: pytorch object detection model in inference mode.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if not self.mmdetection_model:
            model_state_dict = torch.load(target_model_path)
            model = get_faster_rcnn_resnet50_fpn(num_classes=number_of_classes)
            # load weights
            model.load_state_dict(model_state_dict)
            return model.eval()
        else:
            model = init_detector(target_model_path[1], target_model_path[0], device=device)
            return model


    def collate_single(self,batch):
        """
        collate function for the ObjectDetectionDataSetSingle.
        Only used by the dataloader.
        """
        x = [sample["x"] for sample in batch]
        x_name = [sample["x_name"] for sample in batch]
        return x, x_name


    def test_set_prediction(self, confidence_threshold=0.5, plot=False):
        """
        Function for generating predictions on the provided dataset using
        the target model.
        :param confidence_threshold: optional. float. default is 0.5.
        :param plot: optional. bool. If true, plot the predicitons over the
        original images.
        :return: dictionary containing the images id as key and the
        predictions as a value.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else
                              "cpu")
        if not Path(self.output_path).is_dir():
            Path(self.output_path).mkdir(parents=True, exist_ok=True)
        prediction_dict = {}
        model = self.target_model.to(device)
        for sample in self.dataloader:
            x, x_name = sample
            x = torch.stack(x).to(device)
            with torch.no_grad():
                result = model(x)
                try:
                    if result[0]['scores'][0]:
                        if result[0]['scores'][0] > confidence_threshold:
                            prediction_dict[x_name[0].split(".")[0]] = result[0]
                            if (plot):
                                self.plot_prediction(result,x,x_name)
                except:
                    print("An exception occurred")
                    continue
        return prediction_dict

    def plot_prediction(self, prediction, x, x_name):
        """
        Function that plot the predictions on the original images.
        :param prediction: required. dictionary representing the prediction.
        :param x: required. 3D numpy array. The original image.
        :param x_name: required. string. The image id.
        :return: Plots the predictions on the original images and save it on
        predefined path.
        """
        x1 = int(prediction[0]['boxes'][0][0])
        y1 = int(prediction[0]['boxes'][0][1])
        x2 = int(prediction[0]['boxes'][0][2])
        y2 = int(prediction[0]['boxes'][0][3])
        img = x[0].permute(1,2,0).cpu().numpy()*255
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detected_object = cv2.rectangle(img, (x1, y1), (x2, y2),
                                        (0, 255, 0), 3)
        frame = cv2.putText(detected_object, SUPER_STORE_INSTANCE_CATEGORY_NAMES
        [int(prediction[0]['labels'][0])], (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        filename = os.path.join(self.output_path,
                                f'{x_name[0].split(".")[0]}.jpg')
        cv2.imwrite(filename, frame)

    def intersection_over_union(self,
                                boxes_preds, boxes_labels,
                                box_format="midpoint"):
        """
        Calculates intersection over union
        Parameters:
            boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
            boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
            box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
        Returns:
            tensor: Intersection over union for all examples
        """

        # Slicing idx:idx+1 in order to keep tensor dimensionality
        # Doing ... in indexing if there would be additional dimensions
        # Like for Yolo algorithm which would have (N, S, S, 4) in shape
        if box_format == "midpoint":
            box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
            box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
            box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
            box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
            box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
            box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
            box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
            box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

        elif box_format == "corners":
            box1_x1 = boxes_preds[..., 0:1]
            box1_y1 = boxes_preds[..., 1:2]
            box1_x2 = boxes_preds[..., 2:3]
            box1_y2 = boxes_preds[..., 3:4]
            box2_x1 = boxes_labels[..., 0:1]
            box2_y1 = boxes_labels[..., 1:2]
            box2_x2 = boxes_labels[..., 2:3]
            box2_y2 = boxes_labels[..., 3:4]

        x1 = torch.max(box1_x1, box2_x1)
        y1 = torch.max(box1_y1, box2_y1)
        x2 = torch.min(box1_x2, box2_x2)
        y2 = torch.min(box1_y2, box2_y2)

        # Need clamp(0) in case they do not intersect, then we want intersection to be 0
        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
        box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

        return intersection / (box1_area + box2_area - intersection + 1e-6)

    def mean_average_precision(self, pred_boxes, true_boxes,
                               iou_threshold=0.5, box_format="corners",
                               num_classes=21
                               ):
        """
        Calculates mean average precision
        Parameters:
            pred_boxes (list): list of lists containing all bboxes with each bboxes
            specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
            true_boxes (list): Similar as pred_boxes except all the correct ones
            iou_threshold (float): threshold where predicted bboxes is correct
            box_format (str): "midpoint" or "corners" used to specify bboxes
            num_classes (int): number of classes
        Returns:
            float: mAP value across all classes given a specific IoU threshold
        """

        # list storing all AP for respective classes
        average_precisions = []

        # used for numerical stability later on
        epsilon = 1e-6

        for c in range(num_classes):
            detections = []
            ground_truths = []

            # Go through all predictions and targets,
            # and only add the ones that belong to the
            # current class c
            for detection in pred_boxes:
                if detection[1] == c:
                    detections.append(detection)

            for true_box in true_boxes:
                if true_box[1] == c:
                    ground_truths.append(true_box)

            # find the amount of bboxes for each training example
            # Counter here finds how many ground truth bboxes we get
            # for each training example, so let's say img 0 has 3,
            # img 1 has 5 then we will obtain a dictionary with:
            # amount_bboxes = {0:3, 1:5}
            amount_bboxes = Counter([gt[0] for gt in ground_truths])

            # We then go through each key, val in this dictionary
            # and convert to the following (w.r.t same example):
            # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
            for key, val in amount_bboxes.items():
                amount_bboxes[key] = torch.zeros(val)

            # sort by box probabilities which is index 2
            detections.sort(key=lambda x: x[2], reverse=True)
            TP = torch.zeros((len(detections)))
            FP = torch.zeros((len(detections)))
            total_true_bboxes = len(ground_truths)

            # If none exists for this class then we can safely skip
            if total_true_bboxes == 0:
                continue

            for detection_idx, detection in enumerate(detections):
                # Only take out the ground_truths that have the same
                # training idx as detection
                ground_truth_img = [
                    bbox for bbox in ground_truths if bbox[0] == detection[0]
                ]

                num_gts = len(ground_truth_img)
                best_iou = 0

                for idx, gt in enumerate(ground_truth_img):
                    iou = self.intersection_over_union(
                        torch.tensor(detection[3:]),
                        torch.tensor(gt[3:]),
                        box_format=box_format,
                    )

                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

                if best_iou > iou_threshold:
                    # only detect ground truth detection once
                    if amount_bboxes[detection[0]][best_gt_idx] == 0:
                        # true positive and add this bounding box to seen
                        TP[detection_idx] = 1
                        amount_bboxes[detection[0]][best_gt_idx] = 1
                    else:
                        FP[detection_idx] = 1

                # if IOU is lower then the detection is a false positive
                else:
                    FP[detection_idx] = 1

            TP_cumsum = torch.cumsum(TP, dim=0)
            FP_cumsum = torch.cumsum(FP, dim=0)
            recalls = TP_cumsum / (total_true_bboxes + epsilon)
            precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
            precisions = torch.cat((torch.tensor([1]), precisions))
            recalls = torch.cat((torch.tensor([0]), recalls))
            # torch.trapz for numerical integration
            average_precisions.append(torch.trapz(precisions, recalls))

        return sum(average_precisions) / len(average_precisions)
    def get_dict(self):
        """
        Create an empty dict for SUPER STORE dataset predictions.
        :return: empty dict.
        """
        self.prediction_dictionary = {
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
            20: 0,
            21:0,
        }

    def get_expensive_items_dict(self):
        """
        Create dict containing only expensive items from SUPER STORE dataset.
        :return:dict containing only expensive items
        """
        return {
            'Calvin Klein':[],
            'Jack Daniels':[],
            'Nespresso':[],
            'Paco Rabanne':[],
            'Pixel4':[],
            'Samsung_s20':[],
            'Chablis Wine':[],
            'Lindor':[],
            'Piling Sabon':[],
            'Versace':[]
        }

    def get_cheap_items_id(self):
        """
        :return: id's of cheap items in SUPER STORE dataset
        """
        return [1,2,3,5,6,7,10,14,15,19]

    def init_video(self,video_name):
        """
        Video initialization, coding configuration and frame rate.
        :param video_name: name of the video to be saved.
        :return:
        """
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(f'{video_name}.avi', fourcc, 30.0, (640, 480))
        return out

    def transformations(self):
        """
        Image transformations, for target model.
        :return: transformations functions.
        """
        return ComposeSingle([
            FunctionWrapperSingle(np.moveaxis, source=-1, destination=0),
            FunctionWrapperSingle(normalize_01)
        ])

    def init_real_time_detection_dataset(self,inputs,batch_size = 1):
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

    def init_real_time_variables(self,i, curr_output_path=None):
        """
        Variables initialization for real time classification.
        :param i: video number, for output folder path.
        :return: variables for real time detection.
        """
        # initialize the ImageHub object
        if curr_output_path==None:
            curr_output_path= f'{self.output_path}/{i}'
        clean_1 = self.init_video(f'{curr_output_path}/clean_1')
        processed_1 = self.init_video(f'{curr_output_path}/processed_1')
        clean_2 = self.init_video(f'{curr_output_path}/clean_2')
        processed_2 = self.init_video(f'{curr_output_path}/processed_2')
        total_clean = self.init_video(f'{curr_output_path}/total_clean')
        first_frame = True
        open_list = False
        frames_len = 0
        frame_decision = 0
        frame_decision_list = []
        frame_decision_list_per_item = []
        self.shopping_list = []
        self.classification_vector = []
        self.get_dict()
        export_frames = []
        processed_frame_exist = False
        return clean_1,processed_1,clean_2,processed_2,total_clean, \
               first_frame,open_list, first_frame, frames_len,frame_decision, \
               frame_decision_list,frame_decision_list_per_item, \
               processed_frame_exist,export_frames


    def detected_object(self,result):
        """
        Check if there is an object in the frame (if the best object score is greater than 0.3).
        :param result: req. a target model prediction.
        :return: Boolean stating if there is an object in a given frame.
        """
        if len(result['scores']) > 0:
            if result['scores'][0] > 0.3:
                return True
        return False

    def extract_points(self,bounding_box):
        """
        Helper function that extract the bounding box coordinates.
        :param bounding_box: req. list of tensors containing the bounding box
        in faster rcnn format (x1,y1,x2,y2)
        :return: The extracted bounding box coordinates as separated ints.
        """
        x1 = int(bounding_box[0])
        y1 = int(bounding_box[1])
        x2 = int(bounding_box[2])
        y2 = int(bounding_box[3])
        return x1, y1, x2, y2

    def draw_prediction(self,frame, x1, y1, x2, y2, classification):
        """
        Help function that draws the bounding box over the detected object.
        :param frame: req. numpy array representing an image.
        :param x1: req. int of x1.
        :param y1:  int of y1.
        :param x2:  int of x2.
        :param y2:  int of y2.
        :param classification: The target model predicted class.
        :return: The frame with the object bounding box.
        """
        detected_object = cv2.rectangle(frame, (x1, y1), (x2, y2),
                                        (0, 255, 0), 3)

        frame = cv2.putText(detected_object, classification,
                            (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)
        return frame

    def process_prediction_dict(self):
        """
        dictionary processing function for the prediction dictionary
        which normalize all the predictions to range[0,1], with sum of 1.
        :return: normalized prediction dict.
        """
        processed_dict = {}
        for key in self.last_prediction_dictionary:
            processed_dict[SUPER_STORE_INSTANCE_CATEGORY_NAMES[key]] = float(
                self.last_prediction_dictionary[key])
        factor = 1.0 / sum(processed_dict.values())
        for k in processed_dict:
            processed_dict[k] = processed_dict[k] * factor
        return processed_dict

    def update_list(self,curr_time, pred, first_frame_time):
        """
        Helper function to real time detection that update the shopping list.
        :param curr_time: req. time instance showing the current time,
        used for measuring the prediction time.
        :param pred: req. item's id of the target model classification.
        :param first_frame_time: req. time instance of the first frame.
        :return: updated list and initialize variables.
        """
        time0 = curr_time
        self.last_prediction_dictionary = self.prediction_dictionary
        processed_dict = self.process_prediction_dict()
        pred_confidence = processed_dict[SUPER_STORE_INSTANCE_CATEGORY_NAMES[int(pred)]]
        print(SUPER_STORE_INSTANCE_CATEGORY_NAMES[int(pred)],pred_confidence)
        self.shopping_list.append(SUPER_STORE_INSTANCE_CATEGORY_NAMES[int(pred)])
        first_frame = True
        open_list = False
        frame_decision = 0
        frame_decision_list_per_item = []

        self.prediction_dictionary = dict.fromkeys(
            self.prediction_dictionary, 0)
        return time0, first_frame, open_list,frame_decision,frame_decision_list_per_item

    def process_prediction(self,pred,frame):
        """
        Help function for prediction processing - extracting bounding
        box and drawing it on the frame.
        :param pred: req. faster rcnn prediction.
        :param frame: req. numpy array representing a frame.
        :return: a processed prediction.
        """
        x1, y1, x2, y2 = self.extract_points(pred['boxes'][0])
        classification = SUPER_STORE_INSTANCE_CATEGORY_NAMES[
            int(pred['labels'][0])]
        processed_frame = self.draw_prediction(frame, x1, y1, x2, y2,
                                               classification)
        self.classification_vector.append(classification)
        return processed_frame

    def read_frames(self, path):
        """
        Function that reads the frame from input folders. used in the video classification function of the
        video's evaluation set. .
        :param path: req. str. path to the frames folder.
        :return: The frames in a numpy array.
        """
        frames = []
        first_file = os.listdir(path)[0]
        if first_file.split(".")[1] == "npy":
            for frame in os.listdir(path):
                frames.append(np.load(os.path.join(path, frame)))
        else:
            for frame in os.listdir(path):
                frames.append(cv2.imread(os.path.join(path, frame)))
        return frames

    def write_frame(self,rpiName,clean_1,export_frame,processed_frame_exist,
                    processed_1,processed_frame,clean_2,processed_2,total_clean):
        """
        Function that write frames to disk.
        :param rpiName: req. str. the camera used ("camera_1" or "camera_2").
        :param clean_1: req. cv2.VideoWriter instance that record and save a video of the right camera.
        :param export_frame: req. numpy format. the frame to be exported (saved).
        :param processed_frame_exist: req. bool. If a processed frame exist (a frame with predictions).
        :param processed_1: req. cv2.VideoWriter instance that record and save a video of the right camera of processed images.
        :param processed_frame: req. numpy format. the processed frame be exported (saved).
        :param clean_2: req. cv2.VideoWriter instance that record and save a video of the right camera.
        :param processed_2: req. cv2.VideoWriter instance that record and save a video of the left camera of processed images.
        :param total_clean: req. cv2.VideoWriter instance that record and save a video of the both cameras.
        :return: --
        """
        if rpiName == 'camera_1':
            clean_1.write(export_frame)
            if processed_frame_exist:
                processed_1.write(processed_frame)
                processed_frame_exist = False
            else:
                processed_1.write(export_frame)
        else:
            clean_2.write(export_frame)
            if processed_frame_exist:
                processed_2.write(processed_frame)
                processed_frame_exist = False
            else:
                processed_2.write(export_frame)
        total_clean.write(export_frame)
        return processed_frame_exist

    def write_shopping_list(self,frame_decision_list,id):
        """
        Function that write an item to the shopping list and initialize several variables corresponding to that.
        :param frame_decision_list: req. float. The confidence for the predictions of each item in the list.
        :param id: int. the id of the video saved.
        :return: --
        """
        with open(f'{self.output_path}/{id}/shopping_list.txt', 'w') as f:
            for item, frames in zip(self.shopping_list, frame_decision_list):
                f.write(f'{frames} frames prediction: {item}')
                f.write('\n')
        processed_dict = self.process_prediction_dict()
        with open(f'{self.output_path}/{id}/confidence_dict.json',
                  "w") as outfile:
            json.dump(processed_dict, outfile)
        return

    def real_time_object_detection(self):
        """
        Main function for performing real time object detection for smart
        shopping cart.
        :return: artifacts for each experiment - output videos for each
        cameras, shopping list, frames with objects.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else
                              "cpu")
        print("Receiving frame...")
        imageHub = imagezmq.ImageHub(open_port='tcp://*:5555')
        shutil.rmtree(f'{self.output_path}')
        Path(f'{self.output_path}').mkdir(parents=True, exist_ok=True)
        for i in range(15):
            id = int(np.random.rand(1)*100 + (100*i))
            Path(f'{self.output_path}/{id}').mkdir(parents=True, exist_ok=True)
            Path(f'{self.output_path}/{id}/frames').mkdir(parents=True,
                                                          exist_ok=True)
            print(f"experiment {id}")
            clean_1, processed_1, clean_2,processed_2,total_clean, first_frame, open_list, first_frame, frames_len,\
            frame_decision,frame_decision_list,frame_decision_list_per_item, processed_frame_exist, export_frames \
                =self.init_real_time_variables(id)
            processed_frame = None
            time0 = tm.time()
            self.target_model = self.target_model.to(device)
            while frames_len<100:
                (rpiName, frames) = imageHub.recv_image()
                imageHub.send_reply(b'OK')
                frames_len+=1
                if frames_len==20:
                    print("Insert item!")
                dataloader = self.init_real_time_detection_dataset(frames)
                for iter, sample in enumerate(dataloader):
                    with torch.no_grad():
                        frame = sample.to(device)
                        result = self.target_model(frame)
                    frame = frames[iter]
                    export_frame = np.copy(frame)
                    if self.detected_object(result[0]):
                        processed_frame = self.process_prediction(result[0],frame)
                        processed_frame_exist = True
                        curr_time = tm.time()
                        if (curr_time - time0) > 1.5:
                            if first_frame:
                                first_frame = False
                                first_frame_time = tm.time()
                            frame_decision +=1
                            frame_decision_list_per_item.append(frames_len)
                            self.prediction_dictionary[int(result[0]['labels'][
                                                               0])] \
                                += result[0]['scores'][0]
                            export_frames.append(export_frame)
                            if curr_time - first_frame_time > 0.5:
                                pred = max(self.prediction_dictionary,
                                           key=self.prediction_dictionary.get)
                                open_list = True
                            if open_list:
                                frame_decision_list.append(frame_decision_list_per_item)
                                time0, first_frame, open_list,frame_decision, \
                                frame_decision_list_per_item = \
                                    self.update_list(curr_time,pred,
                                                     first_frame_time)

                    processed_frame_exist = self.write_frame(rpiName,clean_1,export_frame,processed_frame_exist,
                                                             processed_1,processed_frame,clean_2,processed_2,total_clean)

            self.write_shopping_list(frame_decision_list,id)
            for idx,export_frame in enumerate(export_frames):
                with open(f'{self.output_path}/{id}/frames/{idx}.npy',
                          'wb') as f:
                    np.save(f, export_frame)

        for i in range(2):
            (rpiName, frames) = imageHub.recv_image()
            imageHub.send_reply(b'DONE')


        imageHub.close()
        clean_1.release()
        clean_2.release()
        processed_1.release()
        processed_2.release()
        cv2.destroyAllWindows()
        return


    def classify_video(self,video_folder,true_item,target_class):
        """
        Main function for classify an item inserted to the cart from a given video.
        :param video_folder: req. str. path to the folder containing the video.
        :param true_item: req. str. name of the true item inserted into the cart.
        :param target_class: req. str. name of the patch target class.
        :return: statistics of the performance of the model identfing the correct item inserted into the cart.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else
                              "cpu")
        self.target_model = self.target_model.to(device)
        frames = self.read_frames(f'{video_folder}/frames')
        frame_counter,target_class_frames_counter, \
        misclassification_frames_counter,target_class_id, true_item_id, \
        cheap_rate,cheap_list,cheap_frames_counter= \
            self.init_variables_for_video_classification(true_item,target_class)

        for frame in frames:
            if not self.mmdetection_model:
                result = self.predict_faster_rcnn(frame,device)
            else:
                result = self.predict_other_models(frame,device)
                # result = self.predict_yolo(frame,device)
            if self.detected_object(result[0]):
                frame_counter += 1
                classification_label = int(result[0]['labels'][0])
                if classification_label == target_class_id:
                    target_class_frames_counter += 1
                if classification_label != true_item_id:
                    misclassification_frames_counter += 1
                self.prediction_dictionary[int(result[0]['labels'][0])] \
                    += result[0]['scores'][0]
                if classification_label in cheap_list:
                    cheap_frames_counter+=1

        if frame_counter==0:
            # print("No item was detected in the video")
            return 0, 0, cheap_rate, \
                   target_class_frames_counter, misclassification_frames_counter, \
                   cheap_frames_counter, frame_counter

        pred = max(self.prediction_dictionary,
                   key=self.prediction_dictionary.get)
        classification = SUPER_STORE_INSTANCE_CATEGORY_NAMES[int(pred)]
        # print(f"True class:{true_item}, Classification: {classification}")
        self.prediction_dict['Video_path'].append(video_folder)
        self.prediction_dict['True_class'].append(true_item)
        self.prediction_dict['Target_class'].append(target_class)
        self.prediction_dict['Target_model_pred'].append(classification)
        if classification == target_class:
            target_class_rate = 1
            self.prediction_dict['Target_attack_succeed'].append('True')
        else:
            # print(f'attack failed for:{video_folder}')
            target_class_rate = 0
            self.prediction_dict['Target_attack_succeed'].append('False')
        if classification != true_item:
            misclassification_rate = 1
            self.prediction_dict['Untarget_attack_succeed'].append('True')
        else:
            misclassification_rate = 0
            self.prediction_dict['Untarget_attack_succeed'].append('False')


        classification_id = [k for k, v in \
                             SUPER_STORE_INSTANCE_CATEGORY_NAMES.items() if v ==
                             classification][0]
        if classification_id in cheap_list:
            cheap_rate = 1
            self.prediction_dict['Cheap_attack_succeed'].append('True')
        else:
            self.prediction_dict['Cheap_attack_succeed'].append('False')
            cheap_rate = 0

        return target_class_rate, misclassification_rate, cheap_rate, \
               target_class_frames_counter, misclassification_frames_counter, \
               cheap_frames_counter,frame_counter

    def predict_faster_rcnn(self, frame,device):
        """
        Inference using Faster R-CNN model.
        :param frame: req. image in a numpy array.
        :param device: req. The device used (CPU/GPU).
        :return: prediction in a dictionary.
        """
        frame = np.expand_dims(frame, axis=0)
        dataloader = self.init_real_time_detection_dataset(frame)
        for iter, sample in enumerate(dataloader):
            with torch.no_grad():
                frame = sample.to(device)
                result = self.target_model(frame)
        return result

    def predict_other_models(self,frame,device):
        """
        Inference using models from mm detection package.
        :param frame: req. image in a numpy array.
        :param device: req. The device used (CPU/GPU).
        :return: prediction in a dictionary.
        """
        result = inference_detector(self.target_model, frame)
        # show_result_pyplot(self.target_model, frame, result,score_thr=0.05)
        result = self.create_faster_rcnn_result_dict_new(device, result)
        return result

    def create_faster_rcnn_result_dict_new(self,device,result):
        """
        Function that transform the result given by an object detector model taken from MMdetection package
        to Faster R-CNN format.
        :param result: req. list. result given by an object detector model taken from MMdetection
        :return: A dictionary presenting the result in Faster R-CNN format.
        """
        result_faster_rcnn_format = []
        result_dict = {'boxes': torch.empty(size=(0, 4), device=device),
                       'labels': torch.empty(size=(0,), device=device, dtype=torch.int64),
                       'scores': torch.empty(size=(0,), device=device)}
        class_prob = {}
        for idx,class_instance in enumerate(result):
            if len(class_instance)>0:
                class_prob[idx+1]=class_instance[0][4]
        if len(class_prob)>0:
            prediction_confidence = max(class_prob.values())
            if prediction_confidence>0.3:
                prediction_id = max(class_prob, key=class_prob.get)
                result_dict['labels'] = torch.tensor([prediction_id] ,device=device)
                result_dict['scores'] = torch.tensor( [prediction_confidence],device=device)
                result_dict['boxes'] = torch.tensor([result[prediction_id-1][0][:-1]])

        result_faster_rcnn_format.append(result_dict)
        return result_faster_rcnn_format

    def init_variables_for_video_classification(self,true_item,target_class):
        """
        Initialization of variables used in the video classification function.
        :param true_item: req. str. the class of the correct item.
        :param target_class: req. str. the class of the targeted item.
        :return: variables used in the video classification function.
        """
        frame_counter = 0
        target_class_frames_counter = 0
        cheap_frames_counter = 0
        self.get_dict()
        cheap_rate = 0
        cheap_list = self.get_cheap_items_id()
        misclassification_frames_counter = 0
        target_class_id = list(SUPER_STORE_INSTANCE_CATEGORY_NAMES.keys())[
            list(SUPER_STORE_INSTANCE_CATEGORY_NAMES.values()).index(
                target_class)]
        true_item_id = list(SUPER_STORE_INSTANCE_CATEGORY_NAMES.keys())[
            list(SUPER_STORE_INSTANCE_CATEGORY_NAMES.values()).index(
                true_item)]
        return frame_counter,target_class_frames_counter, \
               misclassification_frames_counter,target_class_id,true_item_id, \
               cheap_rate,cheap_list,cheap_frames_counter


    def video_classification_wrapper(self,videos_folder_path,adaptive_attack):
        """
        Wrapper function for "classify_video" function which go over on the video files from the evaluation set and
        measure the target model performance.
        :param videos_folder_path: req. str. Path to the videos folder.
        :param adaptive_attack: req. bool. Indicate if to use the videos of the adaptive attack.
        :return: Evaluation of the target model predictions.
        """
        seed_everything(41)
        patch_id =0
        for idx,fold in enumerate(os.listdir(videos_folder_path)):
            # Choose to evaluate on adaptive or non-adaptive attack videos
            if adaptive_attack:
                if fold!='Adaptive attack':
                    continue
            else:
                if fold=='Adaptive attack':
                    continue
            attack_fold = os.path.join(videos_folder_path,fold)
            for attack_type in os.listdir(attack_fold):
                attack_type_path = os.path.join(attack_fold,attack_type)
                for patch_class in tqdm(os.listdir(attack_type_path), desc=f"Predict patch in {attack_type}"):
                    # Reserve classes for Ad-YOLO
                    if patch_class=="Curry Spice" or patch_class=="Tea":
                        continue
                    patch_id+=1
                    target_class_list_rate = 0
                    misclassification_list_rate = 0
                    target_class_frames_rate = 0
                    misclassification_frames_rate = 0
                    cheap_frames_rate = 0
                    videos_counter = 0
                    cheap_list_rate = 0
                    frames_counter = 0
                    patch_folder = os.path.join(attack_type_path,patch_class)
                    for item_class in os.listdir(patch_folder):
                        video_folders = os.path.join(patch_folder,item_class)
                        for idx,video_folder in enumerate(os.listdir(
                                video_folders)):
                            video_folder = os.path.join(video_folders,video_folder)
                            target_class_rate, misclassification_rate, cheap_rate, \
                            target_class_frames, misclassification_frames,cheap_frames,frames_in_curr_video = \
                                self.classify_video(video_folder,item_class,patch_class)
                            if frames_in_curr_video==0:
                                continue
                            videos_counter += 1
                            self.prediction_dict['Patch_id'].append(patch_id)
                            self.prediction_dict['Attack_type'].append(attack_type)
                            target_class_list_rate+=target_class_rate
                            misclassification_list_rate+=misclassification_rate
                            cheap_list_rate+=cheap_rate
                            target_class_frames_rate+=target_class_frames
                            misclassification_frames_rate+=misclassification_frames
                            cheap_frames_rate+=cheap_frames
                            frames_counter+= frames_in_curr_video

                    # print(f'Patch class: {patch_class},target_class_list_rate: '
                    #       f'{target_class_list_rate/videos_counter}, '
                    #       f'misclassification_list_rate: '
                    #       f'{misclassification_list_rate/videos_counter}, '
                    #       f'cheap_list_rate: '
                    #       f'{cheap_list_rate/videos_counter}, '
                    #       f'target_class_frames_rate: '
                    #       f'{target_class_frames_rate/frames_counter}, '
                    #       f'misclassification_frames_rate:'
                    #       f'{misclassification_frames_rate/frames_counter}, '
                    #       f'cheap_frames_rate:{cheap_frames_rate/frames_counter}, '
                    #       f'total adversarial videos: {target_class_list_rate}')
                    self.evaluation_dict['Patch_id'].append(patch_id)
                    self.evaluation_dict['Attack_type'].append(attack_type)
                    self.evaluation_dict['Patch_class'].append(patch_class)
                    self.evaluation_dict['Target_class_list_rate'].append(target_class_list_rate/videos_counter)
                    self.evaluation_dict['Cheap_list_rate'].append(cheap_list_rate/videos_counter)
                    self.evaluation_dict['Untarget_list_rate'].append(misclassification_list_rate/videos_counter)
                    self.evaluation_dict['Target_class_frames_rate'].append(target_class_frames_rate/frames_counter)
                    self.evaluation_dict['Cheap_frames_rate'].append(cheap_frames_rate/frames_counter)
                    self.evaluation_dict['Untarget_frames_rate'].append(misclassification_frames_rate/frames_counter)

        prediction_df = pd.DataFrame.from_dict(self.prediction_dict)
        prediction_df.to_csv(f'{self.output_path}/predictions.csv', index=False)
        self.generate_final_evaluation_report()
        return prediction_df

    def generate_final_evaluation_report(self):
        """
        Generate final evaluation report using the "classify video" function result.
        :return: Save a csv file of the evaluation results.
        """
        evaluation_df = pd.DataFrame.from_dict(self.evaluation_dict)
        dpatch_res = evaluation_df.loc[evaluation_df['Attack_type'] =='Dpatch']
        robust_dpatch_res = evaluation_df.loc[evaluation_df['Attack_type'] =='Robust_dpatch']
        dpatch_stats = self.get_stats_for_attack_type(dpatch_res).to_frame()
        robust_dpatch_stats = self.get_stats_for_attack_type(robust_dpatch_res).to_frame()
        overall_evaluation_stats = self.get_stats_for_attack_type(evaluation_df,overall=True).to_frame()
        frames = [evaluation_df,dpatch_stats.T,robust_dpatch_stats.T, overall_evaluation_stats.T]
        result = pd.concat(frames)
        result.to_csv(f'{self.output_path}/evaluation.csv', index=False)

    def get_stats_for_attack_type(self, df_res,overall=False):
        """
        Help function for generating the final evaluation report.
        :param df_res: req. results in dataframe instance.
        :param overall: opt. bool. if the results focusing on all attacks (not a specific attack).
        :return:
        """

        mean = df_res[['Target_class_list_rate', 'Cheap_list_rate','Untarget_list_rate','Target_class_frames_rate','Cheap_frames_rate','Untarget_frames_rate']].mean()
        mean['Patch_id'] = 'xx'
        if overall:
            mean['Attack_type'] = 'Total'
        else:
            mean['Attack_type'] = df_res['Attack_type'].iloc[0]
        mean['Patch_class'] = 'xx'
        return mean


class ObjectDetectionDatasetSingle(Dataset):
    """
    Builds a dataset with images.
    inputs is expected to be a list of pathlib.Path objects.

    Returns a dict with the following keys: 'x', 'x_name'
    """

    def __init__(self,inputs,transform = None,use_cache = False,):
        self.inputs = inputs
        self.transform = transform
        self.use_cache = use_cache

        if self.use_cache:
            # Use multiprocessing to load images and targets into RAM
            with Pool() as pool:
                self.cached_data = pool.starmap(self.read_images, inputs)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        if self.use_cache:
            x = self.cached_data[index]
        else:
            # Select the sample
            input_ID = self.inputs[index]

            # Load input and target
            x = self.read_images(input_ID)

        # From RGBA to RGB
        if x.shape[-1] == 4:
            x = rgba2rgb(x)

        # Preprocessing
        if self.transform is not None:
            x = self.transform(x)  # returns a np.ndarray

        # Typecasting
        x = torch.from_numpy(x).type(torch.float32)

        return {"x": x, "x_name": self.inputs[index].name}

    @staticmethod
    def read_images(inp):
        return imread(inp)

class ObjectDetectionDatasetSingleFromNumpy(Dataset):
    """
    Builds a dataset with images.
    inputs is expected to be a list of pathlib.Path objects.

    Returns a dict with the following keys: 'x', 'x_name'
    """

    def __init__(
        self,
        inputs: np.array([]),
        transform: ComposeSingle = None,
        use_cache: bool = False,
    ):
        self.inputs = inputs
        self.transform = transform
        self.use_cache = use_cache

        if self.use_cache:
            # Use multiprocessing to load images and targets into RAM
            with Pool() as pool:
                self.cached_data = pool.starmap(self.read_images, inputs)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        if self.use_cache:
            x = self.cached_data[index]
        else:
            real_detection_format= False
            # Select the sample
            try :
                x = self.inputs[index]['x']
            except:
                real_detection_format = True
                x= self.inputs[index]

        # From RGBA to RGB
        if x.shape[-1] == 4:
            x = rgba2rgb(x)

        # Preprocessing
        if self.transform is not None:
            x = self.transform(x)  # returns a np.ndarray

        # Typecasting
        x = torch.from_numpy(x).type(torch.float32)
        if real_detection_format:
            return x
        return {"x": x, "x_name": self.inputs[index]['x_name'] }

    @staticmethod
    def read_images(inp):
        return imread(inp)


def real_time_detection_demo(target_model_path,classes,video_output_path,use_mmdetection):
    """
    Demo function for real time detection.
    :param target_model_path: req. str. path to the target model weights.
    :param classes:
    :param video_output_path: req. str. path which the output video will be saved on.
    :param use_mmdetection: req. bool. Whether the used model is from MM detection package.
    :return: The recorded video is saved in the output folder.
    """
    etm = evaluate_target_model(target_model_path,classes,video_output_path,use_mmdetection)
    etm.real_time_object_detection()

def classification_from_video_demo(target_model_path,classes,video_input_path,video_output_path,
                                   use_mmdetection_model, adaptive_attack):
    """

    :param target_model_path:  req. str. path to the target model weights.
    :param classes: req. int. The number of classes the target was trained on.
    :param video_input_path:  req. str. path of the videos.
    :param video_output_path: req. str. path which the output video will be saved on.
    :param use_mmdetection_model: req. bool. Whether the used model is from MM detection package.
    :param adaptive_attack: req. bool. Whether to use the dataset of the adaptive attacks.
    :return:
    """
    etm = evaluate_target_model(target_model_path, classes,video_output_path,use_mmdetection_model)
    prediction_df = etm.video_classification_wrapper(video_input_path,adaptive_attack)
    return prediction_df



