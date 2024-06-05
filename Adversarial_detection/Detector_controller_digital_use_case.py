import os
import numpy as np
import torch
import cv2
import sys
from art.estimators.object_detection import PyTorchFasterRCNN
from .object_extractor_detector import Prototypes,Pointrend_instance_segmentation, oe_detector_demo_ms_coco
from .scene_processing_detector import scene_processor_detector,frame_sm_detector_demo_ms_coco
from Datasets_util.MSCOCO_util import MS_COCO_util, COCO_INSTANCE_CATEGORY_NAMES,COCO_INSTANCE_CATEGORY_NAMES_80_CLASSES
import datetime
from pathlib import Path
from mmdet.apis import init_detector, inference_detector,show_result_pyplot

"""
Module for executing adversarial detection in the digital use case using the MS COCO dataset. 
"""
class detector_controller_MSCOCO():

    def __init__(self,MSCOCO_data_path, evaluation_data_dir,annotations_file_path,prototypes_path,
                 style_transfer_model_path,target_model_path = None,ad_yolo_path = None, output_path = ""):
        super().__init__()
        self.MSCOCO_data_path = MSCOCO_data_path
        self.annotations_file_path = annotations_file_path
        self.evaluation_data_dir = evaluation_data_dir
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.output_path = output_path
        if target_model_path:
            self.use_mmdetection = True
            self.target_model_path = target_model_path
        else:
            self.use_mmdetection = False

        if ad_yolo_path:
            self.use_ad_yolo = True
            self.ad_yolo_path = ad_yolo_path
        else:
            self.use_ad_yolo = False
        self.target_model = self.init_object_detector()
        self.init_detectors(prototypes_path,style_transfer_model_path)
        if output_path=="":
            self.output_path = self.create_output_folder()
        self.object_extraction_model_path = style_transfer_model_path
        self.scene_manipulation_eval_dict = self.create_sm_eval_dict()



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
                clip_values=(0, 255), channels_first=False, attack_losses=(
                    "loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg")
            )
            frcnn.model.to(self.device)
            frcnn.model.eval()
        else:
            frcnn = init_detector(self.target_model_path[1], self.target_model_path[0], device=self.device)
        if self.use_ad_yolo:
            self.ad_yolo_model = init_detector(self.ad_yolo_path[1], self.ad_yolo_path[0], device=self.device)
        return frcnn


    def get_dict(self):
        """
        Create dictionary that will store the model prediction on several classes from MS COCO dataset.
        :return:
        """
        return {
            'apple': 0,
            'banana': 0,
            'orange': 0,
            'pizza': 0
        }

    def init_detectors(self,prototypes_path,style_transfer_models_path):
        """
        Initialize adversarial detectors.
        :param prototypes_path: req. str. path for the prototype images.
        :param style_transfer_models_path: req. str. path to style transfer models.
        :return: --
        """
        self.prototypes = Prototypes(prototypes_path,style_transfer_models_path)
        self.prototypes.process_prototypes()
        self.pointrend_model = Pointrend_instance_segmentation(style_transfer_models_path)
        self.scene_manipulation_detector = scene_processor_detector(None, self.target_model,style_transfer_models_path,
                                                                    "digital",output_path=self.output_path)
        self.scene_manipulation_detector.create_image_processing_dict()
        self.object_extractor_output_path = os.path.join(self.output_path, "Object_extractor_detector")
        Path(self.object_extractor_output_path).mkdir(parents=True, exist_ok=True)

    def initialize_evaluation_variables(self):
        """
        initialize variables used for the evaluation.
        :return: --
        """
        self.sm_adversarial_detection = 0
        self.oe_adversarial_detection = 0
        self.vertical_ensemble_adversarial_detection = 0
        self.ensemble_adversarial_detection = 0
        self.ad_yolo_adversarial_detection = 0

        self.sm_robustness = 0
        self.oe_robustness = 0
        self.vertical_ensemble_robustness = 0
        self.ensemble_robustness = 0
        self.ad_yolo_robustness = 0

        self.oe_cannot_extract = 0
        self.sample_counter = 0

    def create_output_folder(self):
        """
        Function for crating output folders.
        :return: Create output folders.
        """
        time = datetime.datetime.now().strftime("%d-%m-%Y_%H;%M")
        output_path = f"Output/{time}"
        Path(output_path).mkdir(parents=True, exist_ok=True)
        output_path = f"Output/{time}/detector_results"
        Path(output_path).mkdir(parents=True, exist_ok=True)
        return output_path

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
                if class_prob[prediction_id]>0.2:
                    labels.append(prediction_id+1)
                    scores.append(class_prob[prediction_id])
                    boxes.append(self.get_most_suitable_object(result[prediction_id]))
            result_dict['labels'] = torch.tensor(labels, device=device)
            result_dict['scores'] = torch.tensor(scores, device=device)
            result_dict['boxes'] = torch.tensor(boxes)

        result_faster_rcnn_format.append(result_dict)
        return result_faster_rcnn_format

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
                # frames = frames.to(self.device)
                return self.target_model.predict(frames,batch_size=1)
        else:
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            result = inference_detector(self.target_model, frame)
            # show_result_pyplot(self.target_model, frame, result,score_thr=0.25)
            result = self.create_faster_rcnn_result_dict_new(result)
            return result

    def get_ad_yolo_pred(self, frame):
        """
        Prediction for Ad-YOLO adversarial detection method.
        :param frame: req. image in numpy format.
        :return: Dictionary with the prediction of Ad-YOLO adversarial detection method.
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = inference_detector(self.ad_yolo_model, frame)
        # show_result_pyplot(self.ad_yolo_model, frame, result, score_thr=0.50)
        result = self.create_faster_rcnn_result_dict_new(result)

        return result

    def create_sm_eval_dict(self):
        """
        Create an empty dictionary for scene processing detector evaluation.
        :return: empty dictionary.
        """
        scenes = {}
        scenes['blur'] = {}
        scenes['blur']['total_detection_score'] = 0
        scenes['blur']['adversarial_detection'] = 0
        scenes['style_transfer'] = {}
        scenes['style_transfer']['total_detection_score'] = 0
        scenes['style_transfer']['adversarial_detection'] = 0
        scenes['sharp'] = {}
        scenes['sharp']['total_detection_score'] = 0
        scenes['sharp']['adversarial_detection'] = 0
        scenes['darkness'] = {}
        scenes['darkness']['total_detection_score'] = 0
        scenes['darkness']['adversarial_detection'] = 0
        return scenes

    def load_dataset_from_path(self, data_dir):
        """
        Wrapper function that loads images from a specific folder by a folder path.
        :param data_dir: req. str. path for the folder containing the images.
        :return: ids and images as lists.
        """
        ids_list,images_list = [],[]
        ids_list += self.get_list_id_from_files(data_dir)
        images_list += self.load_images_from_path(data_dir)
        return ids_list,images_list

    def get_list_id_from_files(self,input_path):
        """
        Helper function for "load_dataset_from_path" that returns all the images names (ids) form a specific path.
        :param input_path: req. str. the folder input path.
        :return: images ids as list.
        """
        return [int(file.split(".")[0]) for file in os.listdir(input_path)]

    def load_images_from_path(self,input_path):
        """
        Helper function for "load_dataset_from_path" that loads all the images from a given path.
        :param input_path: req. str. the folder input path.
        :return: images as list
        """
        images_list = []
        ids_as_ints = [int(x.split(".")[0]) for x in os.listdir(input_path)]
        ids_as_ints.sort()
        for file_id in ids_as_ints:
            images_list.append(np.load(os.path.join(input_path,f'{str(file_id)}.npy')))
        return images_list

    def replace_images_image_dicts(self, image_dicts, images,ids_list):
        """
        Arrange the image dictionary.
        :param image_dicts: req. list of dictionary containing the adversarial images and additional
        information regarding them.
        :param images: req. list of images.
        :param ids_list: req. list of ids for the images list.
        :return: image_dicts rearranged.
        """
        ids_list.sort()
        for idx,id in enumerate(ids_list):
            for dict in image_dicts:
                if dict['id'] == id:
                    dict['tested_image'] = images[idx]
        return image_dicts

    def get_main_object_bbox_and_label(self, frame):
        """
        Locate the main object in a frame and return its bounding box and label.
        :param frame: req. image in numpy format.
        :return: The main object label and bounding box.
        """
        largest_index = 0
        biggest_bbox_area = 0
        for idx, detection in enumerate(frame['annotations']):
            if detection['category_id'] in self.object_list:
                bbox_area = detection['bbox'][2] * detection[
                    'bbox'][3]
                if biggest_bbox_area < bbox_area:
                    biggest_bbox_area = bbox_area
                    largest_index = idx

        main_obj_bbox = frame['annotations'][largest_index]['bbox']
        main_obj_bbox[2] = main_obj_bbox[0]+main_obj_bbox[2]
        main_obj_bbox[3] = main_obj_bbox[1]+main_obj_bbox[3]

        main_object_dict = {
            'label':COCO_INSTANCE_CATEGORY_NAMES[frame['annotations'][largest_index]['category_id']],
            'bbox': main_obj_bbox
        }

        return main_object_dict

    def get_total_pred(self, pred_dict):
        """
        Get the prediction with the highest probability.
        :param pred_dict: req. dictionary with all the classes and their probability.
        :return: the prediction with the highest probability
        """
        pred = max(pred_dict,
                   key=pred_dict.get)
        return pred

    def get_iou(self,bbox1,bbox2):
        """
        Function that returns the intersection over union of two bounding boxes.
        :param bbox1: req. list. bounding box 1.
        :param bbox2: req. list. bounding box 2.
        :return: The intersection over union of the two given bounding boxes.
        """
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bb2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        return iou


    def get_object_pred_by_bbox(self, main_object_bbox, target_model_preds,image = None,output_path = None,ad_yolo_model = False):
        """
        Get an object detection model prediction on a given bounding box area.
        :param main_object_bbox: req. the examined bounding box area.
        :param target_model_preds: req. the model predictions for the entire frame.
        :param image: opt. image in a numpy format.
        :param output_path: opt. str.
        :param ad_yolo_model: opt. bool. test on Ad-YOLO model.
        :return:
        """
        best_iou = 0
        best_idx = 0
        target_model_bbox_preds = target_model_preds[0]['boxes']
        target_model_confidence = target_model_preds[0]['scores']
        target_model_labels = target_model_preds[0]['labels']
        for idx,box_pred in enumerate(target_model_bbox_preds):
            if target_model_confidence[idx]> 0.2:
                curr_iou = self.get_iou(np.array(main_object_bbox),box_pred)
                if curr_iou>best_iou:
                    best_iou = curr_iou
                    best_idx = idx
        if not self.use_mmdetection and not ad_yolo_model:
            pred = COCO_INSTANCE_CATEGORY_NAMES[target_model_labels[best_idx]]
        else:
            pred = COCO_INSTANCE_CATEGORY_NAMES_80_CLASSES[target_model_labels[best_idx]]
        if pred!=self.patch_class:
            for idx, box_pred in enumerate(target_model_bbox_preds):
                if target_model_confidence[idx] > 0.4:
                    if self.is_contained(box_pred, main_object_bbox):
                        best_idx = idx
                        break
        if not self.use_mmdetection and not ad_yolo_model:
            pred = COCO_INSTANCE_CATEGORY_NAMES[target_model_labels[best_idx]]
        else:
            pred = COCO_INSTANCE_CATEGORY_NAMES_80_CLASSES[target_model_labels[best_idx]]
        # self.plot_main_object(image, target_model_bbox_preds[best_idx], pred, output_path)
        return pred

    def update_scene_processing_eval(self,item,target_model_pred):
        """
        Update the scene processing dictionary for each scene processing technique.
        :param item: req. str. the correct object.
        :param target_model_pred: req. str. the target model prediction.
        :return: ---
        """
        for key in self.scene_manipulation_detector\
                .prediction_for_image_processing_tech.keys():
            pred = max(
                self.scene_manipulation_detector
                    .prediction_for_image_processing_tech[key]['prediction_dict'],
                       key=self.scene_manipulation_detector
                    .prediction_for_image_processing_tech[key]['prediction_dict'].get)
            if pred == item:
                self.scene_manipulation_eval_dict[key]['total_detection_score']+=1
            if pred!=target_model_pred:
                self.scene_manipulation_eval_dict[key][
                    'adversarial_detection'] += 1


    def get_scene_manipulation_preds(self,frame,main_obj_bbox,img_id):
        """
        Get the scene processing detector classification.
        :param frame: req. image in a numpy format.
        :param main_obj_bbox: req. list. bounding box of the main object in the frame.
        :param img_id: req. int. the image id.
        :return: Scene processing classification and its probability dictionary.
        """
        frame_sm_detector = np.copy(frame)
        scene_manipulation_pred_dict = frame_sm_detector_demo_ms_coco(self.scene_manipulation_detector,
                                                                    frame_sm_detector, self.device,
                                                                    self.get_dict(),main_obj_bbox,
                                                                    img_id,self.use_mmdetection)
        scene_manipulation_pred = self.get_total_pred(scene_manipulation_pred_dict)
        return scene_manipulation_pred, scene_manipulation_pred_dict

    def get_object_extraction_pred(self,frame, main_object_bbox, true_item,target_model_pred,frame_id):
        """
        Get the object extraction detector classification.
        :param frame: req. image in a numpy format.
        :param main_object_bbox: req. list. bounding box of the main object in the frame.
        :param true_item: req. str. the correct item (used for evaluation).
        :param target_model_pred: req. str. the target model prediction.
        :return: Object extraction classification and its probability dictionary.
        """
        frame_oe_detector = np.copy(frame)
        frame_oe_detector = cv2.cvtColor(frame_oe_detector, cv2.COLOR_BGR2RGB)
        object_extraction_final_pred_dict = self.get_dict()
        object_extraction_pred_dict = oe_detector_demo_ms_coco(self.prototypes,self.pointrend_model,frame_oe_detector,
                                                               true_item,main_object_bbox,self.object_extraction_model_path,
                                                               output_folder = self.object_extractor_output_path,
                                                               video_id=1,frame_id = frame_id)
        if object_extraction_pred_dict!=None:
            for pred_key in object_extraction_final_pred_dict.keys():
                if pred_key in object_extraction_pred_dict.keys():
                    object_extraction_final_pred_dict[pred_key] = object_extraction_pred_dict[pred_key]
            object_extraction_pred = self.get_total_pred(object_extraction_final_pred_dict)
        else:
            object_extraction_pred = target_model_pred
            self.oe_cannot_extract+=1
        return object_extraction_pred,object_extraction_final_pred_dict

    def get_ensemble_pred(self,scene_processing_pred_dict, object_extraction_pred_dict):
        """
        Get the ensmble of the object extraction detector and the scene processing detector classifications.
        :param scene_processing_pred_dict: req. probability dictionary of the scene processing detector result.
        :param object_extraction_pred_dict: req. probability dictionary of the object extraction detector result.
        :return: An ensemble prediction.
        """
        for label in scene_processing_pred_dict:
            if label not in object_extraction_pred_dict.keys():
                object_extraction_pred_dict[label] = 0
            scene_processing_pred_dict[label] += float(object_extraction_pred_dict[
                                                        label])
        pred = max(scene_processing_pred_dict,
                   key=scene_processing_pred_dict.get)
        return pred

    def detect(self, evaluation_type):
        """
        Main function for detection an adversarial patch in digital use case.
        :param evaluation_type: req. str. evaluation on 'benign' or 'adversarial' datasets.
        :return: Evaluation summary report of the adversarial detection.
        """

        self.initialize_evaluation_variables()
        for patch_class_folder in os.listdir(self.evaluation_data_dir):
            self.patch_class = patch_class_folder
            self.object_list = self.get_object_list(patch_class_folder)
            # Path(os.path.join(self.output_path,patch_class_folder)).mkdir(parents=True, exist_ok=True)
            ids_list, images = self.load_dataset_from_path(os.path.join(self.evaluation_data_dir,patch_class_folder))
            self.data_util = MS_COCO_util(self.MSCOCO_data_path, self.annotations_file_path, ids_list)
            image_dicts = self.data_util.load_coco_dataset_from_ids_list()
            image_dicts = self.replace_images_image_dicts(image_dicts,images,ids_list)
            for image_dict in image_dicts:
                image_dict['prediction'] = self.predict(image_dict['tested_image'])
                main_object_dict = self.get_main_object_bbox_and_label(image_dict)
                main_object_dict['target_model_pred'] = self.get_object_pred_by_bbox(main_object_dict['bbox'],image_dict['prediction'],np.copy(image_dict['tested_image']),self.output_path+f"/{patch_class_folder}/{str(image_dict['id'])}.jpg")
                if evaluation_type=="benign" and main_object_dict['label']!= main_object_dict['target_model_pred']:
                    continue
                elif evaluation_type=="adversarial" and main_object_dict['target_model_pred']!= self.patch_class:
                    continue
                scene_manipulation_pred, scene_manipulation_pred_dict = \
                    self.get_scene_manipulation_preds(image_dict['tested_image'],main_object_dict['bbox'],image_dict['id'])
                main_object_dict['sm_pred'] = scene_manipulation_pred
                scene_manipulation_pred_dict = self.normalized_preds(scene_manipulation_pred_dict)
                self.update_scene_processing_eval(main_object_dict['label'],main_object_dict['target_model_pred'])
                object_extraction_pred, object_extraction_prediction_dict = \
                    self.get_object_extraction_pred(image_dict['tested_image'], main_object_dict['bbox'],
                                                    main_object_dict['label'],main_object_dict['target_model_pred'],image_dict['id'])
                ensemble_pred = self.get_ensemble_pred(scene_manipulation_pred_dict,
                                                       object_extraction_prediction_dict)
                if self.use_ad_yolo:
                    ad_yolo_preds = self.get_ad_yolo_pred(image_dict['tested_image'])
                    ad_yolo_main_pred = self.get_object_pred_by_bbox(main_object_dict['bbox'], ad_yolo_preds, ad_yolo_model=True)
                    label_to_remove = ((ad_yolo_preds[0]['labels'] == COCO_INSTANCE_CATEGORY_NAMES_80_CLASSES.index(ad_yolo_main_pred)).nonzero().item())
                    ad_yolo_preds[0]['scores'][label_to_remove] = 0
                    ad_yolo_second_pred = self.get_object_pred_by_bbox(main_object_dict['bbox'], ad_yolo_preds,ad_yolo_model=True)
                    ad_yolo_preds = (ad_yolo_main_pred,ad_yolo_second_pred)
                    self.update_result(main_object_dict, scene_manipulation_pred, object_extraction_pred, ensemble_pred,
                                       image_dict['id'],ad_yolo_preds)
                else:
                    self.update_result(main_object_dict,scene_manipulation_pred,object_extraction_pred,ensemble_pred,image_dict['id'])
        self.evaluation_summary_report()


    def get_object_list(self, patch_target_class):
        """
        Get the category id of specific labels in MS COCO dataset.
        :param patch_target_class: req. str. The patch target class.
        :return: A list containing the ids.
        """
        objects_id = []
        object_list = ['apple', 'banana', 'pizza', 'orange']
        object_list.remove(patch_target_class)
        for object_label in object_list:
            objects_id.append(COCO_INSTANCE_CATEGORY_NAMES.index(object_label))

        return objects_id

    def update_result(self, main_object_dict, scene_processing_pred, object_extraction_pred,ensemble_pred,
                      obj_id,ad_yolo_preds = None):
        """
        Update the evaluation results.
        :param main_object_dict: req. dict containing information on the main object.
        :param scene_processing_pred: req. str. the scene processing detector prediction.
        :param object_extraction_pred: req. str. the object extraction detector prediction.
        :param ensemble_pred: req. str. the ensemble detector prediction.
        :param obj_id: req. the image MS COCO id.
        :param ad_yolo_preds: opt. The Ad-YOLO adversarial detection prediction.
        :return: ---
        """
        self.sample_counter+=1
        ground_truth = main_object_dict['label']
        target_model_pred = main_object_dict['target_model_pred']
        # print(f"Ground truth:{ground_truth}")
        # print(f"Target model pred:{target_model_pred}")
        # print(f"Scene processing pred:{scene_manipulation_pred}")
        # print(f"Object extraction pred:{object_extraction_pred}")
        if target_model_pred==scene_processing_pred:
            vertical_ensemble = scene_processing_pred
        else:
            vertical_ensemble = object_extraction_pred
        # print(f"Ensemble pred:{ensemble_pred}")
        # print(f"Vertical ensemble pred:{vertical_ensemble}")
        if target_model_pred != scene_processing_pred:
            self.sm_adversarial_detection+=1

        if target_model_pred != object_extraction_pred:
            self.oe_adversarial_detection +=1

        if target_model_pred != ensemble_pred:
            self.ensemble_adversarial_detection+=1

        if target_model_pred != vertical_ensemble:
            self.vertical_ensemble_adversarial_detection+=1

        if ground_truth == scene_processing_pred:
            self.sm_robustness +=1

        if ground_truth == object_extraction_pred:
            self.oe_robustness +=1

        if ground_truth == ensemble_pred:
            self.ensemble_robustness +=1

        if ground_truth == vertical_ensemble:
            self.vertical_ensemble_robustness+=1

        if ad_yolo_preds:
            print(f"AD-YOLO preds:{ad_yolo_preds}")
            if ad_yolo_preds[0]=="adversarial patch":
                self.ad_yolo_adversarial_detection +=1
            if ad_yolo_preds[0]==ground_truth or ad_yolo_preds[1]==ground_truth:
                self.ad_yolo_robustness+=1


    def evaluation_summary_report(self):
        """
        Generate an evaluation summary report.
        :return: ---
        """

        print(f"Total samples:{self.sample_counter}")
        print(f"Scene processing adversarial rate:{self.sm_adversarial_detection/self.sample_counter}")
        print(f"Object extraction adversarial rate:{self.oe_adversarial_detection/self.sample_counter}")
        print(f"Ensemble adversarial rate:{self.ensemble_adversarial_detection/self.sample_counter}")
        print(f"Vertical ensemble adversarial rate:{self.vertical_ensemble_adversarial_detection/self.sample_counter}")
        # print(f"Ad YOLO adversarial rate:{self.ad_yolo_adversarial_detection/self.sample_counter}")

        print(f"Scene processing robustness rate:{self.sm_robustness / self.sample_counter}")
        print(f"Object extraction robustness rate:{self.oe_robustness / self.sample_counter}")
        print(f"Ensemble robustness rate:{self.ensemble_robustness / self.sample_counter}")
        print(f"Vertical ensemble robustness rate:{self.vertical_ensemble_robustness / self.sample_counter}")
        # print(f"Ad YOLO robustness rate:{self.ad_yolo_robustness / self.sample_counter}")


    def is_contained(self, curr_box_pred, main_object_bbox):
        """
        Check if the current bounding box is contained inside the main object bounding box.
        :param curr_box_pred: req. list. current bounding box.
        :param main_object_bbox: req. list. main object bounding box.
        :return: boolean value.
        """
        return curr_box_pred[0]>main_object_bbox[0] and curr_box_pred[1]>main_object_bbox[1] and curr_box_pred[2] < main_object_bbox[2] and curr_box_pred[3]<main_object_bbox[3]

    def normalized_preds(self, scene_processing_pred_dict):
        """
        normalized the scene processing probability dictionary.
        :param scene_manipulation_pred_dict:
        :return:
        """
        total_preds = sum(scene_processing_pred_dict.values())
        for pred in scene_processing_pred_dict.keys():
            scene_processing_pred_dict[pred] /=total_preds

        return scene_processing_pred_dict



def detection_demo(MSCOCO_dataset_path,annotations_file_path,prototypes_path,style_transfer_model_path,
                   adversarial_data_dir,benign_data_dir,output_path,target_model_path):
    """
    Demo detection function - used for applying adversarial detection.
    :param MSCOCO_dataset_path: req. str. path to the MS COCO dataset.
    :param annotations_file_path: req. str. path to the MS COCO annotations file.
    :param prototypes_path: req. str. path to the prototype images.
    :param style_transfer_model_path: req. str. path to the style transfer models.
    :param adversarial_data_dir: req. str. Path to the adversarial images folder.
    :param benign_data_dir: req. str. Path to the benign images folder.
    :param output_path: req. str. Path to the output folder.
    :param target_model_path: req. str. Path to the target model.
    :return: --
    """
    # Detection evaluation on adversarial scenes
    evaluation_type = "adversarial"
    dc = detector_controller_MSCOCO(MSCOCO_dataset_path, adversarial_data_dir, annotations_file_path, prototypes_path,
                             style_transfer_model_path, target_model_path,output_path =output_path)
    print("Detection evaluation on adversarial scenes...")
    dc.detect(evaluation_type)

    # Detection evaluation on benign scenes
    evaluation_type = "benign"
    dc = detector_controller_MSCOCO(MSCOCO_dataset_path, benign_data_dir, annotations_file_path, prototypes_path,
                                    style_transfer_model_path, target_model_path,output_path = output_path)
    print("Detection evaluation on benign scenes...")
    dc.detect(evaluation_type)
