import json
import os

import pandas as pd
import torch
import time
import cv2
import imagezmq
import datetime
from pathlib import Path
import numpy as np
from .object_extractor_detector import Prototypes,oe_detector_demo_super_store,Pointrend_instance_segmentation
from .scene_processing_detector import scene_processor_detector,frame_sm_detector_demo_super_store
from pytorch_lightning import seed_everything
from Object_detection.Object_detection_physical_use_case import SUPER_STORE_INSTANCE_CATEGORY_NAMES,\
    evaluate_target_model,ObjectDetectionDatasetSingleFromNumpy
# import warnings
# warnings.filterwarnings("ignore")

class detector_controller():

    def __init__(self, target_model_path, number_of_classes, prototypes_path,style_transfer_model_path,
                 use_mmdetection_model,object_extraction_model_path, output_path,
                 ad_yolo_model_path = None, optimization_params = None):
        super().__init__()
        # upload the target model from a given path
        self.target_model = self.init_target_model(
            target_model_path, number_of_classes,use_mmdetection_model)
        if ad_yolo_model_path:
            self.ad_yolo_model = self.init_target_model(
                ad_yolo_model_path, number_of_classes+1,use_mmdetection_model)
            self.use_ad_yolo = True
        else:
            self.use_ad_yolo = False
        self.output_path = output_path
        self.target_model_prediction_dict = self.get_dict()
        self.ad_yolo_prediction_dict = self.get_dict()
        self.object_extraction_prediction_dict = self.get_dict()
        self.scene_manipulation_prediction_dict = self.get_dict()
        self.expensive_items_dict=self.get_items_dict()
        self.cannot_extract_counter = 0
        self.cannot_extract_total = 0
        self.init_detectors(prototypes_path,style_transfer_model_path,object_extraction_model_path,optimization_params)
        self.object_extraction_model_path = object_extraction_model_path


    def get_dict(self):
        """
        Produces an empty dictionary with all 21 superstore classes (including adversarial patch class for Ad-YOLo method).
        This dictionary is used for aggregating smart cart predictions.
        :return: empty dictionary with all 21 superstore classes
        """
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
            20: 0,
            21:0
        }
    def get_items_dict(self):
        """
        Produces a dictionary with all 21 superstore classes (including adversarial patch class for Ad-YOLo method).
        :return: a dictionary with all 21 superstore classes
        """
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
            'Adversarial Patch':21,
        }

    def init_target_model(self,target_model_path,number_of_classes,use_mmdetection_model):
        """
        Wrapper function that initialize target model.
        :param target_model_path: req. str. path to the target model.
        :param number_of_classes: req. int. the number of classes used in the target model.
        :param use_mmdetection_model: req. bool. whether to use mm-detection model-zoo framework.
        :return: object detection model instance.
        """
        target_model = evaluate_target_model(target_model_path,number_of_classes, output_path="",mmdetection_model=use_mmdetection_model)
        return target_model


    def init_detectors(self,prototypes_path,style_transfer_models_path,object_extraction_model_path,optimization_params):
        """
        Init detectors utils (such as prototpes for the object extraction detector, the stle transfer model
        for scene processing detector etc.)
        :param prototypes_path:
        :param style_transfer_models_path:
        :param object_extraction_model_path:
        :param optimization_params:
        :return: ---
        """
        self.prototypes = Prototypes(prototypes_path,object_extraction_model_path)
        self.prototypes.process_prototypes()
        self.pointrend_model = Pointrend_instance_segmentation(object_extraction_model_path)
        self.scene_manipulation_detector = scene_processor_detector(None,self.target_model,style_transfer_models_path,
                                                                    use_case="physical", output_path= self.output_path)
        self.object_extractor_output_path = os.path.join(self.output_path,"Object_extractor_detector")
        Path(self.object_extractor_output_path).mkdir(parents=True,exist_ok=True)


    def detection_demo_for_videos_by_video_annotations_file(self,videos_annotations_df,benign_evaluation= False):
        """
        Main function for detecting adversarial patches in videos, using a video annotations file.
        :param videos_annotations_df: req. video annotations in dataframe instance.
        :return: Print the total evaluation result to the console.
        """
        seed_everything(42)
        output_folder = self.init_variables_for_video_evaluation()
        start = time.time()

        for index, row in videos_annotations_df.iterrows():
            if row['Target_attack_succeed'] or row['Cheap_attack_succeed']:
                video_path = row['Video_path']
                self.classify_video_internal_loop(os.path.dirname(video_path), 1,
                                                  row['True_class'], row['Target_class'], output_folder,Path(video_path).name)
        self.print_total_results(start,benign_evaluation)


    def print_total_results(self, start,benign_evaluation = False):
        """
        Function that prints the total results to the console.
        :param start: req. start time.
        :return: ---
        """
        evaluation = "adversrial"
        if benign_evaluation:
            evaluation = "benign"
            self.adversarial_alerts_ve = self.total_videos - self.adversarial_alerts_ve
            self.adversarial_alerts_mc_ensemble = self.total_videos - self.adversarial_alerts_mc_ensemble
            self.adversarial_alerts_sm = self.total_videos - self.adversarial_alerts_sm
            self.adversarial_alerts_oe = self.total_videos - self.adversarial_alerts_oe
        print(f"Total detection time: {time.time()-start}")
        print("Vertical ensemble detector result:")
        print(
            f"Total {evaluation} samples detected:{self.adversarial_alerts_ve} "
            f"out of {self.total_videos} "
            f"{evaluation} videos.")
        print(f"Total TPR: "
              f"{self.adversarial_alerts_ve / self.total_videos}")
        print(f"Total robustness: "
              f"{self.robustness_score_ve / self.total_videos}")
        print(f"Average detection time: {sum(self.ve_timer)/len(self.ve_timer)}")

        print("MV Ensemble detector result:")
        print(
            f"Total {evaluation} samples detected:{self.adversarial_alerts_mc_ensemble} "
            f"out of {self.total_videos} "
            f"{evaluation} videos.")

        print(f"Total TPR: "
              f"{self.adversarial_alerts_mc_ensemble / self.total_videos}")
        print(f"Total robustness: "
              f"{self.robustness_score_mc_ensemble / self.total_videos}")

        print("Scene processing detector result:")
        print(
            f"Total {evaluation} samples detected:{self.adversarial_alerts_sm} "
            f"out of {self.total_videos} "
            f"{evaluation} videos.")
        print(f"Total TPR: "
              f"{self.adversarial_alerts_sm / self.total_videos}")
        print(f"Total robustness: "
              f"{self.robustness_score_sm / self.total_videos}")
        print(
            f"Average detection time: "
            f"{sum(self.sm_timer) / len(self.sm_timer)}")

        print("Object extractor detector result:")
        print(f"Total {evaluation} samples detected:"
              f"{self.adversarial_alerts_oe} out of {self.total_videos} "
              f"{evaluation} videos.")

        print(f"Total TPR: "
              f"{self.adversarial_alerts_oe / self.total_videos}")
        print(
            f"Total not extracted videos: {self.cannot_extract_total / self.total_videos}")
        print(
            f"Total robustness: {self.robustness_score_oe / self.total_videos}")

        print(
            f"Average detection time: "
            f"{sum(self.oe_timer) / len(self.oe_timer)}")

        print("Ad yolo detector result:")
        print(f"Detection rate:  {self.ad_yolo_adversarial_alert / self.total_videos}")
        print(f"Fooled rate:  {self.ad_yolo_fooled / self.total_videos}")
        print(f"Robustness rate:  {self.ad_yolo_robustness_rate / self.total_videos}")

    def init_variables_for_video_evaluation(self):
        """
        Initialize variables for video evaluation.
        :return: variables for video evaluation
        """
        # output_folder = self.create_output_folder()
        self.adversarial_alerts_oe = 0
        self.adversarial_alerts_sm = 0
        self.adversarial_alerts_ve = 0
        self.adversarial_alerts_ensemble = 0
        self.adversarial_alerts_mc_ensemble = 0
        self.total_videos = 0
        self.robustness_score_oe = 0
        self.robustness_score_sm = 0
        self.robustness_score_ve = 0
        self.robustness_score_ensemble = 0
        self.robustness_score_mc_ensemble = 0
        self.use_oe_detector = 0
        self.scene_manipulation_eval_dict = self.create_sm_eval_dict()
        self.sm_timer = []
        self.oe_timer = []
        self.ve_timer = []
        self.ensemble_timer = []
        self.sm_preds = []
        self.oe_preds = []
        self.ensemble_preds = []
        self.ve_preds = []
        self.most_confidence_ensemble_pred = []
        self.true_items = []
        self.failure_list = []
        self.ad_yolo_adversarial_alert = 0
        self.ad_yolo_fooled = 0
        self.ad_yolo_robustness_rate = 0


    def classify_video_internal_loop(self,videos_folder,videos_counter, item,patch_class, output_folder,video_id_folder=None):
        """
        Hekper function the video classification.
        :param videos_folder: req. str. path to the videos folder.
        :param videos_counter: req. int. video counter.
        :param item: req. str. the correct item inserted into the cart.
        :param patch_class: req. str. the patch target class.
        :param output_folder: req. str. path to output folder to save the explainable outputs.
        :param video_id_folder: opt. id off the videos folder.
        :return: ---
        """
        detection_performance = 0
        for idx, video_folder in enumerate(os.listdir(videos_folder)):
            if video_id_folder==None or video_id_folder==video_folder:
                start = time.time()
                videos_counter += 1
                self.total_videos += 1
                video_folder_path = os.path.join(videos_folder, video_folder)
                target_model_pred, oe_detector_pred,sm_detector_pred,\
                v_e_detector_pred, ensemble_pred,most_confidence_ensemble,ad_yolo_pred,ad_yolo_pred_2 = \
                    self.video_classification(video_folder_path, item,videos_counter,output_folder)
                self.sm_preds.append(sm_detector_pred)
                self.oe_preds.append(oe_detector_pred)
                self.ensemble_preds.append(ensemble_pred)
                self.most_confidence_ensemble_pred.append(most_confidence_ensemble)
                self.ve_preds.append(v_e_detector_pred)
                self.true_items.append(item)

                if oe_detector_pred!=None and target_model_pred != oe_detector_pred and patch_class!=oe_detector_pred:
                    self.adversarial_alerts_oe += 1
                if target_model_pred != sm_detector_pred and patch_class!=sm_detector_pred:
                    self.adversarial_alerts_sm += 1
                if target_model_pred != v_e_detector_pred and patch_class!=v_e_detector_pred:
                    self.adversarial_alerts_ve += 1
                else:
                    # print(f"vertical Ensemble failed at {video_folder_path}")
                    self.failure_list.append(video_folder_path)
                if target_model_pred != ensemble_pred and patch_class!=ensemble_pred:
                    self.adversarial_alerts_ensemble += 1
                else:
                    # print(f"Ensemble failed at {video_folder_path}")
                    self.failure_list.append(video_folder_path)
                if target_model_pred!=most_confidence_ensemble:
                    self.adversarial_alerts_mc_ensemble+=1
                if oe_detector_pred!=None and item == oe_detector_pred:
                    self.robustness_score_oe += 1
                if item == sm_detector_pred:
                    self.robustness_score_sm += 1
                if item == v_e_detector_pred:
                    detection_performance+=1
                    self.robustness_score_ve += 1
                if item == ensemble_pred:
                    self.robustness_score_ensemble += 1
                if item == most_confidence_ensemble:
                    self.robustness_score_mc_ensemble += 1

                if self.use_ad_yolo:
                    if target_model_pred == ad_yolo_pred:
                        self.ad_yolo_fooled+=1
                    if ad_yolo_pred == 'Adversarial Patch' or ad_yolo_pred_2 =='Adversarial Patch':
                        self.ad_yolo_adversarial_alert+=1
                    if item==ad_yolo_pred or item==ad_yolo_pred_2:
                        self.ad_yolo_robustness_rate += 1

                # print(f'Detection time: {time.time()-start}')

    def read_frames(self,path):
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
                frames.append(np.load(os.path.join(path,frame)))
        else:
            for frame in os.listdir(path):
                frames.append(cv2.imread(os.path.join(path, frame)))
        return frames

    def processed_frames(self,frame):
        """
        Process frames towards the target model.
        :param frame: req. image in numpy format.
        :return: processed frames for the object extraction detector, the scene processing detector and the target model.
        """
        frame_oe_detector =  np.copy(frame)
        frame_oe_detector = cv2.cvtColor(frame_oe_detector, cv2.COLOR_BGR2RGB)
        frame_sm_detector = np.copy(frame)
        target_model_frame = np.expand_dims(frame, axis=0)
        return frame_oe_detector,frame_sm_detector,target_model_frame


    def video_classification(self,video_folder,true_item,video_num,output_folder):
        """
        Function that classify a video of item inserted into a shopping cart.
        :param video_folder: req. str. the path to the video folder.
        :param true_item: req. the correct item inserted into the cart.
        :param video_num: req. int. the id of the video examined.
        :param output_folder: req. the output folder path.
        :return: prediction of the adversarial detectors and the target model.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else
                              "cpu")
        self.target_model.target_model = self.target_model.target_model.to(
            device)
        if self.use_ad_yolo:
            self.ad_yolo_model.target_model = self.ad_yolo_model.target_model.to(
                device)
        self.scene_manipulation_detector.create_image_processing_dict()
        frames = self.read_frames(f'{video_folder}/frames')
        start_ve_time = time.time()
        target_model_pred = self.get_target_model_pred(frames,device)
        scene_manipulation_pred, scene_manipulation_pred_dict = \
            self.get_scene_manipulation_preds(frames,device, true_item, target_model_pred,video_folder.split('/')[-1])
        if scene_manipulation_pred!=target_model_pred:
            self.use_oe_detector += 1
            object_extraction_pred, object_extraction_prediction_dict = self.get_object_extraction_pred(frames,
                                                                self.bbox_list, true_item, video_folder.split('/')[-1])
            ve_detection_pred = object_extraction_pred
            self.ve_timer.append(time.time() - start_ve_time)
        else:
            ve_detection_pred = scene_manipulation_pred
            self.ve_timer.append(time.time() - start_ve_time)
            object_extraction_pred, object_extraction_prediction_dict = \
                self.get_object_extraction_pred(frames, self.bbox_list, true_item, video_folder.split('/')[-1])

        ensemble_pred = self.get_ensemble_pred(scene_manipulation_pred_dict,
                                          object_extraction_prediction_dict)
        most_confidence_ensemble = self.get_most_confidence_ensemble_pred(
            scene_manipulation_pred_dict, object_extraction_prediction_dict)
        self.ensemble_timer.append(self.oe_timer[-1] + self.sm_timer[-1])
        if self.use_ad_yolo:
            ad_yolo_pred,ad_yolo_pred_2 = self.get_ad_yolo_preds(frames,device)
        else:
            ad_yolo_pred,ad_yolo_pred_2 = None,None
        return target_model_pred,object_extraction_pred,\
               scene_manipulation_pred,ve_detection_pred,ensemble_pred,most_confidence_ensemble,ad_yolo_pred,ad_yolo_pred_2

    def get_ensemble_pred(self,scene_manipulation_pred_dict,
                                          object_extraction_pred_dict):
        """
        Get the ensmble of the object extraction detector and the scene processing detector classifications.
        :param scene_processing_pred_dict: req. probability dictionary of the scene processing detector result.
        :param object_extraction_pred_dict: req. probability dictionary of the object extraction detector result.
        :return: An ensemble prediction.
        """
        for label in scene_manipulation_pred_dict:
            scene_manipulation_pred_dict[label] += float(object_extraction_pred_dict[
                                                        label])
        pred = max(scene_manipulation_pred_dict,
                   key=scene_manipulation_pred_dict.get)
        return SUPER_STORE_INSTANCE_CATEGORY_NAMES[pred]

    def get_most_confidence_ensemble_pred(self, scene_manipulation_pred_dict,
                                          object_extraction_prediction_dict):
        """
        Get the ensmble of the object extraction detector and the scene processing detector classifications.
        :param scene_processing_pred_dict: req. probability dictionary of the scene processing detector result.
        :param object_extraction_pred_dict: req. probability dictionary of the object extraction detector result.
        :return: An ensemble prediction.
        """
        sm_max_conf = max(scene_manipulation_pred_dict.values())
        oe_max_conf = max(object_extraction_prediction_dict.values())
        if sm_max_conf>oe_max_conf:
            pred = max(scene_manipulation_pred_dict,
                   key=scene_manipulation_pred_dict.get)
        else:
            pred = max(object_extraction_prediction_dict,
                   key=object_extraction_prediction_dict.get)
        return SUPER_STORE_INSTANCE_CATEGORY_NAMES[pred]

    def scene_manipulation_update_prediction(self,scene_manipulation_pred):
        """
        Update the total video prediction of the scene processing detector by the given scene prediction.
        :param scene_manipulation_pred: req. scene processing probability dictionary.
        :return: --
        """

        pred = max(scene_manipulation_pred,
                   key=scene_manipulation_pred.get)
        self.scene_manipulation_prediction_dict[pred] += \
            scene_manipulation_pred[pred]

    def target_model_update_prediction(self,target_model_pred):
        """
        Update the total video prediction of the target model by the given scene prediction.
        :param target_model_pred:  req. target model probability dictionary.
        :return: --
        """
        self.target_model_prediction_dict[
            int(target_model_pred[0]['labels'][0])] \
            += target_model_pred[0]['scores'][0]

    def ad_yolo_model_update_prediction(self,ad_yolo_pred):
        """
        Update the total video prediction of the Ad-YOLO detector by the given scene prediction.
        :param ad_yolo_pred: req. Ad-YOLO detector probability dictionary.
        :return: --
        """
        self.ad_yolo_prediction_dict[
            int(ad_yolo_pred[0]['labels'][0])] \
            += ad_yolo_pred[0]['scores'][0]

    def object_extraction_update_prediction(self,object_extraction_pred, number_of_frames_to_decision):
        """
        Update the total video prediction of the object extraction detector by the given scene prediction.
        :param object_extraction_pred: req. object extraction detector probability dictionary.
        :param number_of_frames_to_decision: req. int. handle variable for the extraction error.
        :return: --
        """
        if object_extraction_pred!=None and len(object_extraction_pred) > 0:
            pred = max(object_extraction_pred,
                       key=object_extraction_pred.get)
            self.object_extraction_prediction_dict[self.expensive_items_dict[pred]]+= \
                object_extraction_pred[pred]
        #same prediction as target model
        else:
            self.cannot_extract_counter+=1
            if self.cannot_extract_counter==number_of_frames_to_decision:
                self.cannot_extract_total+=1
                self.object_extraction_prediction_dict = self.target_model_prediction_dict


    def get_total_pred(self,pred_dict):
        """
        Select the prediction with the highest confidence.
        :param pred_dict: req. dictionary of all probabilities.
        :return: the prediction with the highest confidence
        """
        pred = max(pred_dict,
                   key=pred_dict.get)
        overall_pred = SUPER_STORE_INSTANCE_CATEGORY_NAMES[int(pred)]
        return overall_pred

    def update_scene_processing_eval(self,item,target_model_pred):
        """
        Function that update scene processing performance for each scene processing technique. .
        :param item: req. str. the correct item.
        :param target_model_pred:
        :return: --
        """
        for key in self.scene_manipulation_detector\
                .prediction_for_image_processing_tech.keys():
            pred = max(
                self.scene_manipulation_detector
                    .prediction_for_image_processing_tech[key]['prediction_dict'],
                       key=self.scene_manipulation_detector
                    .prediction_for_image_processing_tech[key]['prediction_dict'].get)
            overall_pred = SUPER_STORE_INSTANCE_CATEGORY_NAMES[int(pred)]
            if overall_pred == item:
                self.scene_manipulation_eval_dict[key]['total_detection_score']+=1
            if overall_pred!=target_model_pred:
                self.scene_manipulation_eval_dict[key][
                    'adversarial_detection'] += 1


    def create_output_folder(self):
        """
        Create output folder.
        :return:  --
        """
        output_path = f"{self.output_path}/detector_results"
        Path(output_path).mkdir(parents=True, exist_ok=True)
        return output_path

    def clean_dict(self):
        """
        Initialize dictionaries.
        :return: --
        """
        self.target_model_prediction_dict = self.get_dict()
        self.object_extraction_prediction_dict = self.get_dict()
        self.scene_manipulation_prediction_dict = self.get_dict()
        self.cannot_extract_counter = 0
        if self.use_ad_yolo:
            self.ad_yolo_prediction_dict = self.get_dict()

    def create_sm_eval_dict(self):
        """
        Create scene processing evaluation dictionary.
        :return: scene processing evaluation dictionary.
        """
        scenes = {}
        scenes['blur'] = {}
        scenes['blur']['total_detection_score'] = 0
        scenes['blur']['adversarial_detection'] = 0
        scenes['style_transfer'] = {}
        scenes['style_transfer']['total_detection_score'] = 0
        scenes['style_transfer']['adversarial_detection'] = 0
        scenes['random_noise'] = {}
        scenes['random_noise']['total_detection_score'] = 0
        scenes['random_noise']['adversarial_detection'] = 0
        scenes['darkness'] = {}
        scenes['darkness']['total_detection_score'] = 0
        scenes['darkness']['adversarial_detection'] = 0
        return scenes


    def get_target_model_pred(self,frames, device):
        """
        Return the target model classification.
        :param frames: req. numpy array of images. Representing the scenes.
        :param device: GPU/ CPU.
        :return: label - the target model classification.
        """
        self.bbox_list = []
        for frame in frames:
            if not self.target_model.mmdetection_model:
                target_model_pred = self.target_model.predict_faster_rcnn(frame,device)
            else:
                target_model_pred = self.target_model.predict_other_models(frame,device)

            if self.target_model.detected_object(target_model_pred[0]):
                self.bbox_list.append(target_model_pred[0]['boxes'][0])
                self.target_model_update_prediction(target_model_pred)

        target_model_pred = self.get_total_pred(
            self.target_model_prediction_dict)
        return target_model_pred

    def get_ad_yolo_preds(self,frames, device):
        """
         Return the Ad-YOLO classification.
        :param frames: req. numpy array of images. Representing the scenes.
        :param device: GPU/ CPU.
        :return: label - the Ad-YOLO model classification.
        """
        for frame in frames:
            if not self.ad_yolo_model.mmdetection_model:
                ad_yolo_pred = self.ad_yolo_model.predict_faster_rcnn(frame, device)
            else:
                ad_yolo_pred = self.ad_yolo_model.predict_other_models(frame,device)

            if self.ad_yolo_model.detected_object(ad_yolo_pred[0]):
                self.ad_yolo_model_update_prediction(ad_yolo_pred)

        target_model_pred = self.get_total_pred(
            self.ad_yolo_prediction_dict)
        self.ad_yolo_prediction_dict[max(self.ad_yolo_prediction_dict,key=self.ad_yolo_prediction_dict.get)] = 0
        target_model_pred_2 = self.get_total_pred(
            self.ad_yolo_prediction_dict)
        return target_model_pred,target_model_pred_2


    def get_scene_manipulation_preds(self, frames, device,true_item, target_model_pred,video_id):
        """
         Return the scene processing detector classification.
        :param frames: req. numpy array of images. Representing the scenes.
        :param device: GPU/ CPU.
        :param true_item: req. str. the correct item in the scene (used for evaluation).
        :param target_model_pred: req. str. the target model classification.
        :param video_id: str. req. the id of the examined video.
        :return: the scene processing detector classification.
        """
        start = time.time()
        for idx,frame in enumerate(frames):
            frame_oe_detector,frame_sm_detector,frame_target_model = self.processed_frames(frame)
            scene_manipulation_pred = frame_sm_detector_demo_super_store(self.scene_manipulation_detector,
                    frame_sm_detector, device, self.get_dict(),video_id,idx)
            self.scene_manipulation_update_prediction(scene_manipulation_pred)

        scene_manipulation_pred = self.get_total_pred(
            self.scene_manipulation_prediction_dict)
        self.sm_timer.append(time.time()-start)
        self.update_scene_processing_eval(true_item,target_model_pred)
        scene_manipulation_pred_dict = self.scene_manipulation_prediction_dict
        self.clean_dict()
        return scene_manipulation_pred,scene_manipulation_pred_dict

    def get_object_extraction_pred(self,frames, bbox_list, true_item,video_id):
        """
        Return the object extraction detector classification.
        :param frames: req. numpy array of images. Representing the scenes.
        :param bbox_list:
        :param true_item: req. str. the correct item in the scene (used for evaluation).
        :param video_id: req. the id of the examined video.
        :return: the object extraction detector classification.
        """
        start = time.time()
        for idx,frame in enumerate(frames):
            frame_oe_detector, frame_sm_detector, frame_target_model = \
                self.processed_frames(frame)
            if len(bbox_list)==idx:
                # print("OUT OF BOUNDS!")
                break
            object_extraction_pred = oe_detector_demo_super_store(self.prototypes,self.pointrend_model,
                frame_oe_detector,bbox_list[idx],self.object_extraction_model_path,true_item,True,
                                                  output_folder = self.object_extractor_output_path,
                                                                  video_id = video_id,frame_id = idx)
            self.object_extraction_update_prediction(object_extraction_pred, len(frames))

        object_extraction_pred = self.get_total_pred(
            self.object_extraction_prediction_dict)
        self.oe_timer.append(time.time()-start)
        object_extraction_prediction_dict = self.object_extraction_prediction_dict
        self.clean_dict()
        return object_extraction_pred,object_extraction_prediction_dict


def adversarial_detection_demo(target_model_path,number_of_classes,use_mmdetection_model,attack_videos_information,
                               benign_videos_information, prototypes_path, style_transfer_model_path,
                               object_extraction_model_path,output_path):
    """
    Main function for using X-Detect adversarial detection.
    :param target_model_path: req. str. path for the target model weights.
    :param number_of_classes: req. int. the number of classes used in the target model.
    :param use_mmdetection_model: req. bool. if the target model used is taken from the MM Detection package.
    :param attack_videos_information: req. pandas dataframe. Information regarding the adversarial videos evaluated.
    :param benign_videos_information:  req. pandas dataframe. Information regarding the benign videos evaluated.
    :param prototypes_path: req. str. path to prototypes instances.
    :param style_transfer_model_path: req. str. path for the style transfer model weights.
    :param object_extraction_model_path: req. str. path for the object extraction model weights.
    :param output_path: req. str. path for saving all X-Detect explainable outputs.
    :return: Evaluation report is printed to the console.
    """
    dc = detector_controller(target_model_path, number_of_classes, prototypes_path, style_transfer_model_path,
                             use_mmdetection_model, object_extraction_model_path,output_path)
    print("Detection evaluation on adversarial videos...")
    dc.detection_demo_for_videos_by_video_annotations_file(attack_videos_information)
    print("Detection evaluation on benign videos...")
    dc.detection_demo_for_videos_by_video_annotations_file(benign_videos_information,benign_evaluation=True)