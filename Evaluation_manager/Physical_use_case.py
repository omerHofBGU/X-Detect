from Object_detection.target_model_prediction import classification_from_video_demo
from Object_detection.target_model_digital_attack_new import classfication_from_scenes_demo
from Adversarial_detection.detector_controller import adversarial_detection_demo
from Adversarial_detection.Detector_controller_MS_COCO import detection_demo
import pandas as pd
import os
from tqdm import tqdm

class demo():
    def __init__(self,attack_scenario):
        super().__init__()
        self.attack_scenario = attack_scenario
        self.use_mmdetection_model = False
        self.set_util_models()
        self.output_path = "Output"

    def set_util_models(self):
        """
        Set utils models path.
        :return: --
        """
        self.style_transfer_model_path = "../Models/Util_models"
        self.object_extraction_model_path = "../Models/Util_models"


class physical_use_case_experiments(demo):

    def __init__(self,attack_scenario):
        super().__init__(attack_scenario)
        self.number_of_classes = 21
        self.set_prototype_path()
        self.set_target_model()



    def set_prototype_path(self):
        """
        Set prototype images path.
        :return: --
        """
        self.prototypes_path = "../Datasets_util/Prototypes/Superstore"

    def set_target_model(self):
        """
        Set target models path according to a given attack scenario.
        :param attack_scenario: req. str. The attack scenario.
        :return: --
        """
        print(f"Starting experiments for physical use case {self.attack_scenario} attack scenario")
        self.adaptive_attack = False
        if self.attack_scenario=="White-Box":
            self.target_model_path = "../Models/Physical_use_case/White_box/Faster_RCNN_seed_42.pt"
        elif self.attack_scenario=="Gray-Box":
           self.target_model_path = "../Models/Physical_use_case/Gray_box/Faster_RCNN_seed_38.pt"
           self.target_model_path = "../Models/Physical_use_case/Gray_box/Faster_RCNN_seed_40.pt"
           self.target_model_path = "../Models/Physical_use_case/Gray_box/Faster_RCNN_seed_44.pt"

        elif self.attack_scenario=="model specific":
            self.use_mmdetection_model = True
            target_model_checkpoint = "../Models/Physical_use_case/Model_specific/Faster_RCNN_caffe_fpn_50.pth"
            target_model_configuration = "../Models/Physical_use_case/Model_specific/Faster_RCNN_caffe_fpn_50_conf.py"
            self.target_model_path = target_model_checkpoint,target_model_configuration
            target_model_checkpoint = "../Models/Physical_use_case/Model_specific/Faster_RCNN_caffe_fpn_101.pth"
            target_model_configuration = "../Models/Physical_use_case/Model_specific/Faster_RCNN_fpn_101_conf.py"
            self.target_model_path = target_model_checkpoint, target_model_configuration
            target_model_checkpoint = "../Models/Physical_use_case/Model_specific/Faster_RCNN_fpn_50_iou.pth"
            target_model_configuration = "../Models/Physical_use_case/Model_specific/Faster_RCNN_fpn_50_iou_conf.py"
            self.target_model_path = target_model_checkpoint, target_model_configuration

        elif self.attack_scenario=="model agnostic":
            self.use_mmdetection_model = True
            target_model_checkpoint = "../Models/Physical_use_case/Model_agnostic/YOLOv3.pth"
            target_model_configuration = "../Models/Physical_use_case/Model_agnostic/yolov3_conf.py"
            self.target_model_path = target_model_checkpoint, target_model_configuration
            target_model_checkpoint = "../Models/Physical_use_case/Model_agnostic/Cascade_rpn_r50.pth"
            target_model_configuration = "../Models/Physical_use_case/Model_agnostic/cascade_rpn_r50_fpn_conf.py"
            self.target_model_path = target_model_checkpoint, target_model_configuration
            target_model_checkpoint = "../Models/Physical_use_case/Model_agnostic/Cascade_rcnn_r50_fpn.pth"
            target_model_configuration = "../Models/Physical_use_case/Model_agnostic/cascade_rcnn_r50_fpn_conf.py"
            self.target_model_path = target_model_checkpoint, target_model_configuration

        elif self.attack_scenario=="adaptive attacks":
            self.adaptive_attack = True
            self.target_model_path = "../Models/Physical_use_case/White_box/Faster_RCNN_seed_42.pt"

        else:
            print("Attack scenario not exist, starting experiment for White-Box attack scenario")
            self.target_model_path = "../Models/Physical_use_case/White_box/Faster_RCNN_seed_42.pt"



    def attack_evaluation(self):
        """
        Attack evaluation function, generate an attack using the adversarial videos on the chosen target model.
        Produce an evaluation summary report.
        :return: The adversarial videos predictions.
        """
        # adversarial_videos_input_path = "../Evaluation_set/physical_use_case/evasion_evaluation_videos/"
        adversarial_videos_input_path = "../Evaluation_set/physical_use_case/benign_evaluation_videos/"
        # Generate attack based on the adversarial video set
        print(f"Evaluate {self.attack_scenario} attack scenario on adversarial videos...")
        attack_predictions = classification_from_video_demo\
            (self.target_model_path,self.number_of_classes,adversarial_videos_input_path,
             self.output_path,self.use_mmdetection_model,self.adaptive_attack)
        print("Attack evaluation report successfully saved as a csv file in the output folder")
        return attack_predictions

    def detection_evaluation(self,attack_videos_information):
        """

        :param attack_predictions:
        :return:
        """
        benign_videos_path = "../Evaluation_set/physical_use_case/benign_evaluation_videos/"
        benign_videos_information = pd.read_csv(benign_videos_path+"/predictions.csv")
        print(f"Evaluate detection on successful attack videos and benign videos...")
        adversarial_detection_demo(self.target_model_path, self.number_of_classes,
                                   self.use_mmdetection_model,attack_videos_information,
                                   benign_videos_information, self.prototypes_path,
                                   style_transfer_model_path=self.style_transfer_model_path,
                                   object_extraction_model_path = self.object_extraction_model_path)


class digital_use_case_experiments(demo):
    def __init__(self, attack_scenario):
        super().__init__(attack_scenario)
        self.number_of_classes = 21
        self.attack_scenario = attack_scenario
        self.data_dir = '../Datasets_util/MS_COCO_dataset'
        self.annotations_file_path = os.path.join(self.data_dir, 'dataset.json')
        self.ids_list_path = f"../Evaluation_set/digital_use_case/MS_COCO_ids_list"
        self.patch_path = f"../Evaluation_set/digital_use_case/Patches"
        self.use_mmdetection_model = False
        self.set_prototype_path()
        self.set_target_model()
        self.output_path = "Output"

    def set_prototype_path(self):
        """
        Set prototype images path.
        :return: --
        """
        self.prototypes_path = "../Datasets_util/Prototypes/MS_COCO"

    def set_target_model(self):
        """
        Set target models path according to a given attack scenario.
        :param attack_scenario: req. str. The attack scenario.
        :return: --
        """
        print(f"Starting experiments for physical use case {self.attack_scenario} attack scenario")

        if self.attack_scenario=="Model specific":
            self.use_mmdetection_model = True
            target_model_checkpoint = "../Models/Digital_use_case/Model_specific/Faster_RCNN_caffe_fpn_50.pth"
            target_model_configuration = "../Models/Digital_use_case/Model_specific/Faster_RCNN_caffe_fpn_50_conf.py"
            self.target_model_path = target_model_checkpoint,target_model_configuration
            target_model_checkpoint = "../Models/Digital_use_case/Model_specific/Faster_RCNN_caffe_fpn_101.pth"
            target_model_configuration = "../Models/Digital_use_case/Model_specific/Faster_RCNN_fpn_101_conf.py"
            self.target_model_path = target_model_checkpoint, target_model_configuration
            target_model_checkpoint = "../Models/Digital_use_case/Model_specific/Faster_RCNN_fpn_50_iou.pth"
            target_model_configuration = "../Models/Digital_use_case/Model_specific/Faster_RCNN_fpn_50_iou_conf.py"
            self.target_model_path = target_model_checkpoint, target_model_configuration

        elif self.attack_scenario=="Model agnostic":
            self.use_mmdetection_model = True
            target_model_checkpoint = "../Models/Digital_use_case/Model_agnostic/Grid_RCNN_fpn_50.pth"
            target_model_configuration = "../Models/Digital_use_case/Model_agnostic/Grid_RCNN_fpn_50_conf.py"
            self.target_model_path = target_model_checkpoint, target_model_configuration
            target_model_checkpoint = "../Models/Digital_use_case/Model_agnostic/Cascade_rcnn_r50_fpn.pth"
            target_model_configuration = "../Models/Digital_use_case/Model_agnostic/cascade_rcnn_r50_fpn_conf.py"
            self.target_model_path = target_model_checkpoint, target_model_configuration

        else:
            print("Attack scenario not exist, starting experiment for White-Box attack scenario")

    def attack_evaluation(self):
        """
        Attack evaluation function, generate an attack using the adversarial images on the chosen target model.
        Produce an evaluation summary report.
        :return: The adversarial images predictions.
        """
        adversarial_videos_input_path = "../Evaluation_set/digital_use_case/Adversarial_images/"
        # Generate attack based on the adversarial video set
        print(f"Evaluate {self.attack_scenario} attack scenario on adversarial scenes...")
        attack_predictions = classfication_from_scenes_demo(self.data_dir,self.annotations_file_path,
                                                            self.ids_list_path,self.patch_path,
                                                            self.use_mmdetection_model)

        print("Attack evaluation report successfully saved as a csv file in the output folder")
        return attack_predictions

    def detection_evaluation(self, attack_videos_information):
        """

        :param attack_predictions:
        :return:
        """
        adversarial_data_dir = "../Evaluation_set/digital_use_case/Adversarial_images"
        benign_data_dir = "../Evaluation_set/digital_use_case/Benign_images"
        print(f"Evaluate detection on successful attack scenes and benign scenes...")
        detection_demo(self.data_dir,self.annotations_file_path,self.prototypes_path,self.style_transfer)



def physical_use_case_demo(attack_scenario):
    """
    Module for physical experiment.
    :param attack_scenario: req. str. The attack scenario upon the attack will be generated and detected,
    according to the following attack scenarios:
    1. "White-Box" - complete knowledge of the target model.
    2. "Gray-box" - no knowledge of the target model’s parameters.
    3. "Model specific" - knowledge on the ML algorithm used.
    4. "Model agnostic" - no knowledge on the target model.
    The demo starts from applying the adversarial videos set on the target model choose according to the
    desirable attack scenario. Next the successful adversarial videos (meaning the adversarial video that succeed to
    fool the target model to misclassify the true class are tested against X-Detect
    :return: ---
    """

    # Step 1 - define attack scenario to test
    physical_experiment = physical_use_case_experiments(attack_scenario)
    # Step 2 - generate and evaluate attack
    attack_predictions = physical_experiment.attack_evaluation()
    # Step 3 - evaluate X-detect on the successful attacks.
    # attack_predictions = pd.read_csv("/sise/home/omerhof/x-detect/Evaluation_manager/Output/16-08-2022_11;50/Attack evaluation/predictions.csv")
    physical_experiment.detection_evaluation(attack_predictions)

def digital_use_case_demo(attack_scenario):
    """
    Module for digital experiment.
    :param attack_scenario: req. str. The attack scenario upon the attack will be generated and detected,
    according to the following attack scenarios:
    1. "White-Box" - complete knowledge of the target model.
    2. "Model specific" - knowledge on the ML algorithm used.
    3. "Model agnostic" - no knowledge on the target model.
    The demo starts from placing the crafted patches to MS-COCO scenes and evaluate the target model chosen
    according to the desirable attack scenario.
    Next the successful adversarial scenes (those the adversarial scene succeed to fool the target model to
    misclassify the true class) are tested against X-Detect.
    :return: ---
    """
    # Step 1 - define attack scenario to test
    digital_experiment = digital_use_case_experiments(attack_scenario)
    # Step 2 - generate and evaluate attack
    attack_predictions = digital_experiment.attack_evaluation()
    # Step 3 - evaluate X-detect on the successful attacks.
    digital_experiment.detection_evaluation(attack_predictions)



# physical_use_case_demo("White-Box")
digital_use_case_demo("White-Box")