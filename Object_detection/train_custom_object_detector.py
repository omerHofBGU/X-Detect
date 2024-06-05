import pathlib
import os
import torch
import albumentations as A
import numpy as np
import sys
from torch.utils.data import DataLoader
from transformations import map_class_to_int
from transformations import ComposeDouble, Clip, AlbumentationWrapper, FunctionWrapperDouble
from transformations import normalize_01
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from typing import Dict, List
from skimage.color import rgba2rgb
from skimage.io import imread
from torch.utils.data import Dataset
from torchvision.ops import box_convert
from multiprocessing import Pool
import json
from Faster_RCNN_util import FasterRCNNLightning, get_faster_rcnn_resnet50_fpn,get_fasterrcnn_mobilenet_v3_large_fpn,get_fasterrcnn_mobilenet_v3_large_320_fpn

"""
This module implements train (fine tuning) for a custom RCNN target model.
Basically, it loads a pytorch implementation of faster RCNN model, and train 
it on a new dataset. 
"""

class train_target_model():

    def __init__(self, args):
        super().__init__()
        # hyper-parameters for training
        self.params = {'BATCH_SIZE': 8,
          'LR': 0.001,
          'PRECISION': 32,
          'CLASSES': 21,
          'SEED': 42,
          'EXPERIMENT': 'SUPER STORE DATASET',
          'MAX_EPOCHS': 30,
          'ANCHOR_SIZE': ((32, 64, 128, 256, 512),),
          'ASPECT_RATIOS': ((0.5, 1.0, 2.0),),
          'MIN_SIZE': 1024,
          'MAX_SIZE': 1024,
          'IMG_MEAN': [0.485, 0.456, 0.406],
          'IMG_STD': [0.229, 0.224, 0.225],
          'IOU_THRESHOLD': 0.5
          }
        self.mapping = {
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
        'Versace':20
        }

        self.root_dir = args[0]
        self.params['NEPTUNE_USER_NAME'] = args[1]
        self.params['NEPTUNE_API_KEY'] = args[2]

        # Upload the dataset an transform it to torch dataloader
        self.dataloader_train,self.dataloader_valid,self.dataloader_test = \
            self.form_dataset_to_dataloader()
        # upload the target model from a given path
        self.model = self.init_model()

    def form_dataset_to_dataloader(self):

        """
        Function that reads the dataset images and create 3 dataloaders:
        train, validation and test.
        :return: Train, validation and test pytorch dataloaders.
        """
        roots = os.listdir(self.root_dir)
        inputs, targets, inputs_test, targets_test =[],[],[],[]
        for class_name in roots:
            root = pathlib.Path(os.path.join(self.root_dir,class_name))
            inputs += self.get_filenames_of_path(root / 'input')
            targets += self.get_filenames_of_path(root / 'input label')
            inputs_test += self.get_filenames_of_path(root / 'test')
            targets_test += self.get_filenames_of_path(root / 'test label')
        inputs.sort()
        targets.sort()
        inputs_test.sort()
        targets_test.sort()
        inputs_train, inputs_valid = [],[]
        targets_train, targets_valid = [],[]
        # training validation test split
        for i in range(0,len(inputs),80):
            inputs_train += inputs[i:i+75]
            targets_train += targets[i:i+75]
            inputs_valid += inputs[i+75:i+80]
            targets_valid += targets[i+75:i+80]

        transforms_training,transforms_validation,transforms_test = \
            self.get_transformations()

        return self.get_dataloaders(inputs_train,targets_train,inputs_valid,
                               targets_valid,inputs_test,targets_test,
                               transforms_training, transforms_validation,
                               transforms_test)

    def get_transformations(self):
        """
        Transformations functions - make the original image fit to the object
        detection model.
        :return: Transformations functions.
        """

        # training transformations and augmentations
        transforms_training = ComposeDouble([
            Clip(),
            AlbumentationWrapper(albumentation=A.HorizontalFlip(p=0.5)),
            AlbumentationWrapper(albumentation=A.RandomScale(p=0.5, scale_limit=0.5)),
            # AlbuWrapper(albu=A.VerticalFlip(p=0.5)),
            FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
            FunctionWrapperDouble(normalize_01)
        ])

        # validation transformations
        transforms_validation = ComposeDouble([
            Clip(),
            FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
            FunctionWrapperDouble(normalize_01)
        ])

        # test transformations
        transforms_test = ComposeDouble([
            Clip(),
            FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
            FunctionWrapperDouble(normalize_01)
        ])

        return transforms_training,transforms_validation,transforms_test

    def get_filenames_of_path(self,path: pathlib.Path, ext: str = "*"):
        """
        Returns a list of files in a directory/path. Uses pathlib.
        """
        filenames = [file for file in path.glob(ext) if file.is_file()]
        assert len(filenames) > 0, f"No files found in path: {path}"
        return filenames

    def collate_double(self,batch):
        """
        collate function for the ObjectDetectionDataSet.
        Only used by the dataloader.
        """
        x = [sample["x"] for sample in batch]
        y = [sample["y"] for sample in batch]
        x_name = [sample["x_name"] for sample in batch]
        y_name = [sample["y_name"] for sample in batch]
        return x, y, x_name, y_name

    def get_dataloaders(self,inputs_train,targets_train,inputs_valid,
                        targets_valid,inputs_test,targets_test,
                        transforms_training,transforms_validation,
                        transforms_test):
        """
        Function that recieve lists of images and form it to a pytorch
        dataset and then to dataloader.
        :param inputs_train: Required. List of numpy array representing the
        train images.
        :param targets_train: Required. List of dicts representing the
        labels and bounding boxes for each train image.
        :param inputs_valid: Required. List of numpy array representing the
        validation images.
        :param targets_valid: Required. List of dicts representing the
        labels and bounding boxes for each validation image.
        :param inputs_test: Required. List of numpy array representing the
        test images.
        :param targets_test: Required. List of dicts representing the
        labels and bounding boxes for each test image.
        :param transforms_training: Required. Transformation functions for
        the train images.
        :param transforms_validation: Required. Transformation functions for
        the validation images.
        :param transforms_test: Required. Transformation functions for
        the test images.
        :return: Train, validation and test pytorch dataloaders.
        """

        # random seed
        seed_everything(self.params['SEED'])
        # dataset training
        dataset_train = ObjectDetectionDataSet(inputs=inputs_train,
                                               targets=targets_train,
                                               transform=transforms_training,
                                               use_cache=True,
                                               convert_to_format=None,
                                               mapping=self.mapping)

        # dataset validation
        dataset_valid = ObjectDetectionDataSet(inputs=inputs_valid,
                                               targets=targets_valid,
                                               transform=transforms_validation,
                                               use_cache=False,
                                               convert_to_format=None,
                                               mapping=self.mapping)

        # dataset test
        dataset_test = ObjectDetectionDataSet(inputs=inputs_test,
                                              targets=targets_test,
                                              transform=transforms_test,
                                              use_cache=False,
                                              convert_to_format=None,
                                              mapping=self.mapping)

        # dataloader training
        dataloader_train = DataLoader(dataset=dataset_train,
                                      batch_size=self.params['BATCH_SIZE'],
                                      shuffle=True,
                                      num_workers=0,
                                      collate_fn=self.collate_double)

        # dataloader validation
        dataloader_valid = DataLoader(dataset=dataset_valid,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=0,
                                      collate_fn=self.collate_double)

        # dataloader test
        dataloader_test = DataLoader(dataset=dataset_test,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=0,
                                     collate_fn=self.collate_double)

        return dataloader_train,dataloader_valid,dataloader_test


    def init_model(self):
        """
        Loads a pytorch faster rcnn50 fpn model. Uses Faster RCNN module.
        More information in
        https://pytorch.org/vision/stable/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html
        :return: a pytorch faster rcnn50 fpn model, trained on MSCOCO dataset.
        """
        model = get_faster_rcnn_resnet50_fpn(num_classes=self.params['CLASSES'])
        # lightning init
        task = FasterRCNNLightning(model=model, lr=self.params['LR'],
                                   iou_threshold=self.params['IOU_THRESHOLD'])
        return task

    def log_model_neptune(self,checkpoint_path,save_directory,name,
                          neptune_logger):
        """
        Init neptune logger, to log the training process.
        :return: Log the entire training process.
        """
        checkpoint = torch.load(checkpoint_path)
        model = checkpoint["hyper_parameters"]["model"]
        torch.save(model.state_dict(), save_directory / name)
        neptune_logger.experiment.set_property("checkpoint_name",
                                               checkpoint_path.name)
        neptune_logger.experiment.log_artifact(str(save_directory / name))
        if os.path.isfile(save_directory / name):
            os.remove(save_directory / name)

    def train(self):
        """
        Train the loaded Faster RCNN model on the given dataset.
        :return: Save all the information in the neptune user page.
        """
        # callbacks
        self.checkpoint_callback = ModelCheckpoint(monitor='Validation_mAP',
                                               mode='max')
        self.learningrate_callback = LearningRateMonitor(logging_interval='step', log_momentum=False)
        self.early_stopping_callback = EarlyStopping(monitor='Validation_mAP', patience=50, mode='max')

        # neptune logger
        api_key = self.params['NEPTUNE_API_KEY']
        self.neptune_logger = NeptuneLogger(
            api_key=api_key,
            project_name=f"{self.params['NEPTUNE_USER_NAME']}/Faster-RCNN-Custom",
            experiment_name=self.params['EXPERIMENT'],
            params=self.params
        )

        # trainer init
        self.trainer = Trainer(devices=1, accelerator="auto",
                          precision=self.params['PRECISION'],  # try 16 with
                          # enable_pl_optimizer=False
                          callbacks=[self.checkpoint_callback,
                                     self.learningrate_callback,
                                     self.early_stopping_callback],
                          default_root_dir="...\Experiments",  # where checkpoints are saved to
                          logger=self.neptune_logger,
                          log_every_n_steps=1,
                          num_sanity_val_steps=0,
                          max_epochs = self.params['MAX_EPOCHS']
                          )
        # start training
        self.trainer.fit(self.model,
                    train_dataloader=self.dataloader_train,
                    val_dataloaders=self.dataloader_valid)

    def test(self):
        """
        Test the trained object detection model.
        :return: Print to the console the mAP score for the best dataset and
        AP score for each class.
        """
        # start testing
        self.trainer.test(ckpt_path='best',
                          test_dataloaders=self.dataloader_test)

        # save model
        checkpoint_path = pathlib.Path(self.checkpoint_callback.best_model_path)
        self.log_model_neptune(checkpoint_path=checkpoint_path,
                          save_directory=pathlib.Path.home(),
                          name='best_model.pt',
                          neptune_logger=self.neptune_logger)



class ObjectDetectionDataSet(Dataset):
    """
    Builds a dataset with images and their respective targets.
    A target is expected to be a json file
    and should contain at least a 'boxes' and a 'labels' key.
    inputs and targets are expected to be a list of pathlib.Path objects.

    In case your labels are strings, you can use mapping (a dict) to int-encode them.
    Returns a dict with the following keys: 'x', 'x_name', 'y', 'y_name'
    """

    def __init__(
        self,
        inputs: List[pathlib.Path],
        targets: List[pathlib.Path],
        transform: ComposeDouble = None,
        use_cache: bool = False,
        convert_to_format: str = None,
        mapping: Dict = None,
    ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.use_cache = use_cache
        self.convert_to_format = convert_to_format
        self.mapping = mapping

        if self.use_cache:
            # Use multiprocessing to load images and targets into RAM
            with Pool() as pool:
                self.cached_data = pool.starmap(self.read_images, zip(inputs, targets))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        if self.use_cache:
            x, y = self.cached_data[index]
        else:
            # Select the sample
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            # Load input and target
            x, y = self.read_images(input_ID, target_ID)

        # From RGBA to RGB
        if x.shape[-1] == 4:
            x = rgba2rgb(x)

        # Read boxes
        try:
            boxes = torch.from_numpy(y["boxes"]).to(torch.float32)
        except TypeError:
            boxes = torch.tensor(y["boxes"]).to(torch.float32)

        # Read scores
        if "scores" in y.keys():
            try:
                scores = torch.from_numpy(y["scores"]).to(torch.float32)
            except TypeError:
                scores = torch.tensor(y["scores"]).to(torch.float32)

        # Label Mapping
        if self.mapping:
            labels = map_class_to_int(y["labels"], mapping=self.mapping)
        else:
            labels = y["labels"]

        # Read labels
        try:
            labels = torch.from_numpy(labels).to(torch.int64)
        except TypeError:
            labels = torch.tensor(labels).to(torch.int64)

        # Convert format
        if self.convert_to_format == "xyxy":
            boxes = box_convert(
                boxes, in_fmt="xywh", out_fmt="xyxy"
            )  # transforms boxes from xywh to xyxy format
        elif self.convert_to_format == "xywh":
            boxes = box_convert(
                boxes, in_fmt="xyxy", out_fmt="xywh"
            )  # transforms boxes from xyxy to xywh format

        # Create target
        target = {"boxes": boxes, "labels": labels}

        if "scores" in y.keys():
            target["scores"] = scores

        # Preprocessing
        target = {
            key: value.numpy() for key, value in target.items()
        }  # all tensors should be converted to np.ndarrays

        if self.transform is not None:
            x, target = self.transform(x, target)  # returns np.ndarrays

        # Typecasting
        x = torch.from_numpy(x).type(torch.float32)
        target = {
            key: torch.from_numpy(value).type(torch.int64)
            for key, value in target.items()
        }

        return {
            "x": x,
            "y": target,
            "x_name": self.inputs[index].name,
            "y_name": self.targets[index].name,
        }


    @staticmethod
    def read_images(inp, tar):
        with open(str(tar), "r") as fp:  # fp is the file pointer
            file = json.loads(s=fp.read())
        return imread(inp), file


def train_target_model_demo():
    """
    Demo for train a Faster R-CNN model using Neptune framework.
    :return: a trained Faster R-CNN model.
    """
    """
    example for Faster R-CNN train on SuperStore dataset:
    1. Download SuperStore dataset and store it in a root folder of your choice.
    2. Execute the super_store_util.py module choosing faster_rcnn format. 
    3. Register to Neptune.ai in https://neptune.ai/home, after registration locate your <API key> stored in 
        your personal profile.
    4. Run this module from the command line to train Faster R-CNN model with SupeStore dataset using the following args: 
        [0] - root path - the folder created after running the super_store_util.py module
        [1] - Neptune user name, located at the top of the page after login to Neptune. 
        [2] - Neptune API key, stored in your Neptune's personal profile.
    train_custom_object_detector.py <SupeStore in Faster R-CNN format folder path> <Neptune user name> <Neptune API key>
    example:
    train_custom_object_detector.py faster_rcnn C://Users//Administrator//faster rcnn format johndoe veyJhcGlfYWRkcmVzcyI6lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzYWFmNzlmYS03NzgyLTQ1M2QtODgzU5NGUifQ==
    After the execution of the script the model weights and additional training information 
    will be saved in Neptune framework.
    """
    tctm = train_target_model(sys.argv[1:])
    tctm.train()
    tctm.test()


train_target_model_demo()




