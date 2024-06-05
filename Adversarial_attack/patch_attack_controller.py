import numpy as np
import cv2
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import datetime
import torch
import imutils
from art.estimators.object_detection import PyTorchFasterRCNN
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dpatch_attack import RobustDPatch
from art.attacks.evasion import DPatch
from pathlib import Path
import torchvision
import json
from Object_detection.Object_detection_physical_use_case import evaluate_target_model

"""
This module is a wrapper for patch attack, specifically Dpatch and Robust Dpatch attacks. 
Dpatch attack - Liu, X., Yang, H., Liu, Z., Song, L., Li, H., & Chen, Y. (2018). Dpatch: An adversarial patch attack 
on object detectors.  https://arxiv.org/abs/1806.02299
Robust Dpatch - Lee, M., & Kolter, Z. (2019). On physical adversarial patches for object detection. 
https://arxiv.org/abs/1906.11897
The implementation here is based on IBM adversarial robustness toolbox (ART).
https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#robustdpatch
"""

# MS-COCO dataset classes, used for the digital attack configuration.

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
]

# Superstore dataset classes, used for the physical attack configuration.

SUPER_STORE_INSTANCE_CATEGORY_NAMES= [
    '__background__',
    'Agnesi Polenta',
    'Almond Milk',
    'Snyders',
    'Calvin Klein',
    'Dr Pepper',
    'Flour',
    'Groats',
    'Jack Daniels',
    'Nespresso',
    'Oil',
    'Paco Rabanne',
    'Pixel4',
    'Samsung_s20',
    'Greek Olives',
    'Curry Spice',
    'Chablis Wine',
    'Lindor',
    'Piling Sabon',
    'Tea',
    'Versace'
]
# Superstore 'cheap' items - use for the targeted attacks.
SUPER_STORE_CHEAP_ITEMS = [
    'Agnesi Polenta',
    'Almond Milk',
    'Snyders',
    'Dr Pepper',
    'Flour',
    'Groats',
    'Oil',
    'Greek Olives',
    'Curry Spice',
    'Tea'
]


# Attack configuration
# Explanation about every param can be found at:
# https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#robustdpatch
config = {
        "attack_losses": ("loss_classifier", "loss_box_reg", "loss_objectness",
                          "loss_rpn_box_reg"),
        "cuda_visible_devices": "1",
        "patch_shape": [100, 100, 3],
        "patch_location": [200, 200],
        "crop_range": [0, 0],
        "brightness_range": [0.8, 1.6],
        "rotation_weights": [0.1, 0.1, 0.1, 0.1],
        "blur_scale":1,
        "noise_scale":0.05,
        "sample_size": 3,
        "learning_rate": 12.0,
        "max_iter": 200,
        "batch_size": 1,
        "image_width":640,
        "image_height":480,
        "adaptive_attack": True,
        "adaptive_type": "Ensemble",
        "targeted": True,
        "target_label":"Flour",
        "resume": False,
        "target_model_path":"path to target model",
        "dataset_path" : "path to the data set",
        "style_transfer_model_path":"path to style transfer model",
        "attack_classes": ['Calvin Klein','Chablis Wine','Jack Daniels',
                           'Lindor','Nespresso','Paco Rabanne','Piling Sabon',
                           'Pixel4','Samsung_s20','Versace','Flour'],
        "classes":21,
        "prototypes_path":"path to prototypes"
        }


def get_faster_rcnn_resnet50_fpn(
    num_classes,
    image_mean = [0.485, 0.456, 0.406],
    image_std = [0.229, 0.224, 0.225],
    min_size= 512,
    max_size = 1024,
    **kwargs,
) -> FasterRCNN:
    """
    Function for obtain a skeleton of faster rcnn resnet 50 fpn model
    (pytorch implementation).
    :param num_classes: Required, int, number of classes in the chosen
    dataset.
    :param image_mean: Optional, list of 3 floats (RGB). The mean pixel value
    in the chosen backbone dataset (default is Imagenet).
    :param image_std: Optional, list of 3 floats (RGB). The std pixel value
    in the chosen backbone dataset (default is Imagenet).
    :param min_size: Optional, int, the minimum image size of the backbone
    dataset. Default is 512.
    :param max_size: Optional, int, the maximum image size of the backbone
    dataset. Default is 512.
    :return: A faster rcnn model.
    """
    """Returns the Faster-RCNN model. Default normalization: ImageNet"""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.num_classes = num_classes
    model.image_mean = image_mean
    model.image_std = image_std
    model.min_size = min_size
    model.max_size = max_size

    return model

def init_object_detector():
    """
    Create ART object detector, based on pytorch implementation of faster
    rcnn, more information can be found at  https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/estimators/object_detection.html
    :return: a pre trained faster RCNN object detector, pytorch model object.
    """

    frcnn = PyTorchFasterRCNN(
        clip_values=(0, 255),channels_first=False,
        attack_losses=config['attack_losses'])
    return frcnn

def load_object_detector(target_model_path,classes):
    """
    A wrapper function that load a trained faster rcnn model.
    Uses get_faster_rcnn_resnet50_fpn function for obtaining a skeleton model
    and init_object_detector to transform it to 'ART' format.
    :param target_model_path: Required. String. The path to the trained
    target model.
    :param classes: Required. int. The number of classes in the chosen dataset.
    :return: A trained faster rcnn model.
    """
    model_state_dict = torch.load(target_model_path)
    model = get_faster_rcnn_resnet50_fpn(num_classes=classes)
    # load weights
    model.load_state_dict(model_state_dict)

    # frcnn = PyTorchFasterRCNN(model, channels_first=False, clip_values=(0,255),
    #       attack_losses=(
    #         "loss_classifier", "loss_box_reg", "loss_objectness",
    #         "loss_rpn_box_reg"))

    frcnn = PyTorchFasterRCNN(model, clip_values=(0, 255),
                              channels_first=False,
                              attack_losses=config['attack_losses'])

    return frcnn


def load_images(images_path, image_width=640, image_height=480, plot=False):
    """
    Load images from the given path list to the RAM.
    :param images_path: Required. list of strings describing each image path.
    :param image_width: Optional. int. The image width.
    :param image_height: Optional. int. The image height.
    :param plot: Optional, bool. Plot the uploaded images. Default is false.
    :return: A list of images (4D numpy array).
    """
    images = []
    for i in range(0, len(images_path)):
        image_0 = cv2.imread(images_path[i])
        image_0 = cv2.cvtColor(image_0, cv2.COLOR_BGR2RGB)  # Convert to RGB
        # resize the images to the predefined size.
        image_0 = cv2.resize(image_0,
                             dsize=(image_width, image_height),
                             interpolation=cv2.INTER_CUBIC)
        images.append(image_0)

    # Stack images
    images = np.array(images).astype(np.float32)
    print("image.shape:", images.shape)

    if plot:
        for i in range(images.shape[0]):
            plt.axis("off")
            plt.title("image {}".format(i))
            plt.imshow(images[i].astype(np.uint8), interpolation="nearest")
            plt.show()
            plt.clf()
    return images

def extract_images(root_path):
    """
    Extracts image path from the dataset root path.
    :param root_path: Required. string. The root path.
    :return: List of strings of each image path.
    """
    images_path = []
    for dir in os.listdir(root_path):
        for file in os.listdir(os.path.join(root_path, dir)):
            images_path.append(os.path.join(root_path, dir, file))
    return images_path

def extract_images_super_store_dataset(root_path, attack_classes):
    """
    Extracts image path from the Super Store dataset root path.
    :param root_path: Required. string. The root path.
    :param attack_classes: Required, int. the number of classes in the dataset.
    :return: List of strings of each image path.
    """
    images_path = []
    for instance_class in attack_classes:
        for iter,file in enumerate(os.listdir(
            os.path.join(root_path, instance_class, "test"))):
            if iter<15:
                continue
            images_path.append(os.path.join(root_path, instance_class,"test", file))
    return images_path


def extract_predictions(predictions_, decide_by_threshold = False,
                        COCO_dataset = True):
    """
    Function for extracting a given prediction. Recive the prediction as a
    dictionary and return a list of the predicted object's classification
    and bounding boxes.
    :param predictions_: Required, dictionary of the prediction objects
    classification, bounding boxes and confidence score.
    :param decide_by_threshold: Optional. bool. Select prediction object by a
    given confidence threshold. default is false.
    :param COCO_dataset: Optional. bool. Select coco dataset or super store
    datast. Default is true (COCO dataset).
    :return: a list of the predicted object's classification
    and bounding boxes.
    """
    if COCO_dataset:
        # Get the predicted class
        predictions_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(predictions_["labels"])]
    else:
        predictions_class = [SUPER_STORE_INSTANCE_CATEGORY_NAMES[i] for i in list(predictions_["labels"])]
    print("\npredicted classes:", predictions_class)

    # Get the predicted bounding boxes
    predictions_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(predictions_["boxes"])]

    # Get the predicted prediction score
    predictions_score = list(predictions_["scores"])
    print("predicted score:", predictions_score)

    # Get a list of index with score greater than threshold
    if decide_by_threshold:
        threshold = 0.5
        predictions_t = [predictions_score.index(x) for x in predictions_score if x > threshold]
        if not predictions_t:
            predictions_t = 0
        else:
            predictions_t = predictions_t[-1]
        predictions_boxes = predictions_boxes[: predictions_t + 1]
        predictions_class = predictions_class[: predictions_t + 1]
    else:
        predictions_boxes = predictions_boxes[: 1]
        predictions_class = predictions_class[: 1]

    return predictions_class, predictions_boxes, predictions_class


def plot_image_with_boxes(img, boxes, pred_cls, output_path,iter):
    """
    A function that plot the prediction on the input scene and saved it on
    a given output path.
    :param img: Required. 3D Numpy array. The input scene.
    :param boxes: Required. list of bounding boxes. Each bounding box is a list
    with 4 coordinated in Faster RCNN
    format (
    x1,y1,x2,y2).
    :param pred_cls: Required. List of strings represent the classification
    of the corresponding object.
    :param output_path: Required. String of the output path to save the plot.
    :param iter: An id of the given img.
    :return: Saved the input image with the prediction in the given output
    path.
    """
    text_size = 2
    text_th = 4
    rect_th = 6

    for i in range(len(boxes)):
        c1 = [int(a) for a in boxes[i][0]]
        c2 = [int(a) for a in boxes[i][1]]
        # Draw Rectangle with the coordinates
        cv2.rectangle(img, c1,  c2, color=(0, 255, 0), thickness=rect_th)

        # Write the prediction class
        cv2.putText(img, pred_cls[i], c1, cv2.FONT_HERSHEY_SIMPLEX, text_size,
                    (0, 255, 0), thickness=text_th)

    fig = plt.figure(figsize=(10, 7))
    plt.axis("off")
    plt.imshow(img.astype(np.uint8), interpolation="nearest")
    # plt.show()
    plt.savefig(f'{output_path}/{iter}.jpg', dpi=fig.dpi)


def generate_predictions_and_plot(images, frcnn, output_path, image_num=None):
    """
    Wrapper function that generate prediction and plot it.
    :param images: Required. 4D numpy array of the input image from the
    chosen dataset.
    :param frcnn: Required. A pretrained faster rcnn model in ART format.
    :param output_path: Required. String of the root path to plot the
    predictions images.
    :param image_num: Optional. Plot images with a specified id. Only used
    when sending 1 image to this function. Default is none.
    :return: Plot the images with the corresponding predictions to the given
    output.
    """
    # Make prediction
    predictions = frcnn.predict(x=images, batch_size=1)

    for i in range(images.shape[0]):
        # Process predictions
        predictions_class, predictions_boxes, predictions_class = \
            extract_predictions(predictions[i], True, True)

        # Plot predictions
        if image_num:
            print("\nPredictions image {}:".format(image_num))
            plot_image_with_boxes(img=images[i].copy(), boxes=predictions_boxes,
                                  pred_cls=predictions_class,
                                  output_path=output_path, iter=image_num)
        else:
            print("\nPredictions image {}:".format(i))
            plot_image_with_boxes(img=images[i].copy(), boxes=predictions_boxes,
                                  pred_cls=predictions_class,
                                  output_path=output_path, iter=i)



def DPatch_robust_attack(frcnn, patch_location=None, patch_shape=None, patch_path
= None):
    """
    Robust DPatch attack module.
    :param frcnn: Required. A pretrained faster rcnn model in ART format.
    :param patch_location: Optional. integers list. The patch location
    in the image.
    :param patch_shape: Optional. integers list. The patch size (width,
    height).
    :param patch_path:
    :return:
    """
    if not patch_location is None:
        config["patch_location"] = patch_location
    if not patch_shape is None:
        config["patch_shape"] = patch_shape
    attack = RobustDPatch(
        frcnn,
        patch_shape=config["patch_shape"],
        patch_location=config["patch_location"],
        crop_range=config["crop_range"],
        brightness_range=config["brightness_range"],
        rotation_weights=config["rotation_weights"],
        blur_scale = config["blur_scale"],
        noise_scale = config["noise_scale"],
        adaptive_attack = config["adaptive_attack"],
        adaptive_type = config["adaptive_type"],
        style_transfer_model_path = config["style_transfer_model_path"],
        sample_size=config["sample_size"],
        learning_rate=config["learning_rate"],
        max_iter=config['max_iter'],
        batch_size=config["batch_size"],
        targeted=config['targeted'],
        verbose=True,
        patch_save_dir=patch_path,
        prototypes_path = config["prototypes_path"]
    )

    return attack


def DPatch_attack(frcnn, patch_shape=None, patch_path
= None):
    """
    Robust DPatch attack module.
    :param frcnn: Required. A pretrained faster rcnn model in ART format.
    :param patch_location: Optional. integers list. The patch location
    in the image.
    :param patch_shape: Optional. integers list. The patch size (width,
    height).
    :param patch_path:
    :return:
    """
    if not patch_shape is None:
        config["patch_shape"] = patch_shape
    attack = DPatch(
        frcnn,
        patch_shape=config["patch_shape"],
        learning_rate=config["learning_rate"],
        max_iter=config['max_iter'],
        batch_size=config["batch_size"],
        verbose=True
    )
    return attack

def get_loss(frcnn, x, y):
    """

    :param frcnn: Required. A pretrained faster rcnn model in ART format.
    :param x: Required. 3D numpy array of the tested image.
    :param y: Required. int. The predicted class.
    :return: Loss of the generated attack.
    """
    frcnn._model.train()
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    image_tensor_list = list()
    x = np.expand_dims(x,axis=0)
    for i in range(x.shape[0]):
        if frcnn.clip_values is not None:
            img = transform(x[i] / frcnn.clip_values[1]).to(frcnn._device)
        else:
            img = transform(x[i]).to(frcnn._device)
        image_tensor_list.append(img)

    for i, y_i in enumerate(y):
        y[i]["boxes"] = torch.from_numpy(y_i["boxes"]).type(torch.float).to(
            frcnn._device)
        y[i]["labels"] = torch.from_numpy(y_i["labels"]).type(torch.int64).to(
            frcnn._device)
        y[i]["scores"] = torch.from_numpy(y_i["scores"]).to(frcnn._device)

    loss = frcnn._model(image_tensor_list, y)
    for loss_type in ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]:
        loss[loss_type] = loss[loss_type].cpu().detach().numpy().item()
    return loss

def append_loss_history(loss_history, current_loss):
    """
    Extract loss history.
    :param loss_history: Required. All the loss logs.
    :param current_loss: Required. The current loss log.
    :return: All the loss logs with the current loss.
    """
    for loss in ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]:
        loss_history[loss] += [current_loss[loss]]
    return loss_history

def get_attack_stats(frcnn,patched_image, y,loss_history,
                     config, output="output"):
    """
    A wrapper function that save all the attack stats.
    :param frcnn: Required. A pretrained faster rcnn model in ART format.
    :param patched_image: Required. 3D numpy array representing the image
    with the patch.
    :param y: Required. int. The model prediction.
    :param loss_history: Required. All the loss logs.
    :param config: Required. The configuration dict containing the
    configuration stats.
    :param output: Optional. String to output path. default is "output".
    :return: Dump the attack stats to a json file.
    """
    loss = get_loss(frcnn, patched_image, y)
    loss_history = append_loss_history(loss_history, loss)
    with open(os.path.join(output, "loss_history.json"), "w") as file:
            file.write(json.dumps(loss_history))
    with open(os.path.join(output,"params.json"), "w") as outfile:
        json.dump(config, outfile)

def save_patch(patch, patch_size,output_path="output"):
    """
    Function for saving the patch to the disk.
    :param patch: Required. 3D numpy array represent the patch.
    :param patch_size: Required. integers list of the patch width and height.
    :param output_path: Optional. String to output path. default is "output".
    :return: Save the patch as an image in the given output path.
    """
    fig = plt.figure(figsize=(patch_size))
    plt.axis("off")
    plt.imshow(patch.astype(np.uint8), interpolation="nearest")
    size = ""
    if patch_size[0] > 2:
        size = "big"
    plt.savefig(f'{output_path}/patch_{size}.png', dpi=fig.dpi)
    with open(f'{output_path}/patch.npy', 'wb') as f:
        np.save(f,patch)

def predict_detections(x,frcnn,only_main_obj):
    """
    object detection Prediction function.
    :param x: Required. 3D numpy array representing the image.
    :param frcnn: Required. A pretrained faster rcnn model in ART format.
    :param only_main_obj: Required. Predict only the main object.
    :return: Object detection Prediction (dictionary).
    """
    y = frcnn.predict(x)
    if only_main_obj:
        for i in range(len(y)):
            y[i]['boxes'] = y[i]['boxes'][:1]
            y[i]['labels'] = y[i]['labels'][:1]
            y[i]['scores'] = y[i]['scores'][:1]
    return y

def change_labels(y,label):
    """
    Change label function, required in the target scenario.
    :param y: Required. Integer list of all labels.
    :param label: Required. int. the label (index) to be change.
    :return:
    """
    for i in range(len(y)):
        y[i]['labels'] = (np.ones(len(y[i]['labels'])) * label).astype(int)
    return y

def create_output_folders():
    """
    Function for crating output folders.
    :return: Create output folders.
    """
    time = datetime.datetime.now().strftime("%d-%m-%Y_%H;%M")
    output_path = f"output/{time}"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    reg_images_detection_path = f'{output_path}/benign_detection'
    Path(reg_images_detection_path).mkdir(parents=True, exist_ok=True)
    patch_path = f'{output_path}/patch'
    Path(patch_path).mkdir(parents=True, exist_ok=True)
    adv_path = f'{output_path}/adversarial'
    Path(adv_path).mkdir(parents=True, exist_ok=True)
    adv_dec_path = f'{output_path}/adversarial_dec'
    Path(adv_dec_path).mkdir(parents=True, exist_ok=True)
    new_img_with_patch = f'{output_path}/new_adversarial'
    Path(new_img_with_patch).mkdir(parents=True, exist_ok=True)
    new_adv_dec_path = f'{output_path}/new_adversarial_dec'
    Path(new_adv_dec_path).mkdir(parents=True, exist_ok=True)
    attack_stats = f'{output_path}/attack_stats'
    Path(attack_stats).mkdir(parents=True, exist_ok=True)
    return reg_images_detection_path,patch_path,adv_path,adv_dec_path,\
           new_img_with_patch, new_adv_dec_path,attack_stats

def get_patch_location_faster_rcnn_format(bbox,patch):
    """
    Function that calculate the best location of the patch in the scene.
    :param bbox: req. list of 4 coordinates (x1,y1,x2,y2).
    :param patch: req. numpy format of the patch.
    :return: patch upper left location by x,y coordinates (x1,y1) and the patch itself in numpy format.
    """
    object_width = bbox[2]-bbox[0]
    object_height = bbox[3]- bbox[1]
    patch_width = object_width * 0.5
    # More control on the patch size in the crafting phase.
    # if patch_width < 300 and patch_width > 60:
    #     patch = imutils.resize(patch, width=int(patch_width))
    # if patch_width < 60:
    #     patch = imutils.resize(patch, width=60)
    if (bbox[1] + object_height) < patch.shape[1]:
        patch = imutils.resize(patch, width=int(object_height * 0.5))

    object_x_middle = (bbox[2]+bbox[0]) / 2
    object_y_middle = (bbox[3]+bbox[1]) / 2

    x_patch_location = object_x_middle - (patch.shape[0] / 2)
    y_patch_location = object_y_middle - (patch.shape[1] / 2)
    # x_patch_location = object_x_middle + (patch.shape[0] / 2)
    # y_patch_location = object_y_middle + (patch.shape[1] / 2)
    return (int(y_patch_location), int(x_patch_location)), patch

def apply_patch(image,patch,patch_location):
    """
    Function that place the patch in the frame.
    :param image: req. numpy array of the frame.
    :param patch: req. numpy format of the patch.
    :param patch_location:  req. tuple of 2 ints, representing the patch upper left location by x,y coordinates (x1,y1).
    :return: The image with the patch placed into it.
    """
    x_patch = image.copy()
    patch_local = patch.copy()
    x_1, y_1 = patch_location
    x_2, y_2 = x_1 + patch_local.shape[0], y_1 + patch_local.shape[1]
    if x_2 > x_patch.shape[0] or y_2 > x_patch.shape[1]:  # pragma: no cover
        raise ValueError("The patch (partially) lies outside the image.")
    x_patch[x_1:x_2, y_1:y_2, :] = patch_local
    return x_patch

def load_patch(patch_path,patch_as_np=True):
    """
    Load existing patch from path.
    :param patch_path: req. str. The path to the patch.
    :param patch_as_np: req. bool. If the patch saved in numpy format.
    :return: The patch in numpy format.
    """
    if patch_as_np:
        patch = np.load(patch_path)
    else:
        exist_patch = cv2.imread(patch_path)
        patch = cv2.cvtColor(exist_patch, cv2.COLOR_BGR2RGB)
    return patch

def apply_patch_to_dataset(dataset_path, patch_path,output_path,
               target_model_path,number_of_classes, target_class='Dr Pepper'):
    """
    Demo function for generating and evaluating digital or physical patch attacks. Check "attack_demo.py" module
    for more information.
    :param dataset_path: req. str. root path to super store dataset in faster rcnn format
    example: "root_dir/super_store/faster rcnn format".
    :param patch_path: req. str. path to a patch in numpy format.
    example: "patches/Almond Milk/patch.npy"
    :param output_path: req. str. output path which scenes with patches and dection plot will be saved to.
    example: "output/test_patch"
    :param target_model_path: req. str. target_model_path
    :param number_of_classes: req. int. number of classes in the target model.
    (for model trained with Super store dataset is 21).
    :param target_class: opt. str. the class the given patch is target on.
    :return: Outputs the attack results to the given output path.
    """
    patch = load_patch(patch_path)
    patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    target_model = evaluate_target_model(target_model_path,
                 number_of_classes,output_path="",
                 dataset_path=dataset_path,custom_classes= True)
    result_dict = target_model.test_set_prediction(confidence_threshold=0.5,
                                                   plot=False)
    images_path = target_model.get_images_paths(dataset_path)
    image_id = [image_path.stem for image_path in images_path]
    image_dict = dict(zip(image_id, images_path))
    misclassification_rate = 0
    cheap_item_label = 0
    target_class_rate = 0
    true_class_list = []
    predicted_list = []
    for idx,key in enumerate(result_dict):
        image = cv2.imread(str(image_dict[key]))
        # image = zoom_in(image)
        bbox = result_dict[key]['boxes'][0].cpu().numpy()
        patch_loc, new_patch = get_patch_location_faster_rcnn_format(bbox,patch)
        patched_image_file = apply_patch(image, new_patch, patch_loc)
        patched_image = [{'x_name':key,'x':patched_image_file}]
        target_model = evaluate_target_model(target_model_path,number_of_classes,
                                             output_path,patched_image)
        output_dict = target_model.test_set_prediction(
            confidence_threshold=0.2, plot=False)
        if not output_dict:
            print("no classification")
        else:
            true_class_index = int(result_dict[key]['labels'][0])
            target_index = \
                SUPER_STORE_INSTANCE_CATEGORY_NAMES.index(target_class)
            if SUPER_STORE_INSTANCE_CATEGORY_NAMES[
                int(output_dict[key]['labels'][0])] in SUPER_STORE_CHEAP_ITEMS:
                cheap_item_label+=1
            if output_dict[key]['labels'][0]!= result_dict[key][
                'labels'][0]:
                misclassification_rate+=1
            if output_dict[key]['labels'][0]== target_index:
                target_class_rate+=1
                cv2.imwrite(f'{output_path}/{key}_patched.jpg',
                            patched_image_file)
            else:
                print(f"Attack failed for {key}, prediction is "
                      f"{SUPER_STORE_INSTANCE_CATEGORY_NAMES[output_dict[key]['labels'][0]]}  ")

            ground_truth = [idx,true_class_index,1.0]
            ground_truth +=result_dict[key]['boxes'][0].tolist()
            prediction = [idx,int(output_dict[key]['labels'][0]),
                          float(output_dict[key]['scores'][0])]
            prediction+=output_dict[key]['boxes'][0].tolist()
            true_class_list.append(ground_truth)
            predicted_list.append(prediction)

    print(f"Misclassification_rate: "
          f"{misclassification_rate / len(result_dict) * 100}")
    print(f"cheap item label rate: "
          f"{cheap_item_label / len(result_dict) * 100}")
    print(f"Target class rate: "
          f"{target_class_rate / len(result_dict) * 100}")
    mean_average_precision = target_model.mean_average_precision(
        predicted_list,true_class_list)
    print(f"mAP: {mean_average_precision*100}")


def digital_attack_demo():
    """
    Demo function that generates digital patches. Uses MSCOCO dataset and the Dpatch attack.
    :return: The crafted adversarial patch is stored in an output folder in a numpy format.
    """

    # init output paths
    reg_images_detection_path, patch_path, adv_path, adv_dec_path, \
    new_img_with_patch, new_adv_dec_path, attack_stats =create_output_folders()
    # init target model and load images
    frcnn = init_object_detector()
    images_path = extract_images(config['dataset_path'])
    images = load_images(images_path)
    split_images = np.array_split(images,(np.ceil(len(images)/10)).astype(int))
    y = []
    for batch in split_images:
        # make benign prediction
        generate_predictions_and_plot(batch, frcnn, reg_images_detection_path)
        y += predict_detections(batch, frcnn,only_main_obj=True)
    # attack!
    attack = DPatch_robust_attack(frcnn, patch_path = patch_path)
    loss_history = {"loss_classifier": [], "loss_box_reg": [], "loss_objectness": [], "loss_rpn_box_reg": []}
    y = change_labels(y,label=52)
    patch = attack.generate(x=images,y=y)
    save_patch(patch,(config["patch_shape"][0]/100,config["patch_shape"][1]/100),
               output_path=patch_path)
    save_patch(patch,(10,7),output_path=patch_path)
    patched_image = attack.apply_patch(images)
    get_attack_stats(frcnn, patched_image[0], y,loss_history,config,attack_stats)

    # check attack on adversarial (patched) images
    images_path = []
    images_path.append("images/apple.jpeg")
    images_path.append("images/apple1.jpeg")
    images = load_images(images_path)
    patched_image = attack.apply_patch(images,patch)
    # plot_image_with_patches(patched_image,new_img_with_patch)
    generate_predictions_and_plot(patched_image, frcnn, new_adv_dec_path)


def physical_attack_demo():
    """
    Demo function that generates physical patches. Uses SuperStore dataset and the Dpatch q robust Dpatch attack.
    :return: The crafted adversarial patch is stored in an output folder in a numpy format.
    """
    # Init output paths
    reg_images_detection_path, patch_path, adv_path, adv_dec_path, \
    new_img_with_patch, new_adv_dec_path, attack_stats = create_output_folders()
    # Init target model and load images
    frcnn = load_object_detector(config['target_model_path'],
                                 config['classes'])
    # Extract images path from the dataset path
    images_path = extract_images_super_store_dataset(config['dataset_path'],
                                                     config['attack_classes'])
    # Load images to RAM
    images = load_images(images_path, image_width=config['image_width'],
                         image_height=config['image_height'])
    split_images = np.array_split(images,
                                  (np.ceil(len(images) / 8)).astype(int))

    # Get benign prediction
    y = []
    for batch in split_images:
        # generate_predictions_and_plot(batch, frcnn, reg_images_detection_path)
        y += predict_detections(batch, frcnn, only_main_obj=True)

    # Init attack
    loss_history = {"loss_classifier": [], "loss_box_reg": [],
                    "loss_objectness": [], "loss_rpn_box_reg": []}
    # Set target label
    y = change_labels(y, label=SUPER_STORE_INSTANCE_CATEGORY_NAMES.index(
        config['target_label']))

    # Generate patch attack
    dpatch = False
    if dpatch:
        attack = DPatch_attack(frcnn, patch_path=patch_path)
        patch = attack.generate(x=images, y=None,target_label=None)
    else:
        attack = DPatch_robust_attack(frcnn, patch_path=patch_path)
        patch = attack.generate(x=images, y=y)
    # Save the final patch
    save_patch(patch,
               (config["patch_shape"][0] / 100, config["patch_shape"][1] / 100),
               output_path=patch_path)
    save_patch(patch, (10, 7), output_path=patch_path)
    patched_image = attack.apply_patch(images)
    # Save attack stats
    get_attack_stats(frcnn, patched_image[0], y, loss_history, config,
                     attack_stats)


