import time
import os
import requests
import torch
from PIL import Image , ImageEnhance
from Datasets_util.MSCOCO_util import COCO_INSTANCE_CATEGORY_NAMES,COCO_INSTANCE_CATEGORY_NAMES_80_CLASSES
from Object_detection.Object_detection_physical_use_case import SUPER_STORE_INSTANCE_CATEGORY_NAMES
from mmdet.apis import inference_detector,show_result_pyplot
import numpy as np
import cv2
import imutils
from pathlib import Path
import shutil
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim

os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


"""
A module for detection adversarial patches using scene manipulations. 
"""

class scene_processor_detector():
    """
    Class representing the scene processing detector. Contain multiple image
    processing techniques which is utilized by the detector and model inference on those manipulations.
    """

    def __init__(self,dataset,target_model,path_to_style_transfer_models,use_case="physical",
                 optimization_params = None, output_path = ""):
        super().__init__()
        self.dataset = dataset
        self.target_model = target_model
        sth = style_transfer_helper()
        self.style_transfer_model = sth.get_style_transfer_models(path_to_style_transfer_models)
        self.optimize = False
        if optimization_params:
            self.optimize = True
            self.optimization_params = optimization_params
        self.use_case = use_case
        self.output_path = os.path.join(output_path,"Scene_processing_detector")
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        self.processed_image_folders = self.create_output_folders()

    def set_dataset(self,dataset):
        """
        Set the dataset which will be used by the image processing techniques.
        :param dataset: req. numpy array of images.
        :return: --
        """
        self.dataset = dataset

    def get_dict(self):
        """
        Create an empty dictionary for SuperStore dataset predictions.
        :return: empty dictionary
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
        }

    def transform_to_blur(self,img, output):
        """
        Function for blur transformation.
        :param img: req. numpy format (3 dims). The input image.
        :param output: req. string. A path to save the transformed image.
        :return: A blur image, that saved in the given output.
        """

        if self.optimize:
            blur = cv2.blur(img, (self.optimization_params[0], self.optimization_params[0]))
        else:
            if self.use_case=="physical":
                blur = cv2.blur(img, (12, 12))
            else:
                blur = cv2.blur(img, (6, 6))
        if output:
            cv2.imwrite(output, blur)
        else:
            return blur

    def transform_to_sharpen(self, img, output):
        """
        Function for sharpen transformation.
        :param img: req. numpy format (3 dims). The input image.
        :param output: req. string. A path to save the transformed image.
        :return: A sharpen image, that saved in the given output.
        """
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        img = cv2.filter2D(img, -1, sharpen_kernel)
        if output:
            cv2.imwrite(output, img)
        else:
            return img

    def random_noise(self,img,output):
        """
        Function for random noise transformation.
        :param img: req. numpy format (3 dims). The input image.
        :param output: req. string. A path to save the transformed image.
        :return: A random noise image, that saved in the given output.
        """
        noise = (np.random.random_sample(img.shape) - 0.5) * 255
        if self.optimize:
            new_np_image = img + self.optimization_params[2] * noise
        else:
            new_np_image = img + 0.25 * noise
        new_np_image = img + 0.35 * noise
        new_np_image = new_np_image.astype(np.uint8)
        if output==None:
            return new_np_image
        else:
            cv2.imwrite(output, new_np_image)


    def darkness(self,img,output):
        """
        Function for darkness transformation.
        :param img: req. PIL format (3 dims). The input image.
        :param output: req. string. A path to save the transformed image.
        :return: A darkness image, that saved in the given output.
        """

        img = Image.fromarray(img)
        enhancer = ImageEnhance.Brightness(img)
        # img_dark = enhancer.enhance(0.1)
        if self.optimize:
            img_dark = enhancer.enhance(self.optimization_params[1])
        else:
            if self.use_case == "physical":
                img_dark = enhancer.enhance(0.01)
            else:
                img_dark = enhancer.enhance(0.1)
        if output==None:
            dark = np.asarray(img_dark)
            return dark
        else:
            img_dark.save(output)

    def style_transfer(self,img,img_path,output):
        """
        Function for style_transfer transformation using deep AI API.
        :param img_path: req. string. The path to the input image.
        :param output: req. string. A path to save the transformed image.
        :param style_path: opt. A path for the style image. In none is
        provided, the original image consider as the style image.
        :return: A 'styled' image, that saved in the given output.
        """
        img_path = img_path+"/temp.jpg"
        cv2.imwrite(img_path,img)
        r = requests.post(
            "https://api.deepai.org/api/fast-style-transfer",
            files={
                    'content':open(img_path, 'rb'),
                    'style': open(img_path, 'rb')
                },
            headers={'api-key': '10562812-d5b1-40b7-b79f-309ef6fcd2bb'}
        )
        # Download remote and save locally
        data = requests.get(r.json()['output_url'])
        if output:
            with open(output+'/output.jpg', 'wb') as file:
                file.write(data.content)
        else:
            return data
        styled_image = cv2.imread(output+'/output.jpg')
        styled_image = imutils.resize(styled_image, width=img.shape[1])
        return styled_image

    def create_image_processing_dict(self):
        """
        Create empty dictionary which will contain all the processed images and the object detector
        prediction coreespond to those images. .
        :return: empty dictionary.
        """
        scenes = {}
        scenes['blur'] = {}
        scenes['blur']['prediction_dict'] = self.get_dict()
        scenes['style_transfer'] = {}
        scenes['style_transfer']['prediction_dict'] = self.get_dict()
        scenes['darkness'] = {}
        scenes['darkness']['prediction_dict'] = self.get_dict()
        if self.use_case == "digital":
            scenes['sharp'] = {}
            scenes['sharp']['prediction_dict'] = self.get_dict()
        else:
            scenes['random_noise'] = {}
            scenes['random_noise']['prediction_dict'] = self.get_dict()
        self.prediction_for_image_processing_tech = scenes

    def scene_transformer(self,style_transfer_path):
        """
        Wrapper function that perform image processing techniques on the given dataset by the given use case.
        :param style_transfer_path: req. str.
        :return: Dictionary with transformed images using the image processing technique.
        """
        scenes = {}
        scenes['blur'] = {}
        scenes['blur']['raw_image'] = self.transform_to_blur(self.dataset, None)
        scenes['style_transfer'] = {}
        scenes['darkness'] = {}
        scenes['darkness']['raw_image'] = self.darkness(self.dataset, None)

        if self.use_case=="digital":
            scenes['sharp'] = {}
            scenes['sharp']['raw_image'] = self.transform_to_sharpen(self.dataset,None)
            scenes['style_transfer']['raw_image'] =  self.style_transfer(self.dataset,style_transfer_path,style_transfer_path)
        else:
            scenes['random_noise'] = {}
            scenes['random_noise']['raw_image'] = self.random_noise(self.dataset, None)
            sth = style_transfer_helper()
            scenes['style_transfer']['raw_image'] = sth.style_img(self.dataset, self.dataset, self.style_transfer_model)
        return scenes

    def create_faster_rcnn_result_dict_new(self, result):
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
        for idx, class_instance in enumerate(result):
            if len(class_instance) > 0:
                class_prob[idx] = class_instance[0][4]
        labels, scores, boxes = [], [], []
        if len(class_prob) > 0:
            for prediction_id in class_prob.keys():
                if class_prob[prediction_id] > 0.3:
                    labels.append(prediction_id + 1)
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

    def predict(self,frame,mmdetection_model):
        """
        Function that uses object detection model to detect objects in a given frame.
        :param frame:req. numpy format. An image.
        :param mmdetection_model: req. bool. If the object detector used is taken from mmdetection package.
        :return: A prediction of which object is located in the given frame and their bounding boxes.
                In Faster R-CNN format.
        """
        if not mmdetection_model:
            with torch.no_grad():
                frames = np.expand_dims(frame, axis=0)
                # frames = frames.to(self.device)
                return self.target_model.predict(frames, batch_size=1)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = inference_detector(self.target_model, frame)
            # show_result_pyplot(self.target_model, frame, result,score_thr=0.65)
            result = self.create_faster_rcnn_result_dict_new(result)
            return result

    def get_main_object_label_and_confidence(self,main_object_bbox,predictions,mmdetection_model=True):
        """
        Function that extract the object index with the predicted object bounding box that fit the best to
        the main object bounding box, suited for MSCOCO dataset.
        :param main_object_bbox: req. list. The bounding box of the main object in the frame.
        :param predictions: req. dictionary. Dict of predictions.
        :param mmdetection_model: req. bool. If the object detector used is taken from mmdetection package.
        :return: The label and confidence of the main object
        """
        best_iou = 0
        best_idx = 0
        target_model_bbox_preds = predictions[0]['boxes']
        target_model_confidence = predictions[0]['scores']
        target_model_labels = predictions[0]['labels']
        for idx, box_pred in enumerate(target_model_bbox_preds):
            if target_model_confidence[idx] > 0.4:
                curr_iou = self.get_iou(np.array(main_object_bbox), box_pred)
                if curr_iou > best_iou:
                    best_iou = curr_iou
                    best_idx = idx
        if not mmdetection_model:
            label = COCO_INSTANCE_CATEGORY_NAMES[target_model_labels[best_idx]]
        else:
            label = COCO_INSTANCE_CATEGORY_NAMES_80_CLASSES[target_model_labels[best_idx]]
        confidence = float(target_model_confidence[best_idx])
        return label,confidence

    def plot_prediction(self, frame, preds, output_path,img_id,mmdetection_model = True):
        """
        Function that plot the predictions on the original images.
        :param prediction: required. dictionary representing the prediction.
        :param x: required. 3D numpy array. The original image.
        :param x_name: required. string. The image id.
        :param mmdetection_model: required. bool. If the object detector used is taken from mmdetection package.
        :return: Plots the predictions on the original images and save it on
        predefined path.
        """

        for i in range(len(preds[0]['boxes'])):
            if float(preds[0]['scores'][i])>0.3:
                x1 = int(preds[0]['boxes'][i][0])
                y1 = int(preds[0]['boxes'][i][1])
                x2 = int(preds[0]['boxes'][i][2])
                y2 = int(preds[0]['boxes'][i][3])
                detected_object = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                if self.use_case=="physical":
                    label = SUPER_STORE_INSTANCE_CATEGORY_NAMES[int(preds[0]['labels'][i])]
                else:
                    if not mmdetection_model:
                        label = COCO_INSTANCE_CATEGORY_NAMES[preds[0]['labels'][i]]
                    else:
                        label = COCO_INSTANCE_CATEGORY_NAMES_80_CLASSES[preds[0]['labels'][i]]
                frame = cv2.putText(detected_object, label, (x1, y1),cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        filename = os.path.join(output_path,
                                f"{str(img_id)}.jpg")
        cv2.imwrite(filename, frame)


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

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bb1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bb2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        return iou

    def create_output_folders(self):
        """
        Function that creates outputs folder for the detector outputs.
        :return: --
        """
        blur_scale = f'{self.output_path}/blur_scale'
        Path(blur_scale).mkdir(parents=True, exist_ok=True)
        darkness_scale = f'{self.output_path}/darkness_scale'
        Path(darkness_scale).mkdir(parents=True, exist_ok=True)
        style_transfer_scale = f'{self.output_path}/style_transfer_scale'
        Path(style_transfer_scale).mkdir(parents=True, exist_ok=True)
        if self.use_case == "digital":
            sharpen_scale = f'{self.output_path}/sharpen_scale'
            Path(sharpen_scale).mkdir(parents=True, exist_ok=True)
            processed_image_folders = [blur_scale, sharpen_scale, style_transfer_scale, darkness_scale]
        else:
            random_noise_scale = f'{self.output_path}/random_noise_scale'
            Path(random_noise_scale).mkdir(parents=True, exist_ok=True)
            processed_image_folders = [blur_scale, random_noise_scale, style_transfer_scale, darkness_scale]

        return processed_image_folders




class style_transfer_helper():
    """
    Class implementing arbitrary style transfer.
    The implementation is based on PyTorch-AdaIN-StyleTransfer published in
    Huang, X., & Belongie, S. (2017). Arbitrary style transfer in real-time with adaptive instance normalization.
    In Proceedings of the IEEE international conference on computer vision (pp. 1501-1510).‏
    using the following github project:
    https://github.com/MAlberts99/PyTorch-AdaIN-StyleTransfer

    MIT License

    Copyright (c) 2020 MAlberts99

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """

    def __init__(self):
        super().__init__()
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")

    def get_style_transfer_models(self,path):
        path_check = os.path.join(path,
                                  "StyleTransfer.tar")
        state_vgg = torch.load(os.path.join(path, "StyleTransfer_normalised.pth"),
                               map_location=torch.device("cpu"))
        return (path_check, state_vgg)

    def style_img(self,content, style, models):
        """
        Wrapper function for arbitrary style transfer which takes two images - style and content
        and blends them together.
        :param content: req. numpy. A content image.
        :param style: req. numpy. A style image.
        :param models: Models for performing the arbitrary style transfer.
        :return: An arbitrary style transfer image.
        """
        transform = self.get_transforms()
        content = self.process_image(content, transform)
        style = self.process_image(style, transform)
        path_check, state_vgg = models
        network = StyleTransferNetwork(self.device, state_vgg, train=False,
                                       load_fromstate=True,
                                       load_path=path_check)
        alpha = 1.0

        out = network(style, content, alpha).cpu()

        # convert to grid/image
        out = torchvision.utils.make_grid(out.clamp(min=-1, max=1), nrow=3,
                                          scale_each=True, normalize=True)

        out1 = (out.numpy().transpose(1, 2, 0)) * 255
        return out1

    def get_transforms(self):
        """
        Transformation function.
        :return: Transformations.
        """
        return transforms.Compose([transforms.Resize(512),
                                   # transforms.CenterCrop(256),
                                   transforms.ToTensor()])

    def process_image(self,image, transform):
        """
        Function that pre-process an image.
        :param image: req. numpy. an image.
        :param transform: req. transformation instance which hold all the transformations functions required.
        :return: The processed image.
        """
        image = Image.fromarray(image)
        image = transform(image).unsqueeze(0).to(device)
        return image


    def calc_mean_std(self,input, eps=1e-5):
        """
        Calculates mean and std channel-wise.
        :param input: req. tensor. a set of images.
        :param eps: req. float. epsilon.
        :return:
        """
        batch_size, channels = input.shape[:2]

        reshaped = input.view(batch_size, channels, -1)  # Reshape channel wise
        mean = torch.mean(reshaped, dim=2).view(batch_size, channels, 1, 1)  # Calculat mean and reshape
        std = torch.sqrt(torch.var(reshaped, dim=2) + eps).view(batch_size, channels, 1,
                                                                1)  # Calculate variance, add epsilon (avoid 0 division),
        # calculate std and reshape
        return mean, std

    def AdaIn(self,content, style):
        assert content.shape[:2] == style.shape[:2]  # Only first two dim, such that different image sizes is possible
        batch_size, n_channels = content.shape[:2]
        mean_content, std_content = self.calc_mean_std(content)
        mean_style, std_style = self.calc_mean_std(style)

        output = std_style * (
                    (content - mean_content) / (std_content)) + mean_style  # Normalise, then modify mean and std
        return output

    def Content_loss(self,input, target):
        """
        # Content loss is a MSE Loss
        :param input: req. numpy. an image.
        :param target: req. numpy. the target image.
        :return: The min square error between the input image and the target image.
        """
        loss = F.mse_loss(input, target)
        return loss

    def Style_loss(self,input, target):
        """
        Uses mean and std to calculate the style loss.
        :param input: req. numpy. an image.
        :param target: req. numpy. the target image.
        :return: The difference between a set of input images and target images.
        """
        mean_loss, std_loss = 0, 0

        for input_layer, target_layer in zip(input, target):
            mean_input_layer, std_input_layer = self.calc_mean_std(input_layer)
            mean_target_layer, std_target_layer = self.calc_mean_std(target_layer)

            mean_loss += F.mse_loss(mean_input_layer, mean_target_layer)
            std_loss += F.mse_loss(std_input_layer, std_target_layer)

        return mean_loss + std_loss

# The style transfer network
class StyleTransferNetwork(nn.Module,style_transfer_helper):
    def __init__(self,
               device, # "cpu" for cpu, "cuda" for gpu
               enc_state_dict, # The state dict of the pretrained vgg19
               learning_rate=1e-4,
               learning_rate_decay=5e-5, # Decay parameter for the learning rate
               gamma=2.0, # Controls importance of StyleLoss vs ContentLoss, Loss = gamma*StyleLoss + ContentLoss
               train=True, # Wether or not network is training
               load_fromstate=False, # Load from checkpoint?
               load_path=None # Path to load checkpoint
               ):
        super().__init__()

        assert device in ["cpu", "cuda"]
        if load_fromstate and not os.path.isfile(load_path):
          raise ValueError("Checkpoint file not found")


        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.train = train
        self.gamma = gamma

        self.encoder = Style_transfer_encoder(enc_state_dict, device) # A pretrained vgg19 is used as the encoder
        self.decoder = Style_transfer_decoder().to(device)

        self.optimiser = optim.Adam(self.decoder.parameters(), lr=self.learning_rate)
        self.iters = 0

        if load_fromstate:
          state = torch.load(load_path)
          self.decoder.load_state_dict(state["Decoder"])
          self.optimiser.load_state_dict(state["Optimiser"])
          self.iters = state["iters"]

    def forward(self, style, content, alpha=1.0): # Alpha can be used while testing to control the importance of the transferred style

        # Encode style and content
        layers_style = self.encoder(style, self.train) # if train: returns all states
        layer_content = self.encoder(content, False) # for the content only the last layer is important

        # Transfer Style
        if self.train:
          style_applied = self.AdaIn(layer_content, layers_style[-1]) # Last layer is "style" layer
        else:
          style_applied = alpha*self.AdaIn(layer_content, layers_style) + (1-alpha)*layer_content # Alpha controls magnitude of style

        # Scale up
        style_applied_upscaled = self.decoder(style_applied)
        if not self.train:
          return style_applied_upscaled # When not training return transformed image

        # Compute Loss
        layers_style_applied = self.encoder(style_applied_upscaled, self.train)

        content_loss = self.Content_loss(layers_style_applied[-1], layer_content)
        style_loss = self.Style_loss(layers_style_applied, layers_style)

        loss_comb = content_loss + self.gamma*style_loss

        return loss_comb, content_loss, style_loss

# The decoder is a reversed vgg19 up to ReLU 4.1. To note is that the last layer is not activated.

class Style_transfer_decoder(nn.Module):
  def __init__(self):
    super().__init__()

    self.padding = nn.ReflectionPad2d(padding=1) # Using reflection padding as described in vgg19
    self.UpSample = nn.Upsample(scale_factor=2, mode="nearest")

    self.conv4_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=0)

    self.conv3_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0)
    self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0)
    self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0)
    self.conv3_4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0)

    self.conv2_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)
    self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0)

    self.conv1_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
    self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=0)


  def forward(self, x):
    out = self.UpSample(F.relu(self.conv4_1(self.padding(x))))

    out = F.relu(self.conv3_1(self.padding(out)))
    out = F.relu(self.conv3_2(self.padding(out)))
    out = F.relu(self.conv3_3(self.padding(out)))
    out = self.UpSample(F.relu(self.conv3_4(self.padding(out))))

    out = F.relu(self.conv2_1(self.padding(out)))
    out = self.UpSample(F.relu(self.conv2_2(self.padding(out))))

    out = F.relu(self.conv1_1(self.padding(out)))
    out = self.conv1_2(self.padding(out))
    return out

# A vgg19 Sequential which is used up to Relu 4.1. To note is that the
# first layer is a 3,3 convolution, different from a standard vgg19

class Style_transfer_encoder(nn.Module):
    def __init__(self, state_dict, device):
        super().__init__()
        self.vgg19 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True), # First layer from which Style Loss is calculated
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(inplace=True), # Second layer from which Style Loss is calculated
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
            nn.ReflectionPad2d(padding=1), # Third layer from which Style Loss is calculated
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True), # This is Relu 4.1 The output layer of the encoder.
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True)
            ).to(device)

        self.vgg19.load_state_dict(state_dict)

        encoder_children = list(self.vgg19.children())
        self.EncoderList = nn.ModuleList([nn.Sequential(*encoder_children[:4]), # Up to Relu 1.1
                                          nn.Sequential(*encoder_children[4:11]), # Up to Relu 2.1
                                          nn.Sequential(*encoder_children[11:18]), # Up to Relu 3.1
                                          nn.Sequential(*encoder_children[18:31]), # Up to Relu 4.1, also the
                                          ])                                       # input for the decoder

    def forward(self, x, intermediates=False): # if training use intermediates = True, to get the output of
        states = []                            # all the encoder layers to calculate the style loss
        for i in range(len(self.EncoderList)):
            x = self.EncoderList[i](x)

            if intermediates:       # All intermediate states get saved in states
                states.append(x)
        if intermediates:
            return states
        return x




def frame_sm_detector_demo_ms_coco(scene_processor_detector,frame,device,pred_dict,main_obj_bbox,img_id,mmdetection_model):
    """
    Main function for using the scene processing detector (SPD) in the digital use case (using MSCOCO images).
    This function receive a frame (image), uses several image processing technique over that frame, insert those frames
    into a given object diction model and aggregate the result for the main object in the frame.
    :param scene_processor_detector: req. An instance of the scene processing detector.
    :param frame: req. numpy format. an image.
    :param device: req. The device use (CPU/GPU).
    :param pred_dict: req. An empty dictionary were al predictions will be stored.
    :param main_obj_bbox: req. list representing the bounding box of the main object in the frame.
    :param output_path: req. str. A path for the output folder.
    :param img_id: req. int. The id of each image (in MS COCO dataset). used for the output plot only.
    :param mmdetection_model: req. bool. If the object detector used is taken from mmdetection package.
    :return: The SPD's prediction of the main object in the frame.
    """
    scene_processor_detector.set_dataset(frame)
    scenes = scene_processor_detector.scene_transformer(scene_processor_detector.processed_image_folders[2])
    for idx,scene in enumerate(scenes.keys()):
        output_path = os.path.join(scene_processor_detector.output_path, f'{scene}_scale', str(idx))
        Path(output_path).mkdir(parents=True, exist_ok=True)
        frame = scenes[scene]['raw_image']
        if len(frame.shape) < 3:
            frame = np.expand_dims(frame, axis=2)
        scenes[scene]['prediction'] = scene_processor_detector.predict(frame,mmdetection_model)
        if len(scenes[scene]['prediction'][0]['labels'])==0:
            continue
        scene_processor_detector.plot_prediction(np.copy(frame), scenes[scene]['prediction'],
                                                 scene_processor_detector.processed_image_folders[idx],img_id,
                                                 mmdetection_model)
        main_obj_label,main_obj_conf = scene_processor_detector.get_main_object_label_and_confidence(
            main_obj_bbox,scenes[scene]['prediction'],mmdetection_model)
        scenes[scene]['prediction'] = main_obj_label
        scenes[scene]['score'] = main_obj_conf
        if main_obj_label not in pred_dict.keys():
            pred_dict[main_obj_label] = 0
        pred_dict[main_obj_label]+= main_obj_conf
        if main_obj_label not in scene_processor_detector.prediction_for_image_processing_tech[scene]['prediction_dict'].keys():
            scene_processor_detector.prediction_for_image_processing_tech[scene]['prediction_dict'][main_obj_label] = 0
        scene_processor_detector.prediction_for_image_processing_tech[scene]['prediction_dict'][main_obj_label] += \
            main_obj_conf
    return pred_dict


def frame_sm_detector_demo_super_store(scene_processor_detector,frame,device,pred_dict,video_id,img_id, optimization_mode = True):
    """
    Main function for using the scene processing detector (SPD) in the physical use case (using SuperStore dataset).
    This function receive a frame (image), uses several image processing technique over that frame, insert those frames
    into a given object diction model and aggregate the result for the main object in the frame.
    :param scene_processor_detector:
    :param frame: req. An instance of the scene processing detector.
    :param device: req. The device use (CPU/GPU).
    :param pred_dict: req. An empty dictionary were al predictions will be stored.
    :param optimization_mode: optional. If using several parameters for each image processing technique for detector
    optimization.
    :return: The SPD's prediction of the main object in the frame.
    """

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    scene_processor_detector.set_dataset(frame)
    scenes = scene_processor_detector.scene_transformer("")
    for idx,scene in enumerate(scenes.keys()):
        output_path = os.path.join(scene_processor_detector.output_path,f'{scene}_scale',video_id)
        Path(output_path).mkdir(parents=True, exist_ok=True)
        frame = scenes[scene]['raw_image']
        if len(frame.shape)<3:
            frame = np.expand_dims(frame, axis=2)
        if not scene_processor_detector.target_model.mmdetection_model:
            scenes[scene]['prediction'] = scene_processor_detector.target_model.predict_faster_rcnn(frame, device)
        else:
            scenes[scene]['prediction'] = scene_processor_detector.target_model.predict_other_models(frame, device)
        if scene=="style_transfer":
            frame = imutils.resize(frame,640)
        scene_processor_detector.plot_prediction(np.copy(frame), scenes[scene]['prediction'], output_path, img_id,
                                                 scene_processor_detector.target_model.mmdetection_model)
        if len(scenes[scene]['prediction'][0]['labels']):
            label = int(scenes[scene]['prediction'][0]['labels'][0])
            score = float(scenes[scene]['prediction'][0]['scores'][0])
            pred_dict[label] += score
            scene_processor_detector.prediction_for_image_processing_tech[scene]['prediction_dict'][label] +=\
                score


    return pred_dict





