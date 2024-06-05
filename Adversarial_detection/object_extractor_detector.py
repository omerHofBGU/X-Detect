import torch
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import datetime
from pathlib import Path
from collections import Counter
import cv2
from pixellib.torchbackend.instance import instanceSegmentation
import os
import seaborn as sns
import math
palette = sns.color_palette("bright", 3)
import time

class Pointrend_instance_segmentation():
    """
    This class loads and uses 'Pointrend' instance segmentation framework.
    implemented in pytorch.
    This class uses PixelLib library, the license presented below.

    MIT License

    Copyright (c) 2020 ayoolaolafenwa

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
    def __init__(self,model_path):
        super().__init__()
        self.model = self.load_pretrained_mask_cnn_model(model_path)

    def load_pretrained_mask_cnn_model(self,model_path):
        """
        load Pointrend pre-trained model on resent50.
        :return:
        """
        ins = instanceSegmentation()
        # ins = instanceSegmentation(infer_speed = "rapid")
        ins.load_model(model_path+"/pointrend_resnet50.pkl",detection_speed = "rapid")
        return ins

    def image_instance_segmentation_from_path(self, img_path, output_path, model = None):
        """
        Function for instance segmentation for a single image.
        :param img_path: path for a given image.
        :param output_path: path to save the image after using Pointrend
        instance segmentation.
        :param model: pre-trained Pointrend model.
        :return: saves the images with its instance segmentations and
        classification.
        """
        if model == None:
            model = self.model
        start = time.time()
        result = model.segmentImage(image_path=img_path,show_bboxes=False,
                                    output_image_name=output_path,
                                    save_extracted_objects=True,
                                    extract_segmented_objects = True)
        end = time.time()
        print("[INFO] applying Pointrend for image took {:.2f} seconds".format(
            end - start))
        return result

    def image_instance_segmentation(self, img, model = None):
        """
        Instance segmentation function.
        :param img: req. image in numpy format.
        :param model: the Instance segmentation model.
        :return:
        """
        if model == None:
            model = self.model
        result = self.segmentImage(img)
        return result


    def batch_instance_segmentation(self,folder_path,model=None):
        """
        Function that performs image segmentation to a batch of images.
        :param folder_path: req. str. path to a folder containing the images.
        :param model: req. The instance segmentation model.
        :return: Segmentations of all the object of all the images in the folder.
        """
        if model == None:
            model = self.model
        result = model.segmentBatch(input_folder=folder_path,
                                    show_bboxes=False,
                                    save_extracted_objects=False,
                                    extract_segmented_objects=True)

        return result

    def video_instance_segmentation(self, video_path, output_path, model = None,
                                    save_output=True):
        """
       Function for instance segmentation in a video.
       :param video_path: path for a given video.
       :param output_path: path to save the video after using Pointrend
       instance segmentation model.
       :param model: pre-trained Pointrend model.
       :param save_output: bool. Option to save the processed video.
        :return: saves the video with its instance segmentation and
        per frame in the given output path.
       """
        cap = cv2.VideoCapture(video_path)
        if save_output:
            out = self.output_configuration(output_path,cap)
        if model == None:
            model = self.model
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                results = model.segmentFrame(frame,show_bboxes=True)
                if save_output:
                    out.write(results[1])
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def output_configuration(self,output_path,cap):
        """
        Help function for video_instance_segmentation for output video
        configuration.
        :param output_path: path to save the video after using yolo object
        detection.
        :param cap: video features (height, width).
        :return: the video configuration (open-cv videoWriter object).
        """
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(output_path,
                              cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                              (frame_width, frame_height))
        return out

    def real_time_segmentation(self):
        """
        Real time instance segmentation.
        :return: shows a real time video (using the computer web cam with
        instance segmentation.
        """
        camera = cv2.VideoCapture(0)
        model = self.model
        while camera.isOpened():
            res, frame = camera.read()
            ### Apply Segmentation
            result = model.segmentFrame(frame, show_bboxes=True)
            image = result[1]
            cv2.imshow('Image Segmentation', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        camera.release()
        cv2.destroyAllWindows()

    def plot_extracted_objects(self,extracted_objects):
        """
        Function that plot the extracted objects.
        :param extracted_objects: req. list of extracted object (numpy format).
        :return: plot to the screen a the extracted objects.
        """
        fig = plt.figure(figsize=(224, 224))
        for seg in extracted_objects:
            fig = plt.figure(figsize=(seg.shape[0]/100, seg.shape[1]/100))
            plt.imshow(seg, interpolation='nearest')
            plt.show()

    def segmentImage(self, img):
        """
        Main function that perform image segmentation on an image.
        :param img: req. image in numpy format.
        :return: An segmentation of the objects in the image.
        """
        outputs = self.model.predictor.segment(img)
        masks = outputs["instances"].pred_masks
        scores = outputs["instances"].scores
        class_ids = outputs["instances"].pred_classes
        boxes = outputs["instances"].pred_boxes.tensor
        boxes = torch.as_tensor(boxes, dtype=torch.int64)
        boxes = boxes.cpu().numpy()

        if torch.cuda.is_available() == False:
            class_ids = class_ids.numpy()
            masks = masks.numpy()
            scores = scores.numpy()

        else:
            class_ids = class_ids.cpu().numpy()
            masks = masks.cpu().numpy()
            scores = scores.cpu().numpy()

        names = []
        for _, a in enumerate(class_ids):
            name = self.model.class_names[a]
            names.append(name)

        scores = scores * 100
        scores = torch.as_tensor(scores, dtype=torch.int64)
        object_counts = Counter(names)

        r = {"boxes": boxes, "class_ids": class_ids, "class_names": names,
             "object_counts": object_counts,
             "scores": scores, "masks": masks, "extracted_objects": []}
        if len(r["masks"]) != 0:
            r["masks"] = np.stack(masks, axis=2)
        extracted_objects = "No Objects Extracted"
        mask = r['masks']
        m = 0
        ex = []
        if len(mask != 0):
            for a in range(mask.shape[2]):
                if names[a]!='person':
                    for b in range(img.shape[2]):
                        img[:, :, b] = img[:, :, b] * mask[:, :, a]
                    m += 1
                    extracted_objects = img[
                        np.ix_(mask[:, :, a].any(1), mask[:, :, a].any(0))]
                    ex.append(extracted_objects)
            extracted_objects = ex
        if extracted_objects!="No Objects Extracted" and len(
                extracted_objects)>0:
            extracted_objects = extracted_objects[0]
        return extracted_objects

class main_object():
    """
    This class represents the main object being examined.
    This class use SIFT algorithm to extract scene features.
    Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints.
    https://link.springer.com/article/10.1023/B:VISI.0000029664.99615.94
    """
    def __init__(self,raw_image,label,path,model_path):
        """
        Init the objects. required three params: the raw images,
        its corresponding label and the image path.
        :param raw_image: required, open-cv format image (flexible size)
        :param label: required. string. The main object ground truth label.
        :param path: required. string. the object local path. e.g. "apple.jpeg".
        """
        super().__init__()
        self.raw_image = raw_image
        self.label = label
        self.path = path
        self.model_path = model_path


    def object_processor(self, main_object=False):
        """
        Main function to process the class' object.
        The function uses help functions to extract the object's key points,
        descriptors amore features in order to compared it to other object in the future.
        :param main_object: optional,boolean, if the object is the tested
        object (the one compared to all prototypes).
        :return: a processed object.
        """
        self.transformed_image =  self.resize_image(self.raw_image)
        raw_image_gray = cv2.cvtColor(self.transformed_image,cv2.COLOR_RGB2GRAY)
        self.keypoints, self.landmarks = self.key_point_extractor(
            raw_image_gray, main_object)
        if len(self.keypoints)==0:
            # print(self.label,self.path)
            # print("crash")
            return "No Keypoints"
        self.pyramid = self.size_augmentations(raw_image_gray)
        self.patch_mat = self.generate_mask_mat(self.keypoints, self.pyramid, main_object)
        self.descriptor = self.get_descriptor(main_object)
        self.features_vec, self.descriptors_mat = self.get_feature_vec()

    def sift_prototype_processor(self):
        """
        Help function for "draw_matches" that process object using SIFT
        algorithm.
        Only used for plot the matches between to objects.
        :return: a processed object.
        """
        sift = cv2.SIFT_create()
        self.transformed_image = self.resize_image(self.raw_image)
        raw_image_gray = cv2.cvtColor(self.transformed_image,
                                      cv2.COLOR_RGB2GRAY)
        self.keypoints, self.descriptor = sift.detectAndCompute(
            raw_image_gray, None)

    def resize_image(self, input_image, target_size=240):
        """
        Function for resize the raw image to a fix size.
        :param input_image: required, the raw image in open-cv format.
        :param target_size: optional, the size of the resize output
        image.Default size is 240X240.
        :return: resized open-cv format image.
        """
        output_image = cv2.resize(input_image, (target_size, target_size),
                                  interpolation=cv2.INTER_LINEAR)

        return output_image


    def key_point_extractor(self, image_grey, main_object=False):
        """
        This method extract key points using open -cv ORB algorithm
        :param image_grey: required, open-cv format grey scale image
        containing the object.
        :param main_object: optional,boolean, if the object is the tested
        object (the one compared to all prototypes).
        :return: object's keypoints.
        """
        if main_object:
            max_features = 1000
        else:
            max_features = 200
        orb = cv2.ORB_create(nfeatures=max_features, scaleFactor=1.2, nlevels=4,
                             edgeThreshold=31,
                             firstLevel=0, WTA_K=2,
                             scoreType=cv2.ORB_FAST_SCORE)
        keypoints = orb.detect(image_grey, None)
        if len(keypoints) >= max_features:
            keypoints = keypoints[0:max_features]
        landmarks = []
        for i in keypoints:
            landmarks.append(((i.pt[0] / image_grey.shape[1]),
                              (i.pt[1] / image_grey.shape[0])))

        return keypoints, landmarks

    def size_augmentations(self, image_grey):
        """
        This method returns an array of different sizes of a given image.
        :param image_grey:required, open-cv format grey scale image
        containing the object.
        :return: An array of the original image in different sizes.
        """
        image_pyramid = []
        src_image = image_grey
        for i in range(4):
            image_pyramid.append(src_image)
            tmp_image = cv2.resize(src_image, (
            round(src_image.shape[0] * 1.0 / 1.2),
            round(src_image.shape[0] * 1.0 / 1.2)))
            src_image = tmp_image
        return image_pyramid

    def extract_mask(self, key_point, image_pyramid, scale_factor=1.2,
                     mask_size=32):
        """
        This function extracts (crop) and augmented image in a given key
        point.
        :param key_point: required, the key point location to extract.
                        opencv orb format.
        :param image_pyramid:required, list of the image in augmented sizes.
        :param scale_factor: optional, float, margin of the orignal mask,
        default is 1.2.
        :param mask_size:optional, int, the area size to extract, default
        is 32.
        :return: cropped image in the key point location.
        """
        img = image_pyramid[key_point.octave]
        scale_factor = 1 / math.pow(scale_factor, key_point.octave)
        center = (key_point.pt[0] * scale_factor, key_point.pt[1] * scale_factor)
        rot = cv2.getRotationMatrix2D(center, key_point.angle, 1.0)
        rot[0][2] = rot[0][2] + mask_size / 2 - center[0]
        rot[1][2] = rot[1][2] + mask_size / 2 - center[1]

        cropped_img = cv2.warpAffine(img, rot, (mask_size, mask_size),
                                     cv2.INTER_LINEAR)
        return cropped_img

    def generate_mask_mat(self, keypoints, pyramid, main_object = False):
        """
        Function that extracts masks in the key point location for augmented
        images and processed those masks.
        :param keypoints:required, a tuple of key points location to extract.
                        opencv orb format.
        :param pyramid: required, list of the image in augmented sizes.
        :param main_object: optional,boolean, if the object is the tested
        object (the one compared to all prototypes).
        :return: The processed mask.
        """
        patch_mat = []
        for i in range(len(keypoints)):
            patch_mat.append(self.extract_mask(keypoints[i], pyramid))

        patch_mat_numpy = np.array(patch_mat)
        patch_mat_numpy = patch_mat_numpy / 128.0 - 1.0
        patch_mat_numpy = patch_mat_numpy.astype('float32')
        patch_mat_numpy = np.expand_dims(patch_mat_numpy, axis=3)

        if main_object:
            max_feaures = 1000
        else:
            max_feaures=200
        if patch_mat_numpy.shape[0] < max_feaures:
            pad_length = max_feaures - patch_mat_numpy.shape[0]
            patch_mat_numpy = np.pad(patch_mat_numpy,
                                      ((0, pad_length), (0, 0), (0, 0), (0, 0)),
                                      'constant', constant_values=(0))
        return patch_mat_numpy

    def get_descriptor(self, main_object=False):
        """
        Generate key points descriptor.
        :param main_object: optional,boolean, if the object is the tested
        object (the one compared to all prototypes).
        :return: Key points descriptor (numpy array).
        """
        if not main_object:
            interpreter = tf.lite.Interpreter(
                model_path=self.model_path+"/descriptor.tflite")
            interpreter.allocate_tensors()
        else:
            interpreter = tf.lite.Interpreter(
                model_path=self.model_path+"/descriptor_1k.tflite")
            interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'],
                               self.patch_mat)
        interpreter.invoke()
        descriptors = interpreter.get_tensor(output_details[0]['index'])
        return descriptors

    def get_feature_vec(self):
        """
        Generate main descriptor using the object key_points and masks.
        :return: main descriptor
        """
        inv_scale = 1.0 / 320
        descriptor_dims = 40

        features_vec= []
        descriptors_mat = np.zeros((len(self.keypoints), descriptor_dims))
        for j in range(len(self.keypoints)):
            features_vec.append((self.keypoints[j].pt[0] * inv_scale,
                                  self.keypoints[j].pt[1] * inv_scale))
            for i in range(descriptor_dims):
                descriptors_mat.itemset((j, i), self.descriptor[j][i])
        return features_vec,descriptors_mat

class Prototypes():
    """
    This class represent a collections of prototypes.
    Those prototypes is used to classified a given object class.
    """
    def __init__(self,prototypes_path,model_path):
        """
        Init function for the prototypes.
        :param prototypes_path: required. string. the prototypes path.
        """
        super().__init__()
        self.prototypes_path = prototypes_path
        self.model_path = model_path
        self.prototypes_instances = self.load_prototypes(prototypes_path)

    def load_prototypes(self,prototypes_path):
        """
        Load prototypes from path function.
        :param prototypes_path: required. string. the prototypes path.
        :return: numpy array of prototypes.
        """
        prototypes = []
        for object_folder in os.listdir(prototypes_path):
            prototype_class = object_folder
            for prototype in os.listdir(os.path.join(prototypes_path,
                                                     object_folder)):
                image = cv2.imread(os.path.join(prototypes_path,
                                                object_folder, prototype))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                prototypes.append(main_object(image, object_folder, prototype,self.model_path))

        prototypes = np.array(prototypes)
        return prototypes

    def process_prototypes(self,):
        """
        Function for processing each of the object. see
        main_object.object_processor for the processing procedure.
        :return: a processed prototypes.
        """
        for i,prototype in enumerate(self.prototypes_instances):
            prototype.object_processor(main_object=False)

    def find_best_match(self, compared_object,prototypes_path,
                        adversarial_path, k=3,save_plot = False,
                        save_path= None):
        """
        Function for find the best match (closest prototype instances to a
        given tested object.
        :param compared_object: required, the tested main object which we aim to
        classify, a main object class.
        :param prototypes_path: required, string, the path for the
        prototypes folder.
        :param adversarial_path: required, string, the path for the
        main object folder.
        :param k: required, integer, number of neighbors to tested for.
        :param save_plot: optional, boolean , whether to save the comparison
        plot.
        :param save_path: optional, string, the output path for the save plot.
        :return: The best match prototypes (list in the length of k) and
        their scores, and (optional) plot of the best match prototype instance.
        """
        similarity_list = []
        for i in range(len(self.prototypes_instances)):
            similarity = self.compute_similarity(compared_object,
                                                 self.prototypes_instances[i])
            similarity_list.append(len(similarity))
        best_match = sorted(range(len(similarity_list)), key=lambda i:
        similarity_list[
            i])[-k:]
        best_scores = [similarity_list[index] for index in best_match]
        # print(sum(best_scores)/len(best_scores))
        if save_plot:
            self.draw_matches(compared_object, self.prototypes_instances[
                best_match[-1]],save_path,prototypes_path,adversarial_path)
        return self.prototypes_instances[best_match],best_scores

    def compute_similarity(self, main_object, prototype):
        """
        Function for compute the similarity between to the tested object and
        a prototype.
        The similarity score is defined by the number of matches between the
        object
        and the prototype.
        :param main_object: required, the tested main object which we aim to
        classify, a main object class.
        :param prototype:  required, a prototype, a main object
        class.
        :return: similarity score (number of matches) between the object and
        the prototype.
        """
        max_match_distance = 0.6
        knn = 1
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(main_object.descriptor,
                              prototype.descriptor, k=knn)
        correspondence_result = []
        correspondence = []
        match_arr = []
        match_idx = 0
        correspondence.append({})
        correspondence[match_idx]['points_frame'] = []
        correspondence[match_idx]['points_index'] = []
        for match_pair in matches:

            if len(match_pair) < knn:
                continue
            best_match = match_pair[0]
            if best_match.distance > max_match_distance or \
                    best_match.distance<0.02:
                continue
            match_arr.append(best_match)
            correspondence_result.append(best_match)

        # print(f"number of matches: {len(correspondence_result)}")
        correspondence_result = sorted(correspondence_result,
                                       key=lambda x: x.distance)
        return correspondence_result

    def compute_similarity_surf(self, main_object, prototype):

        """
        Help function for "draw_matches" that process object using SURF
        algorithm.
        Only used for plot the matches between to objects.
        :param main_object: required, the tested main object which we aim to
        classify, a main object class.
        :param prototype: required, a prototype, a main object
        class.
        :return: similarity score (number of matches) between the object and
        the prototype using SURF.
        """
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = bf.match(main_object.descriptor, prototype.descriptor)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches


    def majority_vote(self, best_matches,scores):
        """
        Function that decided the objet predict class from the best matches
        list.
        :param best_matches: A list of prototypes that best matched the tested
        object.
        :param scores: The corresponding similarity score for each prototype.
        :return: A probabilities dictionary, keys are the labels and values
        are their probabilities.
        """
        labels = []
        for prototype in best_matches:
            labels.append(prototype.label)
        unique_list = list(set(labels))
        dict_of_labels = {i: 0 for i in unique_list}
        for i,score in enumerate(scores):
            dict_of_labels[labels[i]]+=score
        ascending_order_dict = dict(sorted(dict_of_labels.items(), key=lambda
            item: item[1],reverse = True))
        probabilities_dict = {}
        if sum(scores)==0:
            return probabilities_dict
        for key in ascending_order_dict:
            probabilities_dict[key] = dict_of_labels[key] / sum(scores)
        return probabilities_dict

    def draw_matches(self, prototype1, prototype2, save_path,
                     prototypes_path, adversarial_path):
        """

        :param prototype1: req. first prototype, a Prototype instance
        :param prototype2: req. second prototype, a Prototype instance
        :param save_path: req. A string represent the output path to save
        the figure.
        :param prototypes_path: req. A string. the path for the second
        prototype original image.
        :param adversarial_path: req. A string. the path for the first
        prototype original image.
        :return: A plot matching (and connecting) between the prototype key
        points.
        """
        prototype1.raw_image = cv2.cvtColor(prototype1.raw_image,
                                            cv2.COLOR_BGR2RGB)
        prototype2.raw_image = cv2.cvtColor(prototype2.raw_image,
                                            cv2.COLOR_BGR2RGB)
        prototype1 = main_object(prototype1.raw_image, "xx", "xx",self.model_path)
        prototype2 = main_object(prototype2.raw_image, "xx", "xx",self.model_path)
        prototype1.sift_prototype_processor()
        prototype2.sift_prototype_processor()
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = bf.match(prototype1.descriptor, prototype2.descriptor)
        matches = sorted(matches, key=lambda x: x.distance)

        matched_image = cv2.drawMatches(img1=prototype1.transformed_image,
                               keypoints1=prototype1.keypoints,
                               img2=prototype2.transformed_image,
                               keypoints2=prototype2.keypoints,
                               matches1to2=matches,
                               flags=2,
                               outImg = None)

        plt.imshow(matched_image)
        cv2.imwrite(save_path, matched_image)

    def process_main_prototype(self, index):
        """
        Function for processing the test object. Using object_processor
        function. This function is only used for testing the prototypes.
        :param index: required. the index of the main object in the
        prototype list.
        :return: List of prototypes with the main object processed.
        """
        self.prototypes_instances[index].object_processor(
            main_object=True)

    def reset_main_prototype(self, index):
        """
        Function for reset the processing of the tested object. Using
        object_processor function.
        This function is only used for testing the prototypes.
        :param index: required. the index of the main object in the
        prototype list.
        :return: List of processed prototypes.
        """
        self.prototypes_instances[index].object_processor(
            main_object=False)

    def find_best_match_prototypes_only(self, index=0, k=3):
        """
        Function for find the best match (class) for a given main prototype.
        This function is only used for testing the prototypes.
        :param index: required, integer, the location of the tested prototype
        in the prototypes list.
        :param k: required, integer, number of neighbors to tested for.
        :return: The best match prototypes (list in the length of k) and
        their scores.
        """
        similarity_list = []
        for i in range(len(self.prototypes_instances)):
            if i == index:
                continue
            similarity = self.compute_similarity(
                self.prototypes_instances[index],
                self.prototypes_instances[i])
            similarity_list.append(len(similarity))
        best_match = sorted(range(len(similarity_list)), key=lambda i:
        similarity_list[
            i])[-k:]
        best_scores = [similarity_list[index] for index in best_match]
        return self.prototypes_instances[best_match], best_scores

def cut_bbox(frame,bbox,out_path):
    bbox_cpu = bbox.cpu().numpy().astype(int)
    bbox = bbox_cpu.clip(0)
    cropped_image = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    cv2.imwrite(out_path,cropped_image)
    return cropped_image

def cut_bbox_type_list(frame,bbox,out_path):
    cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
    cv2.imwrite(out_path,cropped_image)
    return cropped_image


def create_output_folder():
    """
    Function that creates outputs folder.
    :return: --
    """
    time = datetime.datetime.now().strftime("%d-%m-%Y_%H;%M")
    output_path = f"Output/{time}"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    output_path = f"Output/{time}/detector_results"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    return output_path

def oe_detector_demo_ms_coco(prototypes, pointrend, frame, object_label, bbox,model_path,
                             save_plot = True,output_folder='', video_id = '1', frame_id = '1'):
    """
    Main function for using the object extraction detector on the digital use case with the MS COCO dataset.
    :param prototypes: req. Prototype instances. representing the most related images of each class.
    :param pointrend: req. Pointrent model instance.
    :param frame: req. the examined image in numpy format.
    :param object_label: req. str. the label of the object (use for evaluation).
    :param bbox: req. list. the bbox of the main object in the frame.
    :param model_path: req. str. path to the target model.
    :param save_plot: opt. bool. Whether to save the explainable plot.
    :param output_folder: opt. str. the output folder where the explainable output will be saved in.
    :param video_id: opt. str. id of the video for the output.
    :param frame_id: opt. str. id of the frame for the output.
    :return: The object extraction detector classification for the main item in the frame.
    """
    if output_folder == None:
        output_folder = create_output_folder()
    output_path = os.path.join(output_folder,f'{object_label}_video_'
                                           f'{video_id}_frame_{frame_id}.png')
    extracted_object = cut_bbox_type_list(frame, np.array(bbox), output_path)
    extracted_object = pointrend.image_instance_segmentation(extracted_object)
    probability_dict = None
    if extracted_object != 'No Objects Extracted' and len(extracted_object) > 0:
        main_object_in_scene = main_object(extracted_object, object_label, output_path,model_path)
        result = main_object_in_scene.object_processor(main_object=True)
        if result != 'No Keypoints':
            best_match_prototypes, best_match_scores = prototypes.find_best_match(
                main_object_in_scene, prototypes.prototypes_path,
                main_object_in_scene, k=7,
                save_plot=save_plot,
                save_path=output_path)
            probability_dict = prototypes.majority_vote(best_match_prototypes, best_match_scores)
    else:
        print('No Objects Extracted')
    return probability_dict


def oe_detector_demo_super_store(prototypes, pointrend, frame, bbox, model_path
                                 ,object_label = "not_provided", save_plot = True,output_folder='',
                                 video_id = '1', frame_id = '1'):
    """
    Main function for using the object extraction detector on the physical use case with the MS COCO dataset.
    :param prototypes: req. Prototype instances. representing the most related images of each class.
    :param pointrend: req. Pointrent model instance.
    :param frame: req. the examined image in numpy format.
    :param bbox: req. list. the bbox of the main object in the frame.
    :param model_path: req. str. path to the target model.
    :param object_label: req. str. the label of the object (use for evaluation).
    :param save_plot: opt. bool. Whether to save the explainable plot.
    :param output_folder: opt. str. the output folder where the explainable output will be saved in.
    :param video_id: opt. str. id of the video for the output.
    :param frame_id: opt. str. id of the frame for the output.
    :return: The object extraction detector classification for the main item in the frame.
    """
    if output_folder == None:
        output_folder = create_output_folder()
    else:
        output_folder = os.path.join(output_folder,video_id)
        Path(output_folder).mkdir(parents=True,exist_ok=True)
    output_path = os.path.join(output_folder,f'{object_label}_video_'
                                           f'{video_id}_frame_{frame_id}.png')
    extracted_object = pointrend.image_instance_segmentation(frame)
    if extracted_object=='No Objects Extracted':
        extracted_object = cut_bbox(frame,bbox,output_path)
        extracted_object = pointrend.image_instance_segmentation(extracted_object)
    probability_dict = None
    if extracted_object!='No Objects Extracted' and len(extracted_object)>0:
        main_object_in_scene = main_object(extracted_object,object_label,output_path,model_path)
    # cv2.imwrite(f'{output_path}/{object_label}_{1}.jpg',image)
        result = main_object_in_scene.object_processor(main_object=True)
        if result!='No Keypoints':
            best_match_prototypes, best_match_scores = prototypes.find_best_match(
                main_object_in_scene, prototypes.prototypes_path,
                main_object_in_scene, k=7,
                save_plot=save_plot,
                save_path=output_path)
            probability_dict = prototypes.majority_vote(best_match_prototypes,
                                                        best_match_scores)
    return probability_dict
