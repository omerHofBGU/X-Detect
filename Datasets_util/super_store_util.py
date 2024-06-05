import os
import shutil
import sys
import csv
import glob
import cv2
import pandas as pd
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import math
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import PIL.ImageFont as ImageFont
import random

"""
A module for handling Superstore dataset.
"""

def create_dataset(argv):
    """
    Main function that get the Superstore dataset path and adjust it to be usable to object detection Models.
    Args:
        argv: [0] - format to store the dataset (yolo or faster rcnn).
              [1] - path of SuperStore dataset, a folder contains folders of all classes and train, test csv files.

    Returns: Superstore dataset in YOLO or Faster R-CNn format.
    """

    products = ["Agnesi Polenta", "Snyders", "Calvin Klein", "Dr Pepper",
                "Flour", "Groats", "Jack Daniels", "Almond Milk", "Nespresso",
                "Oil", "Paco Rabanne", "Pixel4", "Samsung_s20", "Greek Olives",
                "Curry Spice", "Chablis Wine", "Lindor", "Piling Sabon", "Tea",
                "Versace"]

    root_path = argv[1]
    csv_train_path = os.path.join(root_path, 'train_set.csv')
    csv_test_path = os.path.join(root_path, 'test_set.csv')
    classes_file_path = os.path.join(root_path, 'classes.txt')

    if 'yolo' in argv[0]:
        yc = yolo_class(products, root_path, csv_train_path, csv_test_path, classes_file_path)
        yc.create_yolo_format()

    if 'faster_rcnn' in argv[0]:
        fc = faster_rcnn_class(products, root_path, csv_train_path, csv_test_path)
        fc.create_faster_rcnn_format()
    else:
        mc = coco_format(root_path, csv_train_path, csv_test_path)
        mc.dataset_as_coco_format_wrapper()


class yolo_class:
    """
        Class for generate Super Store dataset in yolo format
    """

    def __init__(self, products, root_path, csv_train_path, csv_test_path, classes_file_path):
        self.products = products
        self.root_path = root_path
        self.csv_train_path = csv_train_path
        self.csv_test_path = csv_test_path
        self.classes_file_path = classes_file_path

    def create_yolo_format(self):
        """
            Wrapper function that first create all the folders in each product according to yolo format (train, test, val)
            and copies the images to them. Then,add the annotations for each image. And finally, concatenate all
            train,test and val folders in each product to one train test and val folder.
            :return: Super Store dataset in yolo format
        """

        for product in self.products:
            # <editor-fold desc="Paths declaration">
            train_images_path = os.path.join(self.root_path, product, 'train', 'images')
            train_labels_path = os.path.join(self.root_path, product, 'train', 'labels')
            val_images_path = os.path.join(self.root_path, product, 'val', 'images')
            val_labels_path = os.path.join(self.root_path, product, 'val', 'labels')
            test_images_path = os.path.join(self.root_path, product, 'test', 'images')
            test_labels_path = os.path.join(self.root_path, product, 'test', 'labels')
            # </editor-fold>

            self.open_directories_and_files(product, train_images_path, train_labels_path, val_images_path,
                                            val_labels_path,
                                            test_images_path, test_labels_path)

            self.read_from_csv(product, train_images_path, train_labels_path, val_images_path, val_labels_path,
                               test_images_path, test_labels_path)

            self.concatenate_files_to_yolo_format(product)
        print(f"Super Store dataset in YOLO format saved in {self.destination_root}")
        print("Now you can train a YOLO model using the data in this directory by executing the "
              "train_custom_object_detector.py module")

    def open_directories_and_files(self, product, train_images_path, train_labels_path, val_images_path,
                                   val_labels_path, test_images_path, test_labels_path):
        """
            A function that creates in each product folder the train test and val folders, when each one of them contains
            the images and labels folders.
            Copy the images to each image folder in the train and test folders.
        """
        if os.path.exists(self.root_path + '/' + product + '/' + 'input'):
            self.create_empty_folder(train_images_path)
            self.create_empty_folder(train_labels_path)
            self.create_empty_folder(val_images_path)
            self.create_empty_folder(val_labels_path)
        self.create_empty_folder(test_images_path)
        self.create_empty_folder(test_labels_path)
        try:
            input_folder = os.path.join(self.root_path, product, 'input')
            test_folder = os.path.join(self.root_path, product, 'test')

            if os.path.exists(input_folder):
                # move images from the given input folder to the images folder in train and val folders.
                self.arrange_files(input_folder, train_images_path)
                os.rmdir(input_folder)
            if os.path.exists(test_folder):
                # move images from the given test folder to the images folder in a test folder
                self.arrange_files(test_folder, test_images_path)
        except:
            pass

    def read_from_csv(self, product, train_images_path, train_labels_path, val_images_path, val_labels_path,
                      test_images_path, test_labels_path):
        """
             Read annotations from CSV files (train and test) and write them into the labels folders in train, test and val
             of each product.
             Now, each product folder contains train, test and val folders when inside each one of them,
             there are images and labels folders that contain the images and annotations respectively.
        """
        if os.path.exists(train_images_path):
            self.write_to_labels_files(product, self.csv_train_path, train_images_path, train_labels_path)
            self.write_to_labels_files(product, self.csv_train_path, val_images_path, val_labels_path)

            # move images and labels from train folder to val folder
            self.move_to_val_folder(train_images_path, val_images_path)
            self.move_to_val_folder(train_labels_path, val_labels_path)

        self.write_to_labels_files(product, self.csv_test_path, test_images_path, test_labels_path)

    def concatenate_files_to_yolo_format(self, product):
        """
            Creates "yolo format" folder (in the same path of the original data folder) that contains the data according
            to yolo format, by concatenating all train, test and val folders in each product to one train, test and val folders.
        """

        # <editor-fold desc="Paths declaration">
        root_path_as_Path_instance = Path(self.root_path)
        self.destination_root = os.path.join(root_path_as_Path_instance.parent.absolute(),"yolo format")
        yolo_train_images = os.path.join(self.destination_root, "train", "images")
        yolo_train_labels = os.path.join(self.destination_root, "train", "labels")
        yolo_test_images = os.path.join(self.destination_root, "test", "images")
        yolo_test_labels = os.path.join(self.destination_root, "test", "labels")
        yolo_val_images = os.path.join(self.destination_root, "val", "images")
        yolo_val_labels = os.path.join(self.destination_root, "val", "labels")

        train_images_path = os.path.join(self.root_path, product, 'train', 'images')
        train_labels_path = os.path.join(self.root_path, product, 'train', 'labels')
        val_images_path = os.path.join(self.root_path, product, 'val', 'images')
        val_labels_path = os.path.join(self.root_path, product, 'val', 'labels')
        test_images_path = os.path.join(self.root_path, product, 'test', 'images')
        test_labels_path = os.path.join(self.root_path, product, 'test', 'labels')
        # </editor-fold>

        self.create_empty_folder(self.destination_root)
        self.create_empty_folder(os.path.join(self.destination_root, "train"))
        self.create_empty_folder(os.path.join(self.destination_root, "test"))
        self.create_empty_folder(os.path.join(self.destination_root, "val"))

        if os.path.exists(train_images_path):
            self.concat(train_images_path, yolo_train_images)
            self.concat(train_labels_path, yolo_train_labels)

            self.concat(val_images_path, yolo_val_images)
            self.concat(val_labels_path, yolo_val_labels)
        if os.path.exists(test_images_path):
            self.concat(test_images_path, yolo_test_images)
            self.concat(test_labels_path, yolo_test_labels)

    # <editor-fold desc="Help functions">

    def create_empty_folder(self, path):
        """
        Function that creates an empty folder.
        Args:
            path: req. srt. The folder's path.

        Returns: Create an empty folder.
        """
        if not os.path.exists(path):
            os.makedirs(path)

    def create_labels_files(self, images_path, labels_path):
        """
        Create label files in YOLOv5 format.
        Args:
            images_path: req. str. path for the image folder.
            labels_path: req. str. path for the labels folder.
        Returns: create labels files in YOLOv5 format in the relevant folders.

        """
        files_names = []
        for file in os.listdir(images_path):
            with open(os.path.join(images_path, file)) as f:
                files_names.append(file.split(".", 1)[0])

        for file_name in files_names:
            out_path = labels_path + '/' + file_name + '.txt'
            if not os.path.exists(out_path):
                open(out_path, 'w')

    def arrange_files(self, source_folder, destination_folder):
        """
        Move files to relevant folders.
        Args:
            source_folder: req. str. the path to the source folder.
            destination_folder: req. str. the path to the destination folder.
        Returns:--
        """
        for file_name in os.listdir(source_folder):
            source = os.path.join(source_folder, file_name)
            destination = os.path.join(destination_folder, file_name)
            if os.path.isfile(source):
                shutil.move(source, destination)

    def write_to_labels_files(self, product, csv_path, images_path, labels_path):
        """
        Function that write the labels into corresponding files.
        Args:
            product: req. str. class name.
            csv_path: req. str. path to the csv file.
            images_path: req. str. path for the image folder.
            labels_path: req. str. path for the labels folder.
        Returns: --

        """
        self.create_labels_files(images_path, labels_path)
        files_path = glob.glob(labels_path + '/*.txt')
        product_labels = self.load_labels_from_csv(product, csv_path)
        for label, file in zip(product_labels, files_path):
            output = open(file, 'w')
            output.write(label)
        shutil.copy(self.classes_file_path, labels_path)  # copy 'classes' file to each labels directory

    def load_labels_from_csv(self, product, csv_path):
        """
        Function to load labels from a given csv file.
        Args:
            product: req. str. class name.
            csv_path:  str. path to the csv file.
        Returns: --

        """
        product_labels = []
        with open(csv_path, "rt", encoding='ascii') as f:
            csvreader = csv.reader(f)
            next(csvreader)
            for row in csvreader:
                if product == row[3]:
                    yolo_label = row[2] + " " + row[4] + " " + row[5] + " " + row[6] + " " + row[7]
                    product_labels.append(yolo_label)
        return product_labels

    def move_to_val_folder(self, train_folder, val_folder):
        """
        Function that move files from train to validation folder.
        Args:
            train_folder:  req. str. the path to the train folder.
            val_folder: req. str. the path to the validation folder.
        Returns: --

        """
        for file_name in os.listdir(train_folder):
            source = os.path.join(train_folder, file_name)
            destination = os.path.join(val_folder, file_name)
            if source.endswith('_1.jpeg') or source.endswith('_1.txt'):
                shutil.move(source, destination)

    def concat(self, source_path, destination_path):
        """
        Function that move files from one folder to another.
        Args:
            source_path: req. str. the path to the source folder.
            destination_path: req. str. the path to the destination folder.

        Returns: --

        """
        self.create_empty_folder(destination_path)
        for file_name in os.listdir(source_path):
            source = os.path.join(source_path, file_name)
            destination = os.path.join(destination_path, file_name)
            shutil.copy(source, destination)
    # </editor-fold>


class faster_rcnn_class:
    """
    Class for generate Super Store dataset in faster rcnn format
    """

    def __init__(self, products, root_path, csv_train_path, csv_test_path):
        self.products = products
        self.root_path = root_path
        self.csv_train_path = csv_train_path
        self.csv_test_path = csv_test_path

    def create_faster_rcnn_format(self):
        """
        Wrapper function that first copy the image to a new faster rcnn
        folder and then adds the annotations for each image.
        :return: Super Store dataset in faster rcnn format
        """
        self.copy_images()
        self.add_annotations()
        print(f"Super Store dataset in Faster R-CNN format saved in {self.destination_root}")
        print("Now you can train a Faster R-CNN model using the data in this directory by executing the "
              "train_custom_object_detector.py module ")

    def copy_images(self):
        """
        Function for copy folder and images to a new folder
        :return: Super Store images in a new faster rcnn format folder
        """
        path = Path(self.root_path)
        self.destination_root = os.path.join(path.parent.absolute(),
                                             "faster_rcnn_format")
        destination = shutil.copytree(self.root_path, self.destination_root,
                                      ignore=shutil.ignore_patterns('*.csv',
                                                                    '*.txt'))

    def add_annotations(self):
        """
        Function for adding annotations for each image.
        :return: Super Store annotations in a faster rcnn format.
        """
        df_train = pd.read_csv(self.csv_train_path)
        df_test = pd.read_csv(self.csv_test_path)
        self.csv_to_faster_rcnn_files_wrapper(df_train, train=True)
        self.csv_to_faster_rcnn_files_wrapper(df_test, train=False)

    def create_empty_folder(self, folder_path):
        """
        FUnction for creating an empty folder
        :param folder_path: Required. string. The path for the new folder.
        :return: A new folder is created in the given path.
        """
        Path(folder_path).mkdir(parents=True, exist_ok=True)

    def csv_to_faster_rcnn_files_wrapper(self, df, train=True):
        """
        A wrapper functions to extract the annotations form a pandas
        dataframe and save it to a json files. Uses dump_files and
        extract_to_faster_rcnn_file functions.
        :param df: required. pandas dataframe. A dataframe containing Super
        Store train-set/ test-set.
        :param train: required. bool. If the given dataframe is the trainset
        or testset. Default is trainset.
        :return: Json files containing the annotations of each image are
        saved in the new faster rcnn folder.
        """
        if train:
            folder_type = "input label"
        else:
            folder_type = "test label"
        for class_instance in os.listdir(self.destination_root):
            folder_path = os.path.join(self.destination_root, class_instance,
                                       folder_type)
            self.create_empty_folder(folder_path)
            temp_df = df.loc[df['label'] == class_instance]
            self.dump_files(temp_df, folder_path)

    def dump_files(self, df, output_path):
        """
        Function for dumping a whole class into annotations files in
        faster rcnn formats (wrapper function for extract_to_faster_rcnn_file)
        :param df: required. pandas dataframe. A dataframe containing Super
        Store train-set/ test-set.
        :param output_path: required. string. An output path to save the
        annotations in.
        :return: Json files containing the annotations is extracted to the
        given output path.
        """
        for iter, row in df.iterrows():
            row = row.to_dict()
            self.extract_to_faster_rcnn_file(row, output_path)

    def extract_to_faster_rcnn_file(self, row, output_path):
        """
        This function dump a specific row to an annotation json file in a faster
        rcnn format.
        :param row: required. Pandas dataframe row. Describe a single record
        in the dataset.
        :param output_path: required. string. An output path to save the
        annotation in.
        :return: A json file containing the annotations is extracted to the
        given output path.
        """
        output_dict = {}
        output_dict['labels'] = [row['label']]
        bounding_box = [[row['f_rcnn_x1'],
                         row['f_rcnn_y1'],
                         row['f_rcnn_x2'],
                         row['f_rcnn_y2']]]
        output_dict['boxes'] = bounding_box
        output_dict['light'] = row['light']
        output_dict['expensive'] = row['expensive']
        output_dict['hand_location'] = row['hand_location']
        output_dict['background'] = row['background']
        output_dict['visual_angle'] = row['visual_angle']
        with open(f'{output_path}/{row["sample_id"]}.json', 'w') as \
                json_file:
            json.dump(output_dict, json_file)

class coco_format:
    """
    Class that transforms Super Store dataset to COCO format.
    """

    def __init__(self, root_path,csv_train_path,csv_test_path):

        self.root_path = root_path
        self.train_set = pd.read_csv(csv_train_path)
        self.test_set = pd.read_csv(csv_test_path)
        self.image_width = 640
        self.image_height = 360

        self.mapping = {
            'Agnesi Polenta': 1,
            'Almond Milk': 2,
            'Snyders': 3,
            'Calvin Klein': 4,
            'Dr Pepper': 5,
            'Flour': 6,
            'Groats': 7,
            'Jack Daniels': 8,
            'Nespresso': 9,
            'Oil': 10,
            'Paco Rabanne': 11,
            'Pixel4': 12,
            'Samsung_s20': 13,
            'Greek Olives': 14,
            'Curry Spice': 15,
            'Chablis Wine': 16,
            'Lindor': 17,
            'Piling Sabon': 18,
            'Tea': 19,
            'Versace': 20,
            'Adversarial Patch':21
        }

    def image(self,row):
        """
        Create image annotation.
        Args:
            row: pandas series instance that contained information regarding the image.
        Returns: dictionary with the image annotation.
        """
        image = {}
        image["height"] = self.image_height
        image["width"] = self.image_width
        image["id"] = row.name
        image["file_name"] = f'{row.sample_id}.jpeg'
        return image

    def category(self,item_label):
        """
        Create category annotation.
        Args:
            item_label: the label of the item.
        Returns: dictionary with the category annotation.

        """
        category = {}
        category["supercategory"] = 'none'
        category["id"] = self.mapping[item_label]
        category["name"] = item_label
        return category

    def annotation(self,row):
        """
        Process annotations to mscoco format.
        Args:
            row: pandas series instance that contained information regarding the image.

        Returns: dictionary with the image object's annotation.

        """
        annotation = {}
        area = self.image_width*self.image_height
        annotation["segmentation"] = []
        annotation["iscrowd"] = 0
        annotation["area"] = float(area)
        annotation["image_id"] = row.name

        if row.f_rcnn_x1<0:
            row.f_rcnn_x1=0
        if row.f_rcnn_y1<0:
            row.f_rcnn_y1=0

        annotation["bbox"] = [row.f_rcnn_x1, row.f_rcnn_y1, row.f_rcnn_x2-row.f_rcnn_x1, row.f_rcnn_y2-row.f_rcnn_y1]

        annotation["category_id"] = self.mapping[row.label]
        annotation["id"] = row.name
        return annotation

    def dataset_as_coco_format_wrapper(self):
        """
        Wrapper function that changes the dataset into COCO format. Uses dataset_as_coco_format function.
        Returns: train and test sets as COCO formats.

        """
        self.dataset_as_coco_format(self.train_set,type="train")
        self.dataset_as_coco_format(self.test_set,type="test")

    def dataset_as_coco_format(self,data,type):
        """
        Function that changes the dataset into COCO format
        Args:
            data: req. dataframe instance, representing the dataset.
            type: req, str. "train" or "test".

        Returns: The given dataset in COCO format.

        """
        images = []
        categories = []
        annotations = []

        for index, row in data.iterrows():
            annotations.append(self.annotation(row))
            images.append(self.image(row))
        for item in (self.mapping.keys()):
            categories.append((self.category(item)))

        data_coco = {}
        data_coco["images"] = images
        data_coco["categories"] = categories
        data_coco["annotations"] = annotations
        json.dump(data_coco, open(f'{self.root_path}/{type}_ms_coco_format.json', "w"), indent=4)




class Super_store_dataset_analysis:
    """
    Class that analyzes Super Store dataset
    """

    def __init__(self, root_path):
        csv_train_path = os.path.join(root_path, 'train_set.csv')
        csv_test_path = os.path.join(root_path, 'test_set.csv')

        self.root_path = root_path
        self.train_set = pd.read_csv(csv_train_path)
        self.test_set = pd.read_csv(csv_test_path)
        self.overall_dataset = self.train_set.append(self.test_set,
                                                     ignore_index=True)

        # Bounding box colors
        self.COLORS = ['Green',
                       'Red', 'Pink',
                       'Olive', 'Brown', 'Orange']

    def dataset_overview(self):
        """
        Main function for exploratory data analysis.
        :return: General information about Super Store dataset, plots and
        records examples.
        """
        self.general_stats()
        self.data_plots()
        # self.print_example_images()

    def general_stats(self):
        """
        Function that prints Super Store general information.
        :return: Super Store general information
        """

        print("\033[1m Super Store general stats \033[0m")
        print(f"Number of records: {len(self.overall_dataset)}")
        print(f"Number of classes: "
              f"{len(self.train_set['label'].unique())}")

        print(f"Train set size: {len(self.train_set)} "
              f"({math.ceil((len(self.train_set) / len(self.overall_dataset)) * 100)}%)")
        print(f"Test set size: {len(self.test_set)} "
              f"({math.floor((len(self.test_set) / len(self.overall_dataset)) * 100)}%)")

    def data_plots(self):
        """
        Wrapper function for plotting information about the dataset.
        :return: Plots of different dataset features.
        """
        self.plot_train_test_labels()
        self.plot_stat_as_pie(self.overall_dataset['expensive'],
                              ['Expensive items', 'Cheap items'],
                              title='Distribution of expensive and cheap items')

        self.plot_stat_as_pie(self.overall_dataset['light'],
                              ['Good lights', 'Bad lights'],
                              title='Distribution of recorded light '
                                    'conditions')

        self.plot_stat_as_pie(self.overall_dataset['background'],
                              ['Products in the background', 'Office'],
                              title='Distribution of recorded background')

        self.plot_stat_as_pie(self.overall_dataset['hand_location'],
                              ['Top', 'Side', 'Bottom'],
                              title='Distribution of hand location holding '
                                    'the item')
        self.plot_stat_as_pie(self.overall_dataset['visual_angle'],
                              ['Straight', 'Right', 'Left'],
                              title='Distribution of the item rotation angle '
                                    'in the scene')

    def print_example_images(self):
        """
        Wrapper function for plotting Super Store dataset records.
        :return: Plots of several chosen records.
        """
        image_list = []
        for i in range(100):
            image_list.append(random.randint(0, 1599))

        images = self.train_set.iloc[image_list]
        for index, row in images.iterrows():
            image_path = os.path.join(self.root_path, row['label'], 'input',
                                      f"{row['sample_id']}.jpeg")
            processed_image = self.draw_labeled_boxes(
                cv2.imread(image_path), row)
            self.show_image(row['sample_id'], processed_image)
        plt.show()

    def plot_train_test_labels(self):
        """
        Plot of train test sets distribution per class.
        :return: A plot showing the distribution of each class in the train
        and test sets.
        """
        _, train_counts = np.unique(self.train_set['label'], return_counts=True)
        _, test_counts = np.unique(self.test_set['label'], return_counts=True)
        pd.DataFrame({'train': train_counts, 'test': test_counts},
                     index=self.train_set['label'].unique()).plot.barh()

        plt.show()

    def plot_stat_as_pie(self, column, labels_list, title):
        """
        Function for plotting dataset features as pie chart.
        :param column: required. Pandas series object. A dataframe column
        containing a dataset  feature.
        :param labels_list: required. string list. List of all the unique
        values in the corresponding feature.
        :param title: required. string. The title of the plot.
        :return: A pie chart plot showing the values distrubution of the
        given feature.
        """
        values_list = column.value_counts().tolist()
        plt.pie(values_list,
                explode=None,
                labels=labels_list,
                autopct='%1.0f%%')
        plt.axis('equal')
        plt.title(title)
        plt.show()

    def draw_labeled_boxes(self, image_np, record):
        """
        Draws labeled boxes according to results on the given image.
        :param image_np: required. numpy array image
        :param record: required. Pandas series (row) represents the object
        information.
        :return: numpy array image with labeled boxes drawn
        """
        box = [record['f_rcnn_y1'], record['f_rcnn_x1'], record['f_rcnn_y2'],
               record['f_rcnn_x2']]
        image_np_copy = image_np.copy()
        color_idx = random.randint(0, len(self.COLORS) - 1)
        color = self.COLORS[color_idx]

        image_pil = Image.fromarray(np.uint8(image_np_copy)).convert(
            'RGB')
        image_pil = self.draw_bounding_box_on_image(image_pil, box, color, record[
            'label'])
        np.copyto(image_np_copy, np.array(image_pil))

        return image_np_copy

    def draw_bounding_box_on_image(self, image, box, color, box_label):
        """
        Draws the box and label on the given image.
        :param image: required. PIL image
        :param box: required. numpy array containing the bounding box
        information [top, left, bottom, right]
        :param color: required. bounding box color
        :param box_label: required. bounding box label
        :return: the given image with the corresponding bounding box and
        label of the main object.
        """
        im_width, im_height = image.size
        top, left, bottom, right = box

        # Draw the detected bounding box
        line_width = int(max(im_width, im_height) * 0.005)
        draw = ImageDraw.Draw(image)
        draw.rectangle(((left, top), (right, bottom)),
                       width=line_width,
                       outline=color)

        # Get a suitable font (in terms of size with respect to the image)
        font = ImageFont.load_default()
        text_width, text_height = font.getsize(box_label)

        # Draw the box label rectangle
        text_bottom = top + text_height
        text_rect = ((left, top),
                     (left + text_width + 2 * line_width,
                      text_bottom + 2 * line_width))
        draw.rectangle(text_rect, fill=color)

        # Draw the box label text
        # right below the upper-left horizontal line of the bounding box
        text_position = (left + line_width, top + line_width)
        draw.text(text_position, box_label, fill='white', font=font)
        return image

    def show_image(self, filename, image):
        """
        Shows the given image with its filename as title.
        :param filename: image filename
        :param image: image to show
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 8))
        plt.title(filename)
        plt.axis('off')
        plt.imshow(image)


if __name__ == '__main__':
    """
    example for creating SuperStore dataset:
    1. Download the dataset and store it in a root folder of your choice. 
    2. Run this module from the command line with the following inputs:
    super_store_util.py <dataset format> <root folder path>
    example:
    super_store_util.py faster_rcnn C://Users//Administrator//SuperStore_dataset
    After the execution of the script a new folder will be created next to the original one, in the chosen format. 
    """
    create_dataset(sys.argv[1:])

    # Below is code to analyse the super store dataset.
    # analysis = Super_store_dataset_analysis(sys.argv[2])
    # analysis.dataset_overview()
