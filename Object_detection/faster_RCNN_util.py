from enum import Enum
import pytorch_lightning as pl
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor,fasterrcnn_resnet50_fpn,\
    fasterrcnn_mobilenet_v3_large_fpn,fasterrcnn_mobilenet_v3_large_320_fpn
from itertools import chain
import numpy as np
import pandas as pd
import sys
from collections import Counter

"""
Module representing Faster RCNN model, mainly for loading the model and help 
functions. 
"""


def get_faster_rcnn_resnet50_fpn(num_classes,
                                 image_mean = [0.485, 0.456, 0.406],
                                 image_std = [0.229, 0.224, 0.225], min_size: int = 512,
                                 max_size: int = 1024):
    """

    :param num_classes: required. int. The number of classes the
    model was trained on.
    :param image_mean: optional. list of ints represent the mean of each
    pixel color (RGB) in the backbone layers of the model, default values
    are based on imagenet.
    :param image_std: optional. list of ints represent the std of each
    pixel color (RGB) in the backbone layers of the model, default values
    are based on imagenet.
    :param min_size: optional. int. image min size.
    :param max_size: optional. int. image max size.
    :return: Pytorch faster-RCNN model.
    """
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,
                                                      num_classes)
    model.num_classes = num_classes
    model.image_mean = image_mean
    model.image_std = image_std
    model.min_size = min_size
    model.max_size = max_size

    return model

def get_fasterrcnn_mobilenet_v3_large_fpn(num_classes,
                                 image_mean = [0.485, 0.456, 0.406],
                                 image_std = [0.229, 0.224, 0.225], min_size: int = 512,
                                 max_size: int = 1024):
    """

    :param num_classes: required. int. The number of classes the
    model was trained on.
    :param image_mean: optional. list of ints represent the mean of each
    pixel color (RGB) in the backbone layers of the model, default values
    are based on imagenet.
    :param image_std: optional. list of ints represent the std of each
    pixel color (RGB) in the backbone layers of the model, default values
    are based on imagenet.
    :param min_size: optional. int. image min size.
    :param max_size: optional. int. image max size.
    :return: Pytorch faster-RCNN model.
    """

    model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,
                                                      num_classes)
    model.num_classes = num_classes
    model.image_mean = image_mean
    model.image_std = image_std
    model.min_size = min_size
    model.max_size = max_size

    return model

def get_fasterrcnn_mobilenet_v3_large_320_fpn(num_classes,
                                 image_mean = [0.485, 0.456, 0.406],
                                 image_std = [0.229, 0.224, 0.225], min_size: int = 512,
                                 max_size: int = 1024):
    """

    :param num_classes: required. int. The number of classes the
    model was trained on.
    :param image_mean: optional. list of ints represent the mean of each
    pixel color (RGB) in the backbone layers of the model, default values
    are based on imagenet.
    :param image_std: optional. list of ints represent the std of each
    pixel color (RGB) in the backbone layers of the model, default values
    are based on imagenet.
    :param min_size: optional. int. image min size.
    :param max_size: optional. int. image max size.
    :return: Pytorch faster-RCNN model.
    """

    model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,
                                                      num_classes)
    model.num_classes = num_classes
    model.image_mean = image_mean
    model.image_std = image_std
    model.min_size = min_size
    model.max_size = max_size

    return model

class BBType(Enum):
    """
    Class representing if the bounding box is groundtruth or not.
    """

    GROUND_TRUTH = 1
    DETECTED = 2


class BBFormat(Enum):
    """
    Class representing the format of a bounding box.
    """

    XYWH = 1
    XYX2Y2 = 2
    PASCAL_XML = 3
    YOLO = 4

class CoordinatesType(Enum):
    """
    Class representing if the coordinates are relative to the
    image size or are absolute values.
        Developed by: Rafael Padilla
        Last modification: Apr 28 2018
    """

    RELATIVE = 1
    ABSOLUTE = 2


class MethodAveragePrecision(Enum):
    """
    Class representing if the coordinates are relative to the
    image size or are absolute values.
        Developed by: Rafael Padilla
        Last modification: Apr 28 2018
    """

    EVERY_POINT_INTERPOLATION = 1
    ELEVEN_POINT_INTERPOLATION = 2

class FasterRCNNLightning(pl.LightningModule):
    def __init__(
        self, model: torch.nn.Module, lr: float = 0.0001, iou_threshold: float = 0.5
    ):
        super().__init__()

        # Model
        self.model = model

        # Classes (background inclusive)
        self.num_classes = self.model.roi_heads.box_predictor.cls_score\
            .out_features

        # Learning rate
        self.lr = lr

        # IoU threshold
        self.iou_threshold = iou_threshold

        # Transformation parameters
        self.mean = model.image_mean
        self.std = model.image_std
        self.min_size = model.min_size
        self.max_size = model.max_size

        # Save hyperparameters
        # Saves model arguments to the ``hparams`` attribute.
        self.save_hyperparameters()

    def forward(self, x):
        self.model.eval()
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Batch
        x, y, x_name, y_name = batch  # tuple unpacking

        loss_dict = self.model(x, y)
        loss = sum(loss for loss in loss_dict.values())

        self.log_dict(loss_dict)
        return loss

    def from_dict_to_boundingbox(self, file: dict, name: str,
                                 groundtruth: bool = True):
        """Returns list of BoundingBox objects from groundtruth or prediction."""
        labels = file["labels"]
        boxes = file["boxes"]
        scores = np.array(file["scores"].cpu()) if not groundtruth else [
                                                                            None] * len(
            boxes)

        gt = BBType.GROUND_TRUTH if groundtruth else BBType.DETECTED

        return [
            BoundingBox(
                image_name=name,
                class_id=int(l),
                coordinates=tuple(bb),
                format=BBFormat.XYX2Y2,
                bb_type=gt,
                confidence=s,
            )
            for bb, l, s in zip(boxes, labels, scores)
        ]

    def validation_step(self, batch, batch_idx):
        # Batch
        x, y, x_name, y_name = batch

        # Inference
        preds = self.model(x)

        gt_boxes = [
            self.from_dict_to_boundingbox(file=target, name=name,
                                        groundtruth=True)
            for target, name in zip(y, x_name)
        ]
        gt_boxes = list(chain(*gt_boxes))

        pred_boxes = [
            self.from_dict_to_boundingbox(file=pred, name=name,
                                         groundtruth=False)
            for pred, name in zip(preds, x_name)
        ]
        pred_boxes = list(chain(*pred_boxes))

        return {"pred_boxes": pred_boxes, "gt_boxes": gt_boxes}

    def validation_epoch_end(self, outs):
        gt_boxes = [out["gt_boxes"] for out in outs]
        gt_boxes = list(chain(*gt_boxes))
        pred_boxes = [out["pred_boxes"] for out in outs]
        pred_boxes = list(chain(*pred_boxes))

        metric = self.get_pascalvoc_metrics(
            gt_boxes=gt_boxes,
            det_boxes=pred_boxes,
            iou_threshold=self.iou_threshold,
            method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
            generate_table=True,
        )

        per_class, m_ap = metric["per_class"], metric["m_ap"]
        self.log("Validation_mAP", m_ap)

        for key, value in per_class.items():
            self.log(f"Validation_AP_{key}", value["AP"])

    def test_step(self, batch, batch_idx):
        # Batch
        x, y, x_name, y_name = batch

        # Inference
        preds = self.model(x)

        gt_boxes = [
            self.from_dict_to_boundingbox(file=target, name=name,
                                        groundtruth=True)
            for target, name in zip(y, x_name)
        ]
        gt_boxes = list(chain(*gt_boxes))

        pred_boxes = [
            self.from_dict_to_boundingbox(file=pred, name=name,
                                         groundtruth=False)
            for pred, name in zip(preds, x_name)
        ]
        pred_boxes = list(chain(*pred_boxes))

        return {"pred_boxes": pred_boxes, "gt_boxes": gt_boxes}

    def test_epoch_end(self, outs):
        gt_boxes = [out["gt_boxes"] for out in outs]
        gt_boxes = list(chain(*gt_boxes))
        pred_boxes = [out["pred_boxes"] for out in outs]
        pred_boxes = list(chain(*pred_boxes))

        metric = self.get_pascalvoc_metrics(
            gt_boxes=gt_boxes,
            det_boxes=pred_boxes,
            iou_threshold=self.iou_threshold,
            method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
            generate_table=True,
        )

        per_class, m_ap = metric["per_class"], metric["m_ap"]
        self.log("Test_mAP", m_ap)

        for key, value in per_class.items():
            self.log(f"Test_AP_{key}", value["AP"])

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.005
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.75, patience=30, min_lr=0
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "Validation_mAP",
        }

    def calculate_ap_every_point(self,rec, prec):
        mrec = []
        mrec.append(0)
        [mrec.append(e) for e in rec]
        mrec.append(1)
        mpre = []
        mpre.append(0)
        [mpre.append(e) for e in prec]
        mpre.append(0)
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        ii = []
        for i in range(len(mrec) - 1):
            if mrec[1:][i] != mrec[0:-1][i]:
                ii.append(i + 1)
        ap = 0
        for i in ii:
            ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
        return [ap, mpre[0: len(mpre) - 1], mrec[0: len(mpre) - 1], ii]

    def calculate_ap_11_point_interp(self,rec, prec, recall_vals=11):
        mrec = []
        # mrec.append(0)
        [mrec.append(e) for e in rec]
        # mrec.append(1)
        mpre = []
        # mpre.append(0)
        [mpre.append(e) for e in prec]
        # mpre.append(0)
        recall_values = np.linspace(0, 1, recall_vals)
        recall_values = list(recall_values[::-1])
        rho_interp = []
        recallValid = []
        # For each recall_values (0, 0.1, 0.2, ... , 1)
        for r in recall_values:
            # Obtain all recall values higher or equal than r
            argGreaterRecalls = np.argwhere(mrec[:] >= r)
            pmax = 0
            # If there are recalls above r
            if argGreaterRecalls.size != 0:
                pmax = max(mpre[argGreaterRecalls.min():])
            recallValid.append(r)
            rho_interp.append(pmax)
        # By definition AP = sum(max(precision whose recall is above r))/11
        ap = sum(rho_interp) / len(recall_values)
        # Generating values for the plot
        rvals = []
        rvals.append(recallValid[0])
        [rvals.append(e) for e in recallValid]
        rvals.append(0)
        pvals = []
        pvals.append(0)
        [pvals.append(e) for e in rho_interp]
        pvals.append(0)
        # rho_interp = rho_interp[::-1]
        cc = []
        for i in range(len(rvals)):
            p = (rvals[i], pvals[i - 1])
            if p not in cc:
                cc.append(p)
            p = (rvals[i], pvals[i])
            if p not in cc:
                cc.append(p)
        recall_values = [i[0] for i in cc]
        rho_interp = [i[1] for i in cc]
        return [ap, rho_interp, recall_values, None]

    def get_pascalvoc_metrics(self, gt_boxes, det_boxes,
            iou_threshold=0.5,
            method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
            generate_table=False,
    ):
        """Get the metrics used by the VOC Pascal 2012 challenge.
        Args:
            boundingboxes: Object of the class BoundingBoxes representing ground truth and detected
            bounding boxes;
            iou_threshold: IOU threshold indicating which detections will be considered tp or fp
            (dget_pascalvoc_metricsns:
            A dictioanry contains information and metrics of each class.
            The key represents the class and the values are:
            dict['class']: class representing the current dictionary;
            dict['precision']: array with the precision values;
            dict['recall']: array with the recall values;
            dict['AP']: average precision;
            dict['interpolated precision']: interpolated precision values;
            dict['interpolated recall']: interpolated recall values;
            dict['total positives']: total number of ground truth positives;
            dict['total tp']: total number of True Positive detections;
            dict['total fp']: total number of False Positive detections;"""
        ret = {}
        # Get classes of all bounding boxes separating them by classes
        gt_classes_only = []
        classes_bbs = {}
        for bb in gt_boxes:
            c = bb.get_class_id()
            gt_classes_only.append(c)
            classes_bbs.setdefault(c, {"gt": [], "det": []})
            classes_bbs[c]["gt"].append(bb)
        gt_classes_only = list(set(gt_classes_only))
        for bb in det_boxes:
            c = bb.get_class_id()
            classes_bbs.setdefault(c, {"gt": [], "det": []})
            classes_bbs[c]["det"].append(bb)

        # Precision x Recall is obtained individually by each class
        for c, v in classes_bbs.items():
            # Report results only in the classes that are in the GT
            if c not in gt_classes_only:
                continue
            npos = len(v["gt"])
            # sort detections by decreasing confidence
            dects = [
                a
                for a in sorted(v["det"], key=lambda bb: bb.get_confidence(),
                                reverse=True)
            ]
            tp = np.zeros(len(dects))
            fp = np.zeros(len(dects))
            # create dictionary with amount of expected detections for each image
            detected_gt_per_image = Counter(
                [bb.get_image_name() for bb in gt_boxes])
            for key, val in detected_gt_per_image.items():
                detected_gt_per_image[key] = np.zeros(val)
            # print(f'Evaluating class: {c}')
            dict_table = {
                "image": [],
                "confidence": [],
                "tp": [],
                "fp": [],
                "acc tp": [],
                "acc fp": [],
                "precision": [],
                "recall": [],
            }
            # Loop through detections
            for idx_det, det in enumerate(dects):
                img_det = det.get_image_name()

                if generate_table:
                    dict_table["image"].append(img_det)
                    dict_table["confidence"].append(
                        f"{100 * det.get_confidence():.2f}%")

                # Find ground truth image
                gt = [gt for gt in classes_bbs[c]["gt"] if
                      gt.get_image_name() == img_det]
                # Get the maximum iou among all detectins in the image
                iou_max = sys.float_info.min
                # Given the detection det, find ground-truth with the highest iou
                for j, g in enumerate(gt):
                    # print('Ground truth gt => %s' %
                    #       str(g.get_absolute_bounding_box(format=BBFormat.XYX2Y2)))
                    iou = BoundingBox.iou(det, g)
                    if iou > iou_max:
                        iou_max = iou
                        id_match_gt = j
                # Assign detection as tp or fp
                if iou_max >= iou_threshold:
                    # gt was not matched with any detection
                    if detected_gt_per_image[img_det][id_match_gt] == 0:
                        tp[idx_det] = 1  # detection is set as true positive
                        detected_gt_per_image[img_det][
                            id_match_gt
                        ] = 1  # set flag to identify gt as already 'matched'
                        # print("tp")
                        if generate_table:
                            dict_table["tp"].append(1)
                            dict_table["fp"].append(0)
                    else:
                        fp[idx_det] = 1  # detection is set as false positive
                        if generate_table:
                            dict_table["fp"].append(1)
                            dict_table["tp"].append(0)
                        # print("fp")
                # - A detected "cat" is overlaped with a GT "cat" with IOU >= iou_threshold.
                else:
                    fp[idx_det] = 1  # detection is set as false positive
                    if generate_table:
                        dict_table["fp"].append(1)
                        dict_table["tp"].append(0)
                    # print("fp")
            # compute precision, recall and average precision
            acc_fp = np.cumsum(fp)
            acc_tp = np.cumsum(tp)
            rec = acc_tp / npos
            prec = np.divide(acc_tp, (acc_fp + acc_tp))
            if generate_table:
                dict_table["acc tp"] = list(acc_tp)
                dict_table["acc fp"] = list(acc_fp)
                dict_table["precision"] = list(prec)
                dict_table["recall"] = list(rec)
                table = pd.DataFrame(dict_table)
            else:
                table = None
            # Depending on the method, call the right implementation
            if method == MethodAveragePrecision.EVERY_POINT_INTERPOLATION:
                [ap, mpre, mrec, ii] = self.calculate_ap_every_point(rec, prec)
            elif method == MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION:
                [ap, mpre, mrec, _] = self.calculate_ap_11_point_interp(rec,
                                                                       prec)
            else:
                Exception("method not defined")
            # add class result in the dictionary to be returned
            ret[c] = {
                "precision": prec,
                "recall": rec,
                "AP": ap,
                "interpolated precision": mpre,
                "interpolated recall": mrec,
                "total positives": npos,
                "total tp": np.sum(tp),
                "total fp": np.sum(fp),
                "method": method,
                "iou": iou_threshold,
                "table": table,
            }
        # For m_ap, only the classes in the gt set should be considered
        m_ap = sum(
            [v["AP"] for k, v in ret.items() if k in gt_classes_only]) / len(
            gt_classes_only
        )
        return {"per_class": ret, "m_ap": m_ap}

class BoundingBox:
    """Class representing a bounding box."""

    def __init__(self, image_name,class_id = None,coordinates = None,
        type_coordinates = CoordinatesType.ABSOLUTE,
        img_size = None,
        bb_type = BBType.GROUND_TRUTH,
        confidence= None,
        format= BBFormat.XYWH,
    ):
        """ Constructor.
        Parameters
        ----------
            image_name : str
                String representing the name of the image.
            class_id : str
                String value representing class id.
            coordinates : tuple
                Tuple with 4 elements whose values (float) represent coordinates of the bounding \\
                    box.
                The coordinates can be (x, y, w, h)=>(float,float,float,float) or(x1, y1, x2, y2)\\
                    =>(float,float,float,float).
                See parameter `format`.
            type_coordinates : Enum (optional)
                Enum representing if the bounding box coordinates (x,y,w,h) are absolute or relative
                to size of the image. Default:'Absolute'.
            img_size : tuple (optional)
                Image size in the format (width, height)=>(int, int) representinh the size of the
                image of the bounding box. If type_coordinates is 'Relative', img_size is required.
            bb_type : Enum (optional)
                Enum identifying if the bounding box is a ground truth or a detection. If it is a
                detection, the confidence must be informed.
            confidence : float (optional)
                Value representing the confidence of the detected object. If detectionType is
                Detection, confidence needs to be informed.
            format : Enum
                Enum (BBFormat.XYWH or BBFormat.XYX2Y2) indicating the format of the coordinates of
                the bounding boxes.
                BBFormat.XYWH: <left> <top> <width> <height>
                BBFomat.XYX2Y2: <left> <top> <right> <bottom>.
                BBFomat.YOLO: <x_center> <y_center> <width> <height>. (relative)
        """

        self._image_name = image_name
        self._type_coordinates = type_coordinates
        self._confidence = confidence
        self._class_id = class_id
        self._format = format
        if bb_type == BBType.DETECTED and confidence is None:
            raise IOError(
                "For bb_type='Detected', it is necessary to inform the confidence value."
            )
        self._bb_type = bb_type

        if img_size is None:
            self._width_img = None
            self._height_img = None
        else:
            self._width_img = img_size[0]
            self._height_img = img_size[1]

        self.set_coordinates(
            coordinates, img_size=img_size, type_coordinates=self._type_coordinates
        )

    def set_coordinates(self, coordinates, type_coordinates, img_size=None):
        self._type_coordinates = type_coordinates
        if type_coordinates == CoordinatesType.RELATIVE and img_size is None:
            raise IOError(
                "Parameter 'img_size' is required. It is necessary to inform the image size."
            )

        # If relative coordinates, convert to absolute values
        # For relative coords: (x,y,w,h)=(X_center/img_width , Y_center/img_height)
        if type_coordinates == CoordinatesType.RELATIVE:
            self._width_img = img_size[0]
            self._height_img = img_size[1]
            if self._format == BBFormat.XYWH:
                (self._x, self._y, self._w, self._h) = \
                    self.convert_to_absolute_values(
                    img_size, coordinates
                )
                self._x2 = self._w
                self._y2 = self._h
                self._w = self._x2 - self._x
                self._h = self._y2 - self._y
            elif self._format == BBFormat.XYX2Y2:
                x1, y1, x2, y2 = coordinates
                # Converting to absolute values
                self._x = round(x1 * self._width_img)
                self._x2 = round(x2 * self._width_img)
                self._y = round(y1 * self._height_img)
                self._y2 = round(y2 * self._height_img)
                self._w = self._x2 - self._x
                self._h = self._y2 - self._y
            else:
                raise IOError(
                    "For relative coordinates, the format must be XYWH (x,y,width,height)"
                )
        # For absolute coords: (x,y,w,h)=real bb coords
        else:
            self._x = coordinates[0]
            self._y = coordinates[1]
            if self._format == BBFormat.XYWH:
                self._w = coordinates[2]
                self._h = coordinates[3]
                self._x2 = self._x + self._w
                self._y2 = self._y + self._h
            else:  # self._format == BBFormat.XYX2Y2: <left> <top> <right> <bottom>.
                self._x2 = coordinates[2]
                self._y2 = coordinates[3]
                self._w = self._x2 - self._x
                self._h = self._y2 - self._y
        # Convert all values to float
        self._x = float(self._x)
        self._y = float(self._y)
        self._w = float(self._w)
        self._h = float(self._h)
        self._x2 = float(self._x2)
        self._y2 = float(self._y2)

    def get_absolute_bounding_box(self, format=BBFormat.XYWH):
        """Get bounding box in its absolute format.
        Parameters
        ----------
        format : Enum
            Format of the bounding box (BBFormat.XYWH or BBFormat.XYX2Y2) to be retreived.
        Returns
        -------
        tuple
            Four coordinates representing the absolute values of the bounding box.
            If specified format is BBFormat.XYWH, the coordinates are (upper-left-X, upper-left-Y,
            width, height).
            If format is BBFormat.XYX2Y2, the coordinates are (upper-left-X, upper-left-Y,
            bottom-right-X, bottom-right-Y).
        """
        if format == BBFormat.XYWH:
            return self._x, self._y, self._w, self._h
        elif format == BBFormat.XYX2Y2:
            return self._x, self._y, self._x2, self._y2

    def convert_to_relative_values(self,size, box):
        dw = 1.0 / (size[0])
        dh = 1.0 / (size[1])
        cx = (box[1] + box[0]) / 2.0
        cy = (box[3] + box[2]) / 2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = cx * dw
        y = cy * dh
        w = w * dw
        h = h * dh
        # YOLO's format
        # x,y => (bounding_box_center)/width_of_the_image
        # w => bounding_box_width / width_of_the_image
        # h => bounding_box_height / height_of_the_image
        return x, y, w, h

    # size => (width, height) of the image
    # box => (centerX, centerY, w, h) of the bounding box relative to the image
    def convert_to_absolute_values(self,size, box):
        w_box = size[0] * box[2]
        h_box = size[1] * box[3]

        x1 = (float(box[0]) * float(size[0])) - (w_box / 2)
        y1 = (float(box[1]) * float(size[1])) - (h_box / 2)
        x2 = x1 + w_box
        y2 = y1 + h_box
        return round(x1), round(y1), round(x2), round(y2)

    def get_relative_bounding_box(self, img_size=None):
        """Get bounding box in its relative format.
        Parameters
        ----------
        img_size : tuple
            Image size in the format (width, height)=>(int, int)
        Returns
        -------
        tuple
            Four coordinates representing the relative values of the bounding box (x,y,w,h) where:
                x,y : bounding_box_center/width_of_the_image
                w   : bounding_box_width/width_of_the_image
                h   : bounding_box_height/height_of_the_image
        """
        if img_size is None and self._width_img is None and self._height_img is None:
            raise IOError(
                "Parameter 'img_size' is required. It is necessary to inform the image size."
            )
        if img_size is not None:
            return self.convert_to_relative_values(
                (img_size[0], img_size[1]), (self._x, self._x2, self._y, self._y2)
            )
        else:
            return self.convert_to_relative_values(
                (self._width_img, self._height_img),
                (self._x, self._x2, self._y, self._y2),
            )

    def get_image_name(self):
        """Get the string that represents the image.
        Returns
        -------
        string
            Name of the image.
        """
        return self._image_name

    def get_confidence(self):
        """Get the confidence level of the detection. If bounding box type is BBType.GROUND_TRUTH,
        the confidence is None.
        Returns
        -------
        float
            Value between 0 and 1 representing the confidence of the detection.
        """
        return self._confidence

    def get_format(self):
        """Get the format of the bounding box (BBFormat.XYWH or BBFormat.XYX2Y2).
        Returns
        -------
        Enum
            Format of the bounding box. It can be either:
                BBFormat.XYWH: <left> <top> <width> <height>
                BBFomat.XYX2Y2: <left> <top> <right> <bottom>.
        """
        return self._format

    def set_class_id(self, class_id):
        self._class_id = class_id

    def set_bb_type(self, bb_type):
        self._bb_type = bb_type

    def get_class_id(self):
        """Get the class of the object the bounding box represents.
        Returns
        -------
        string
            Class of the detected object (e.g. 'cat', 'dog', 'person', etc)
        """
        return self._class_id

    def get_image_size(self):
        """Get the size of the image where the bounding box is represented.
        Returns
        -------
        tupe
            Image size in pixels in the format (width, height)=>(int, int)
        """
        return self._width_img, self._height_img

    def get_area(self):
        # assert isclose(self._w * self._h, (self._x2 - self._x) * (self._y2 - self._y))
        assert self._x2 > self._x
        assert self._y2 > self._y
        return (self._x2 - self._x + 1) * (self._y2 - self._y + 1)

    def get_coordinates_type(self):
        """Get type of the coordinates (CoordinatesType.RELATIVE or CoordinatesType.ABSOLUTE).
        Returns
        -------
        Enum
            Enum representing if the bounding box coordinates (x,y,w,h) are absolute or relative
                to size of the image (CoordinatesType.RELATIVE or CoordinatesType.ABSOLUTE).
        """
        return self._type_coordinates

    def get_bb_type(self):
        """Get type of the bounding box that represents if it is a ground-truth or detected box.
        Returns
        -------
        Enum
            Enum representing the type of the bounding box (BBType.GROUND_TRUTH or BBType.DETECTED)
        """
        return self._bb_type

    def __str__(self):
        abs_bb_xywh = self.get_absolute_bounding_box(format=BBFormat.XYWH)
        abs_bb_xyx2y2 = self.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
        area = self.get_area()
        return (
            f"image name: {self._image_name}\n"
            f"image size: {self.get_image_size()}\n"
            f"class: {self._class_id}\n"
            f"bb (XYWH): {abs_bb_xywh}\n"
            f"bb (X1Y1X2Y2): {abs_bb_xyx2y2}\n"
            f"area: {area}\n"
            f"bb_type: {self._bb_type}"
        )

    def __eq__(self, other):
        if not isinstance(other, BoundingBox):
            # unrelated types
            return False
        return str(self) == str(other)

    def __repr__(self):
        abs_bb_xywh = self.get_absolute_bounding_box(format=BBFormat.XYWH)
        abs_bb_xyx2y2 = self.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
        area = self.get_area()
        return f"{self._bb_type}(bb (XYWH): {abs_bb_xywh}, bb (X1Y1X2Y2): {abs_bb_xyx2y2}, area: {area}), class: {self._class_id})"

    @staticmethod
    def iou(box_a, box_b):
        coords_a = box_a.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
        coords_b = box_b.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
        # if boxes do not intersect
        if BoundingBox.have_intersection(coords_a, coords_b) is False:
            return 0
        inter_area = BoundingBox.get_intersection_area(coords_a, coords_b)
        union = BoundingBox.get_union_areas(box_a, box_b, inter_area=inter_area)
        # intersection over union
        iou = inter_area / union
        assert iou >= 0
        return iou

    # boxA = (Ax1,Ay1,Ax2,Ay2)
    # boxB = (Bx1,By1,Bx2,By2)
    @staticmethod
    def have_intersection(box_a, box_b):
        if isinstance(box_a, BoundingBox):
            box_a = box_a.get_absolute_bounding_box(BBFormat.XYX2Y2)
        if isinstance(box_b, BoundingBox):
            box_b = box_b.get_absolute_bounding_box(BBFormat.XYX2Y2)
        if box_a[0] > box_b[2]:
            return False  # boxA is right of boxB
        if box_b[0] > box_a[2]:
            return False  # boxA is left of boxB
        if box_a[3] < box_b[1]:
            return False  # boxA is above boxB
        if box_a[1] > box_b[3]:
            return False  # boxA is below boxB
        return True

    @staticmethod
    def get_intersection_area(box_a, box_b):
        if isinstance(box_a, BoundingBox):
            box_a = box_a.get_absolute_bounding_box(BBFormat.XYX2Y2)
        if isinstance(box_b, BoundingBox):
            box_b = box_b.get_absolute_bounding_box(BBFormat.XYX2Y2)
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])
        # intersection area
        return (x_b - x_a + 1) * (y_b - y_a + 1)

    @staticmethod
    def get_union_areas(box_a, box_b, inter_area=None):
        area_a = box_a.get_area()
        area_b = box_b.get_area()
        if inter_area is None:
            inter_area = BoundingBox.get_intersection_area(box_a, box_b)
        return float(area_a + area_b - inter_area)

    @staticmethod
    def get_amount_bounding_box_all_classes(bounding_boxes, reverse=False):
        classes = list(set([bb._class_id for bb in bounding_boxes]))
        ret = {}
        for c in classes:
            ret[c] = len(BoundingBox.get_bounding_box_by_class(bounding_boxes, c))
        # Sort dictionary by the amount of bounding boxes
        ret = {
            k: v
            for k, v in sorted(ret.items(), key=lambda item: item[1], reverse=reverse)
        }
        return ret

    @staticmethod
    def get_bounding_box_by_class(bounding_boxes, class_id):
        # get only specified bounding box type
        return [bb for bb in bounding_boxes if bb.get_class_id() == class_id]

    @staticmethod
    def get_bounding_boxes_by_image_name(bounding_boxes, image_name):
        # get only specified bounding box type
        return [bb for bb in bounding_boxes if bb.get_image_name() == image_name]

    @staticmethod
    def get_total_images(bounding_boxes):
        return len(list(set([bb.get_image_name() for bb in bounding_boxes])))

    @staticmethod
    def get_average_area(bounding_boxes):
        areas = [bb.get_area() for bb in bounding_boxes]
        return sum(areas) / len(areas)




