# X-Detect

Code project for X-Detect, a noval adversarial patch detector for object detection. 


## Description

We present X-Detect, a novel adversarial patch detector that can:
1. Detect adversarial samples in real time, allowing the defender to take preventive action.
2. Provide explanations for the alerts raised to support the defender's decision-making process.
3. Handle unfamiliar threats in the form of new attacks.

Given a new scene, X-Detect uses an ensemble of explainable-by-design detectors 
that utilize object extraction, scene manipulation, and feature transformation techniques 
to determine whether an alert needs to be raised.

This project evaluate X-detect capabilities on both physical and digital use cases using five different attack scenarios 
(including adaptive attacks) and the MS COCO dataset and our new Superstore dataset.

For more information, please see the paper draft.

## Getting started

In order to use X-Detect's full capabilities, please perform the following steps: 
1. Download the object detection models from the following link and store the models under the project's root folder: https://drive.google.com/file/d/1GH03Gx4vD6aVyHQd3Ekkk_SAyF8x7Z9m/view?usp=sharing
2. Download the evaluation set (adversarial scenes and videos that used to evaluate X-Detect) from the following link and store the evaluation set under the project's root folder:  https://drive.google.com/file/d/1v-1p6ghKnSz83h4zP9oWwBwiFFeA1_Ah/view?usp=sharing
3. Download the data util (MS COCO annotations file, MS COCO prototypes and Superstore prototypes) from https://drive.google.com/file/d/1rWUfUq_0CskTG796rEE9mc78otcHfJVK/view?usp=sharing
4. Please use the following files tree to store the downloaded directories in the correct location: 

![Alt text](samples_for_readme/files_tree.PNG?raw=true "Title")

5. Install all requireid packages from the requirements.txt file.  
6. Start a demo experiment using the Evaluation_manager/Demo.py module:
   1. Chose which use-case to explore in the experiment (physical/digital/small physical).
   2. Chose which attack scenario to use in the experiment from the following scenarios:
      1. Physical use-case - White-Box, Gray-box, Model specific, Model agnostic, Adaptive attacks.
      2. Digital use-case - White-Box, Model specific, Model agnostic.
      3. Small physical - "White-Box" by default.

***

<!-- 
## License
For open source projects, say how it is licensed. -->
