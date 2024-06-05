import patch_attack_controller

def generate_digital_patch_attack():
    """
    Demo function that generates digital patches. Uses MSCOCO dataset and the Dpatch attack.
    :return: The crafted adversarial patch is stored in an output folder in a numpy format.
    """
    patch_attack_controller.digital_attack_demo()

def generate_physical_patch_attack():
    """
   Demo function that generates physical patches. Uses SuperStore dataset and the Dpatch q robust Dpatch attack.
   :return: The crafted adversarial patch is stored in an output folder in a numpy format.
   """
    patch_attack_controller.physical_attack_demo()


def evaluate_patch_attack_demo(dataset_path,patch_path,output_path,target_model_path,number_of_classes,target_class):
    """
    Demo function for generating and evaluating digital or physical patch attacks.

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

    patch_attack_controller.apply_patch_to_dataset(dataset_path, patch_path, output_path,
                                                   target_model_path, number_of_classes,
                                                   target_class)

