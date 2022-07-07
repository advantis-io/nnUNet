from nnunet.experiment_planning.experiment_planner_baseline_2DUNet_v21 import ExperimentPlanner2D_v21
from nnunet.experiment_planning.experiment_planner_baseline_3DUNet_v21 import ExperimentPlanner3D_v21
from nnunet.paths import *


class ExperimentPlanner2D_v21_Prostate(ExperimentPlanner2D_v21):
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlanner2D_v21, self).__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.data_identifier = "nnUNetData_plans_v2.1_2D_crop"
        self.plans_fname = join(self.preprocessed_output_folder, "nnUNetPlansv2.1_crop_plans_2D.pkl")
        self.unet_base_num_features = 32
        self.preprocessor_name = "ProstatePreprocessorFor2D"


class ExperimentPlanner3D_v21_Prostate(ExperimentPlanner3D_v21):
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlanner3D_v21, self).__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.data_identifier = "nnUNetData_plans_v2.1_crop"
        self.plans_fname = join(self.preprocessed_output_folder, "nnUNetPlansv2.1_crop_plans_3D.pkl")
        self.unet_base_num_features = 32
        self.preprocessor_name = "ProstatePreprocessor"
