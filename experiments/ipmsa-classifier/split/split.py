# System imports
import sys
import os
from typing import List, Tuple, Union, Dict
from pvg.runner.dataset.dataset import VolumeLoader
from pvg.constants.pipeline import SplitKeys
from torch.utils.data import Dataset

# Add project root to sys.path
projectroot = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if projectroot not in sys.path:
    sys.path.insert(1, projectroot)

from constants.loris.general import Sequences, Timepoints as TP, Trials
from constants.loris.indexing import SubjectKeys as SJ_KEYS, ClinicalKeys as CK, UserKeys as UK
from constants.loris.lut import CLINICAL_MRI_VISITS_REGEX
from indexing.index_builder import IndexBuilder as IB
from indexing.split_utils import Splits, SplitOptions

class SplitGenerator:
    @staticmethod
    def create_exp_index(merged_index):
        """
            @params merged_index: merged version of indexes that were specified when generate_exp_splits.py is called
            @return exp_index: slimmed down version of merged_index containing data needed for a particular experiment
        """
        exp_index = {}
        
        ################# WRITE YOUR CODE HERE ################

        # Define required trials
        included_trials = [ Trials.OPERA1, Trials.OPERA2, Trials.DEFINE_ENDORSE, Trials.BRAVO ]

        # Define required sequences
        tp0_required_seqs = [Sequences.BEAST, Sequences.FLR,  Sequences.GVF, Sequences.CT2F ]
        tp1_required_seqs = tp0_required_seqs + [Sequences.NEWT2]
        tp2_required_seqs = tp1_required_seqs
        tp3_required_seqs = tp2_required_seqs

        # Create sequence templates. Note: sequence templates should be dicts with placeholder values (i.e. None).
        tp0_sequence_template = IB.make_leaf_template(tp0_required_seqs)
        tp1_sequence_template = IB.make_leaf_template(tp1_required_seqs)
        tp2_sequence_template = IB.make_leaf_template(tp2_required_seqs)
        tp3_sequence_template = IB.make_leaf_template(tp3_required_seqs)

        # Create clinical data templates
        clinical_data_template = {
            CK.Reference.REFERENCE_AGE: None,
            CK.Reference.REFERENCE_EDSS_SCORE: None,
            CK.Subject.TRIAL_ARM: None,
            CK.Subject.SEX: None,
            CK.MRI.LESION_GAD_CONSENSUS_COUNT: None,
            CK.MRI.LESION_T2_VOL: None,
        }

        # Create general template to use
        template = \
        {
            **{
                trial: {
                    SJ_KEYS.MRI_AND_LABEL: {
                        TP.W000: tp0_sequence_template,
                        TP.W024: tp1_sequence_template,
                        TP.W048: tp2_sequence_template,
                        TP.W096: tp3_sequence_template
                    },
                    SJ_KEYS.CLINICAL: {
                        CLINICAL_MRI_VISITS_REGEX[trial][TP.W000]: clinical_data_template,
                        CLINICAL_MRI_VISITS_REGEX[trial][TP.W048]: clinical_data_template,
                        CLINICAL_MRI_VISITS_REGEX[trial][TP.W096]: clinical_data_template
                    }
                } for trial in included_trials if trial != Trials.BRAVO
            },
            **{
                Trials.BRAVO: {
                    SJ_KEYS.MRI_AND_LABEL: {
                        TP.W000: tp0_sequence_template,
                        TP.W048: tp1_sequence_template,
                        TP.W096: tp2_sequence_template
                    },
                    SJ_KEYS.CLINICAL: {
                        CLINICAL_MRI_VISITS_REGEX[Trials.BRAVO][TP.W000]: clinical_data_template,
                        CLINICAL_MRI_VISITS_REGEX[Trials.BRAVO][TP.W048]: clinical_data_template,
                        CLINICAL_MRI_VISITS_REGEX[Trials.BRAVO][TP.W096]: clinical_data_template
                    }
                }
            }
        }

        # Copy info from merged_index into exp_index according to template
        IB.copy_with_template(merged_index, exp_index, template, regex_mode=True)

        # Rename timepoint keys under each trial
        trial_keys_to_rename = {
            **{
                trial: {
                    TP.W000: 'tp0',
                    TP.W024: 'tp1',
                    TP.W048: 'tp2',
                    TP.W096: 'tp3',
                    CLINICAL_MRI_VISITS_REGEX[trial][TP.W000]: 'tp0',
                    CLINICAL_MRI_VISITS_REGEX[trial][TP.W048]: 'tp2',
                    CLINICAL_MRI_VISITS_REGEX[trial][TP.W096]: 'tp3'
                } for trial in included_trials if trial != Trials.BRAVO
            },
            **{
                Trials.BRAVO: {
                    TP.W000: 'tp0',
                    TP.W048: 'tp2',
                    TP.W096: 'tp3',
                    CLINICAL_MRI_VISITS_REGEX[Trials.BRAVO][TP.W000]: 'tp0',
                    CLINICAL_MRI_VISITS_REGEX[Trials.BRAVO][TP.W048]: 'tp2',
                    CLINICAL_MRI_VISITS_REGEX[Trials.BRAVO][TP.W096]: 'tp3'
                }
            }
        }
        IB.rename_keys_in_groups(exp_index, trial_keys_to_rename, regex_mode=True)

        #######################################################

        return exp_index

    @staticmethod
    def create_splits(exp_index: dict):
        """
            @params exp_index: slimmed down index produced by create_exp_index method
            @return splits: dictionary containing 1 or more splits of train/val/test set
        """
        splits = {}

        ################# WRITE YOUR CODE HERE ################

        #splits = Splits.IID.generate_experiment(exp_index, (0.70,0.25,0.05))
        splits = Splits.IID.generate_experiment(exp_index, (0.8,0.1,0.1), SplitOptions.KEEP_GROUPS)
        #splits = Splits.IID.generate_kfold(exp_index, 3)
        #splits = Splits.IID.generate_kfold(exp_index, 3, SplitOptions.KEEP_GROUPS)

        #######################################################

        return splits
    
    
# ==== PVG specific split information ==== #

rootdir_args: Dict[str, str] = {'loris_dir': SJ_KEYS.MRI_AND_LABEL, 'clinical_dir': SJ_KEYS.CLINICAL}

class DatesetKeys:
    MRI = 'MRI'
    CLINICAL = 'CLINICAL'

class DatasetDescription:
    """
        Creates the train/val/test datasets given a data split and a dict containing all
        the required rootdirs to find a file.

        If the train/val/test dataset is a Dataset object, the default DataLoader
        will be used with Hyperparameters.batch_size and Hyperparameters.num_workers.

        If the train/val/test dataset is a list/tuple of Dataset objects, the custom
        DataLoader will be used instead.
    """
    def __init__(self):
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup_datasets(self, data_split, rootdir_dict):
        ################## MODIFY HERE #######################
        item_template = {
            'FLAIR': [
                (SJ_KEYS.MRI_AND_LABEL, 'tp0', Sequences.FLR),
                (SJ_KEYS.MRI_AND_LABEL, 'tp2', Sequences.FLR),
                (SJ_KEYS.MRI_AND_LABEL, 'tp3', Sequences.FLR)
            ],
            'GAD': [
                (SJ_KEYS.MRI_AND_LABEL, 'tp0', Sequences.GVF),
                (SJ_KEYS.MRI_AND_LABEL, 'tp2', Sequences.GVF),
                (SJ_KEYS.MRI_AND_LABEL, 'tp3', Sequences.GVF)
            ],
            'CT2F': [
                (SJ_KEYS.MRI_AND_LABEL, 'tp0', Sequences.CT2F),
                (SJ_KEYS.MRI_AND_LABEL, 'tp2', Sequences.CT2F),
                (SJ_KEYS.MRI_AND_LABEL, 'tp3', Sequences.CT2F)
            ],
            'NEWT2': [
                (SJ_KEYS.MRI_AND_LABEL, 'tp1', Sequences.NEWT2),
                (SJ_KEYS.MRI_AND_LABEL, 'tp2', Sequences.NEWT2),
                (SJ_KEYS.MRI_AND_LABEL, 'tp3', Sequences.NEWT2)
            ],
            'MASK': [
                (SJ_KEYS.MRI_AND_LABEL, 'tp0', Sequences.BEAST),
                (SJ_KEYS.MRI_AND_LABEL, 'tp2', Sequences.BEAST),
                (SJ_KEYS.MRI_AND_LABEL, 'tp3', Sequences.BEAST)
            ],
            'AGE': [(SJ_KEYS.CLINICAL, 'tp0', CK.Reference.REFERENCE_AGE)],
            'EDSS': [(SJ_KEYS.CLINICAL, 'tp0', CK.Reference.REFERENCE_EDSS_SCORE)],
            'TRIAL_ARM': [(SJ_KEYS.CLINICAL, 'tp0', CK.Subject.TRIAL_ARM)],
            'SEX': [(SJ_KEYS.CLINICAL, 'tp0', CK.Subject.SEX)],
            'LESION_GAD_CONSENSUS_COUNT': [
                (SJ_KEYS.CLINICAL, 'tp0', CK.MRI.LESION_GAD_CONSENSUS_COUNT),
                (SJ_KEYS.CLINICAL, 'tp2', CK.MRI.LESION_GAD_CONSENSUS_COUNT),
                (SJ_KEYS.CLINICAL, 'tp3', CK.MRI.LESION_GAD_CONSENSUS_COUNT)
            ],
            'LESION_T2_VOL': [
                (SJ_KEYS.CLINICAL, 'tp0', CK.MRI.LESION_T2_VOL),
                (SJ_KEYS.CLINICAL, 'tp2', CK.MRI.LESION_T2_VOL),
                (SJ_KEYS.CLINICAL, 'tp3', CK.MRI.LESION_T2_VOL)
            ]
        }

        self.train_dataset = VolumeLoader(data_split[SplitKeys.TRAIN], rootdir_dict, item_template)
        self.val_dataset = VolumeLoader(data_split[SplitKeys.VALIDATION], rootdir_dict, item_template)
        self.test_dataset = VolumeLoader(data_split[SplitKeys.TEST], rootdir_dict, item_template)
        ######################################################

    def get_train_dataset(self) -> Union[Dataset, List[Dataset], Tuple[Dataset]]:
        return self.train_dataset

    def get_val_dataset(self) -> Union[Dataset, List[Dataset], Tuple[Dataset]]:
        return self.val_dataset

    def get_test_dataset(self) -> Union[Dataset, List[Dataset], Tuple[Dataset]]:
        return self.test_dataset