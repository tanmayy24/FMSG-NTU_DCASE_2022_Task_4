import os
import random
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from torch.autograd import Variable
from dcase_util.data import ProbabilityEncoder
import torchmetrics

from desed_task.data_augm import mixup
from desed_task.utils.scaler import TorchScaler
import numpy as np
from local.data_aug import *

from .utils import (
    batched_decode_preds,
    log_sedeval_metrics,
)
from desed_task.evaluation.evaluation_measures import (
    compute_per_intersection_macro_f1,
    compute_psds_from_operating_points,
)

def mixup_data_sup(x, y, alpha=1.0):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = np.random.permutation(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_data(x, y, z, alpha=1.0):
    '''Compute the mixup data. Return mixed inputs, mixed target, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = np.random.permutation(batch_size)
    x, y, z = x.data.cpu().numpy(), y.data.cpu().numpy(), z.data.cpu().numpy()
    mixed_x = torch.Tensor(lam * x + (1 - lam) * x[index,:])
    mixed_y = torch.Tensor(lam * y + (1 - lam) * y[index,:])
    mixed_z = torch.Tensor(lam * z + (1 - lam) * z[index,:])

    mixed_x = Variable(mixed_x.cuda())
    mixed_y = Variable(mixed_y.cuda())
    mixed_z = Variable(mixed_z.cuda())
    return mixed_x, mixed_y, mixed_z, lam

class SEDTask4(pl.LightningModule):
    """ Pytorch lightning module for the SED 2021 baseline
    Args:
        config: dict, the dictionary to be used for the current experiment/
        encoder: ManyHotEncoder object, object to encode and decode labels.
        sed_student: torch.Module, the student model to be trained. The teacher model will be
        opt: torch.optimizer.Optimizer object, the optimizer to be used
        train_data: torch.utils.data.Dataset subclass object, the training data to be used.
        valid_data: torch.utils.data.Dataset subclass object, the validation data to be used.
        test_data: torch.utils.data.Dataset subclass object, the test data to be used.
        train_sampler: torch.utils.data.Sampler subclass object, the sampler to be used in the training dataloader.
        scheduler: asteroid.engine.schedulers.BaseScheduler subclass object, the scheduler to be used. This is
            used to apply ramp-up during training for example.
        fast_dev_run: bool, whether to launch a run with only one batch for each set, this is for development purpose,
            to test the code runs.
    """

    def __init__(
        self,
        config,
        encoder,
        sed_student,
        opt=None,
        train_data=None,
        valid_data=None,
        test_data=None,
        train_sampler=None,
        scheduler=None,
        fast_dev_run=False,
        evaluation=False
    ):
        super(SEDTask4, self).__init__()
        self.config = config

        self.encoder = encoder
        self.sed_student = sed_student
        self.sed_teacher = deepcopy(sed_student)
        self.opt = opt
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.train_sampler = train_sampler
        self.scheduler = scheduler
        self.fast_dev_run = fast_dev_run
        self.evaluation=evaluation
        if self.fast_dev_run:
            self.num_workers = 1
        else:
            self.num_workers = self.config["training"]["num_workers"]

        feat_params = self.config["feats"]
        self.mel_spec = MelSpectrogram(
            sample_rate=feat_params["sample_rate"],
            n_fft=feat_params["n_window"],
            win_length=feat_params["n_window"],
            hop_length=feat_params["hop_length"],
            f_min=feat_params["f_min"],
            f_max=feat_params["f_max"],
            n_mels=feat_params["n_mels"],
            window_fn=torch.hamming_window,
            wkwargs={"periodic": False},
            power=1,
        )

        for param in self.sed_teacher.parameters():
            param.detach_()

        # instantiating losses
        self.supervised_loss = torch.nn.BCELoss()
        if config["training"]["self_sup_loss"] == "mse":
            self.selfsup_loss = torch.nn.MSELoss()
        elif config["training"]["self_sup_loss"] == "bce":
            self.selfsup_loss = torch.nn.BCELoss()
        else:
            raise NotImplementedError

        # for weak labels we simply compute f1 score
        self.get_weak_student_f1_seg_macro = torchmetrics.classification.f_beta.F1(
            len(self.encoder.labels),
            average="macro",
            compute_on_step=False,
        )

        self.get_weak_teacher_f1_seg_macro = torchmetrics.classification.f_beta.F1(
            len(self.encoder.labels),
            average="macro",
            compute_on_step=False,
        )

        self.scaler = self._init_scaler()

        # buffer for event based scores which we compute using sed-eval

        self.val_buffer_student_synth = {
            k: pd.DataFrame() for k in self.config["training"]["val_thresholds"]
        }
        self.val_buffer_teacher_synth = {
            k: pd.DataFrame() for k in self.config["training"]["val_thresholds"]
        }

        self.val_buffer_student_test = {
            k: pd.DataFrame() for k in self.config["training"]["val_thresholds"]
        }
        self.val_buffer_teacher_test = {
            k: pd.DataFrame() for k in self.config["training"]["val_thresholds"]
        }

        test_n_thresholds = self.config["training"]["n_test_thresholds"]
        test_thresholds = np.arange(
            1 / (test_n_thresholds * 2), 1, 1 / test_n_thresholds
        )
        self.test_psds_buffer_student = {k: pd.DataFrame() for k in test_thresholds}
        self.test_psds_buffer_teacher = {k: pd.DataFrame() for k in test_thresholds}

        self.decoded_student_05_buffer = pd.DataFrame()
        self.decoded_teacher_05_buffer = pd.DataFrame()
        self.prediction_dfs = pd.DataFrame()

    def update_ema(self, alpha, global_step, model, ema_model):
        """ Update teacher model parameters

        Args:
            alpha: float, the factor to be used between each updated step.
            global_step: int, the current global step to be used.
            model: torch.Module, student model to use
            ema_model: torch.Module, teacher model to use
        """
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_params, params in zip(ema_model.parameters(), model.parameters()):
            ema_params.data.mul_(alpha).add_(params.data, alpha=1 - alpha)

    def _init_scaler(self):
        """ Scaler inizialization

        Raises:
            NotImplementedError: in case of not Implemented scaler

        Returns:
            TorchScaler: returns the scaler
        """

        if self.config["scaler"]["statistic"] == "instance":
            scaler = TorchScaler(
                "instance",
                self.config["scaler"]["normtype"],
                self.config["scaler"]["dims"],
            )

            return scaler
        elif self.config["scaler"]["statistic"] == "dataset":
            # we fit the scaler
            scaler = TorchScaler(
                "dataset",
                self.config["scaler"]["normtype"],
                self.config["scaler"]["dims"],
            )
        else:
            raise NotImplementedError
        if self.config["scaler"]["savepath"] is not None:
            if os.path.exists(self.config["scaler"]["savepath"]):
                scaler = torch.load(self.config["scaler"]["savepath"])
                print(
                    "Loaded Scaler from previous checkpoint from {}".format(
                        self.config["scaler"]["savepath"]
                    )
                )
                return scaler

        self.train_loader = self.train_dataloader()
        scaler.fit(
            self.train_loader,
            transform_func=lambda x: self.take_log(self.mel_spec(x[0])),
        )

        if self.config["scaler"]["savepath"] is not None:
            torch.save(scaler, self.config["scaler"]["savepath"])
            print(
                "Saving Scaler from previous checkpoint at {}".format(
                    self.config["scaler"]["savepath"]
                )
            )
            return scaler

    def take_log(self, mels):
        """ Apply the log transformation to mel spectrograms.
        Args:
            mels: torch.Tensor, mel spectrograms for which to apply log.

        Returns:
            Tensor: logarithmic mel spectrogram of the mel spectrogram given as input
        """

        amp_to_db = AmplitudeToDB(stype="amplitude")
        amp_to_db.amin = 1e-5  # amin= 1e-5 as in librosa
        return amp_to_db(mels).clamp(min=-50, max=80)  # clamp to reproduce old code

    def detect(self, mel_feats, model):
        return model(self.scaler(self.take_log(mel_feats)))

    def training_step(self, batch, batch_indx):
        """ Apply the training for one batch (a step). Used during trainer.fit

        Args:
            batch: torch.Tensor, batch input tensor
            batch_indx: torch.Tensor, 1D tensor of indexes to know which data are present in each batch.

        Returns:
           torch.Tensor, the loss to take into account.
        """

        audio, labels, padded_indxs = batch
        indx_synth, indx_weak, indx_unlabelled = self.config["training"]["batch_size"]
        features = self.mel_spec(audio)

        batch_num = features.shape[0]
        # deriving masks for each dataset
        strong_mask = torch.zeros(batch_num).to(features).bool()
        weak_mask = torch.zeros(batch_num).to(features).bool()
        ul_mask = torch.zeros(batch_num).to(features).bool()
        strong_mask[:indx_synth] = 1
        weak_mask[indx_synth : indx_weak + indx_synth] = 1
        ul_mask[indx_weak + indx_synth:] = 1

        # deriving weak labels
        labels_weak = (torch.sum(labels[weak_mask], -1) > 0).float()
        #Frame Shift
        features, labels = frame_shift(features, labels, self.config["training"]["net_pooling"])
        
        mixup_type = self.config["training"].get("mixup")
        if mixup_type is not None and 0.5 > random.random():
            features[weak_mask], labels_weak = mixup(
                features[weak_mask], labels_weak, mixup_label_type=mixup_type
            )
            features[strong_mask], labels[strong_mask] = mixup(
                features[strong_mask], labels[strong_mask], mixup_label_type=mixup_type
            )
        
        # Time masking
        features[strong_mask], labels[strong_mask] = time_mask(features[strong_mask], labels[strong_mask], self.config["training"]["net_pooling"], mask_ratios=self.config["training"]["time_mask_ratios"])
        
        mixup_sup_alpha = 1.0
        mixup_usup_alpha = 2.0
        mixup_consistency = 1.0

        # sed student forward
        strong_preds_student, weak_preds_student = self.detect(
            features, self.sed_student
        )

        # supervised loss on strong labels
        loss_strong = self.supervised_loss(
            strong_preds_student[strong_mask], labels[strong_mask]
        )
        # supervised loss on weakly labelled
        loss_weak = self.supervised_loss(weak_preds_student[weak_mask], labels_weak)
        # total supervised loss
        tot_loss_supervised = loss_strong + loss_weak
        
        mixed_input_weak, target_a_weak, target_b_weak, lam_weak = mixup_data_sup(features[weak_mask], labels_weak, mixup_sup_alpha)
        _, output_mixed_weak =  self.detect(mixed_input_weak, self.sed_student)
        loss_func_weak = mixup_criterion(target_a_weak, target_b_weak, lam_weak)
        mixup_weak_class_loss = loss_func_weak(self.supervised_loss, output_mixed_weak)
        total_mixup_loss = mixup_weak_class_loss
            
        
        mixed_input_strong, target_a_strong, target_b_strong, lam_strong = mixup_data_sup(features[strong_mask], labels[strong_mask], mixup_sup_alpha)
        output_mixed_strong, _ = self.detect(mixed_input_strong, self.sed_student)
        loss_func_strong = mixup_criterion(target_a_strong, target_b_strong, lam_strong)
        mixup_strong_class_loss = loss_func_strong(self.supervised_loss, output_mixed_strong)
        total_mixup_loss += mixup_strong_class_loss


        with torch.no_grad():
            strong_preds_teacher, weak_preds_teacher = self.detect(
                features, self.sed_teacher
            )
            loss_strong_teacher = self.supervised_loss(
                strong_preds_teacher[strong_mask], labels[strong_mask]
            )

            loss_weak_teacher = self.supervised_loss(
                weak_preds_teacher[weak_mask], labels_weak
            )
        # we apply consistency between the predictions, use the scheduler for learning rate (to be changed ?)
        weight = (
            self.config["training"]["const_max"]
            * self.scheduler["scheduler"]._get_scaling_factor()
        )

        strong_self_sup_loss = self.selfsup_loss(
            strong_preds_student, strong_preds_teacher.detach()
        )
        weak_self_sup_loss = self.selfsup_loss(
            weak_preds_student, weak_preds_teacher.detach()
        )
        tot_self_loss = (strong_self_sup_loss + weak_self_sup_loss) * weight
        
        mixedup_x, mixedup_target, mixedup_target_weak, lam = mixup_data(features[ul_mask], strong_preds_teacher[ul_mask].detach(), weak_preds_teacher[ul_mask].detach(), mixup_usup_alpha)
        output_mixed_u, output_mixed_u_weak = self.detect(
                mixedup_x, self.sed_teacher
            )
        mixup_consistency_weak_loss = self.selfsup_loss(output_mixed_u_weak, mixedup_target_weak)# / 12
        mixup_consistency_strong_loss = self.selfsup_loss(output_mixed_u, mixedup_target)# / 12
        total_mixup_loss += (mixup_consistency_weak_loss + mixup_consistency_strong_loss)*weight
        tot_loss = tot_loss_supervised + tot_self_loss + total_mixup_loss

        self.log("train/student/loss_strong", loss_strong)
        self.log("train/student/loss_weak", loss_weak)
        self.log("train/teacher/loss_strong", loss_strong_teacher)
        self.log("train/teacher/loss_weak", loss_weak_teacher)
        self.log("train/step", self.scheduler["scheduler"].step_num, prog_bar=True)
        self.log("train/student/tot_self_loss", tot_self_loss, prog_bar=True)
        self.log("train/weight", weight)
        self.log("train/student/tot_supervised", strong_self_sup_loss, prog_bar=True)
        self.log("train/student/weak_self_sup_loss", weak_self_sup_loss)
        self.log("train/student/strong_self_sup_loss", strong_self_sup_loss)
        self.log("train/lr", self.opt.param_groups[-1]["lr"], prog_bar=True)

        return tot_loss

    def on_before_zero_grad(self, *args, **kwargs):
        # update EMA teacher
        self.update_ema(
            self.config["training"]["ema_factor"],
            self.scheduler["scheduler"].step_num,
            self.sed_student,
            self.sed_teacher,
        )

    def validation_step(self, batch, batch_indx):
        """ Apply validation to a batch (step). Used during trainer.fit

        Args:
            batch: torch.Tensor, input batch tensor
            batch_indx: torch.Tensor, 1D tensor of indexes to know which data are present in each batch.
        Returns:
        """

        audio, labels, padded_indxs, filenames = batch

        # prediction for student
        mels = self.mel_spec(audio)
        strong_preds_student, weak_preds_student = self.detect(mels, self.sed_student)
        # prediction for teacher
        strong_preds_teacher, weak_preds_teacher = self.detect(mels, self.sed_teacher)

        # we derive masks for each dataset based on folders of filenames
        weak_mask = (
            torch.tensor(
                [
                    str(Path(x).parent)
                    == str(Path(self.config["data"]["weak_folder"]))
                    for x in filenames
                ]
            )
            .to(audio)
            .bool()
        )
        mask_synth = (
            torch.tensor(
                [
                    str(Path(x).parent)
                    == str(Path(self.config["data"]["synth_val_folder"]))
                    for x in filenames
                ]
            )
            .to(audio)
            .bool()
        )

        if torch.any(weak_mask):
            labels_weak = (torch.sum(labels[weak_mask], -1) >= 1).float()
            loss_weak_student = self.supervised_loss(
                weak_preds_student[weak_mask], labels_weak
            )
            loss_weak_teacher = self.supervised_loss(
                weak_preds_teacher[weak_mask], labels_weak
            )
            self.log("val/weak/student/loss_weak", loss_weak_student)
            self.log("val/weak/teacher/loss_weak", loss_weak_teacher)

            # accumulate f1 score for weak labels
            self.get_weak_student_f1_seg_macro(
                weak_preds_student[weak_mask], labels_weak.int()
            )
            self.get_weak_teacher_f1_seg_macro(
                weak_preds_teacher[weak_mask], labels_weak.int()
            )

        if torch.any(mask_synth):
            loss_strong_student = self.supervised_loss(
                strong_preds_student[mask_synth], labels[mask_synth]
            )
            loss_strong_teacher = self.supervised_loss(
                strong_preds_teacher[mask_synth], labels[mask_synth]
            )

            self.log("val/synth/student/loss_strong", loss_strong_student)
            self.log("val/synth/teacher/loss_strong", loss_strong_teacher)

            filenames_synth = [
                x
                for x in filenames
                if Path(x).parent == Path(self.config["data"]["synth_val_folder"])
            ]

            decoded_student_strong = batched_decode_preds(
                strong_preds_student[mask_synth],
                weak_preds_student[mask_synth],
                filenames_synth,
                self.encoder,
                median_filter=self.config["training"]["median_window"],
                decode_weak=self.config["training"]["decode_weak_valid"],
                thresholds=list(self.val_buffer_student_synth.keys()),
            )

            for th in self.val_buffer_student_synth.keys():
                self.val_buffer_student_synth[th] = self.val_buffer_student_synth[
                    th
                ].append(decoded_student_strong[th], ignore_index=True)

            decoded_teacher_strong = batched_decode_preds(
                strong_preds_teacher[mask_synth],
                weak_preds_teacher[mask_synth],
                filenames_synth,
                self.encoder,
                median_filter=self.config["training"]["median_window"],
                decode_weak=self.config["training"]["decode_weak_valid"],
                thresholds=list(self.val_buffer_teacher_synth.keys()),
            )
            
            for th in self.val_buffer_teacher_synth.keys():
                self.val_buffer_teacher_synth[th] = self.val_buffer_teacher_synth[
                    th
                ].append(decoded_teacher_strong[th], ignore_index=True)

        return

    def validation_epoch_end(self, outputs):
        """ Fonction applied at the end of all the validation steps of the epoch.

        Args:
            outputs: torch.Tensor, the concatenation of everything returned by validation_step.

        Returns:
            torch.Tensor, the objective metric to be used to choose the best model from for example.
        """

        weak_student_f1_macro = self.get_weak_student_f1_seg_macro.compute()
        weak_teacher_f1_macro = self.get_weak_teacher_f1_seg_macro.compute()

        # synth dataset
        intersection_f1_macro_student = compute_per_intersection_macro_f1(
            self.val_buffer_student_synth,
            self.config["data"]["synth_val_tsv"],
            self.config["data"]["synth_val_dur"],
        )

        synth_student_event_macro = log_sedeval_metrics(
            self.val_buffer_student_synth[0.5], self.config["data"]["synth_val_tsv"],
        )[0]

        intersection_f1_macro_teacher = compute_per_intersection_macro_f1(
            self.val_buffer_teacher_synth,
            self.config["data"]["synth_val_tsv"],
            self.config["data"]["synth_val_dur"],
        )

        synth_teacher_event_macro = log_sedeval_metrics(
            self.val_buffer_teacher_synth[0.5], self.config["data"]["synth_val_tsv"],
        )[0]

        obj_metric_synth_type = self.config["training"].get("obj_metric_synth_type")
        if obj_metric_synth_type is None:
            synth_metric = intersection_f1_macro_student
        elif obj_metric_synth_type == "event":
            synth_metric = synth_student_event_macro
        elif obj_metric_synth_type == "intersection":
            synth_metric = intersection_f1_macro_student
        else:
            raise NotImplementedError(
                f"obj_metric_synth_type: {obj_metric_synth_type} not implemented."
            )

        obj_metric = torch.tensor(weak_student_f1_macro.item() + synth_metric)

        self.log("val/obj_metric", obj_metric, prog_bar=True)
        self.log("val/weak/student/macro_F1", weak_student_f1_macro)
        self.log("val/weak/teacher/macro_F1", weak_teacher_f1_macro)
        self.log(
            "val/synth/student/intersection_f1_macro", intersection_f1_macro_student
        )
        self.log(
            "val/synth/teacher/intersection_f1_macro", intersection_f1_macro_teacher
        )
        self.log("val/synth/student/event_f1_macro", synth_student_event_macro)
        self.log("val/synth/teacher/event_f1_macro", synth_teacher_event_macro)

        # free the buffers
        self.val_buffer_student_synth = {
            k: pd.DataFrame() for k in self.config["training"]["val_thresholds"]
        }
        self.val_buffer_teacher_synth = {
            k: pd.DataFrame() for k in self.config["training"]["val_thresholds"]
        }

        self.get_weak_student_f1_seg_macro.reset()
        self.get_weak_teacher_f1_seg_macro.reset()

        return obj_metric

    def on_save_checkpoint(self, checkpoint):
        checkpoint["sed_student"] = self.sed_student.state_dict()
        checkpoint["sed_teacher"] = self.sed_teacher.state_dict()
        return checkpoint

    def test_step(self, batch, batch_indx):
        """ Apply Test to a batch (step), used only when (trainer.test is called)

        Args:
            batch: torch.Tensor, input batch tensor
            batch_indx: torch.Tensor, 1D tensor of indexes to know which data are present in each batch.
        Returns:
        """
        
        audio, labels, padded_indxs, filenames = batch        
        
        # prediction for student
        mels = self.mel_spec(audio)
        strong_preds_student, weak_preds_student = self.detect(mels, self.sed_student)
        # prediction for teacher
        strong_preds_teacher, weak_preds_teacher = self.detect(mels, self.sed_teacher)
        
        weak_preds_teacher = weak_preds_teacher.cpu().data.numpy()
        
        if len(weak_preds_teacher.shape) == 3:
            # average data to have weak labels
            weak_preds_teacher = np.max(weak_preds_teacher, axis=1)
        
        binarization_type = 'global_threshold'
        thresh = 0.5
                                        
        for j, pred_weak_it in enumerate(weak_preds_teacher):
            pred_weak_m = ProbabilityEncoder().binarization(pred_weak_it, binarization_type=binarization_type,threshold=thresh,time_axis=0)
            pred = [','.join(self.encoder.decode_weak(pred_weak_m))]
            if pred[0] != "":
                pred = pd.DataFrame(pred, columns=["event_labels"])
                pred["filename"] = filenames[j].split('/')[-1]
                pred = pred[["filename", "event_labels"]]
                self.prediction_dfs = self.prediction_dfs.append(pred, ignore_index=True)

#         if not self.evaluation:
#             loss_strong_student = self.supervised_loss(strong_preds_student, labels)
#             loss_strong_teacher = self.supervised_loss(strong_preds_teacher, labels)

#             self.log("test/student/loss_strong", loss_strong_student)
#             self.log("test/teacher/loss_strong", loss_strong_teacher)

#         # compute psds
#         decoded_student_strong = batched_decode_preds(
#             strong_preds_student,
#             weak_preds_student,
#             filenames,
#             self.encoder,
#             median_filter=self.config["training"]["median_window"],
#             decode_weak=self.config["training"]["decode_weak_test"],
#             thresholds=list(self.test_psds_buffer_student.keys()),
#         )

#         for th in self.test_psds_buffer_student.keys():
#             self.test_psds_buffer_student[th] = self.test_psds_buffer_student[
#                 th
#             ].append(decoded_student_strong[th], ignore_index=True)

#         decoded_teacher_strong = batched_decode_preds(
#             strong_preds_teacher,
#             weak_preds_teacher,
#             filenames,
#             self.encoder,
#             median_filter=self.config["training"]["median_window"],
#             decode_weak=self.config["training"]["decode_weak_test"],
#             thresholds=list(self.test_psds_buffer_teacher.keys()),
#         )

#         for th in self.test_psds_buffer_teacher.keys():
#             self.test_psds_buffer_teacher[th] = self.test_psds_buffer_teacher[
#                 th
#             ].append(decoded_teacher_strong[th], ignore_index=True)


#         # compute f1 score
#         decoded_student_strong = batched_decode_preds(
#             strong_preds_student,
#             weak_preds_student,
#             filenames,
#             self.encoder,
#             median_filter=self.config["training"]["median_window"],
#             thresholds=[0.5],
#             decode_weak=self.config["training"]["decode_weak_test"],

#         )

#         self.decoded_student_05_buffer = self.decoded_student_05_buffer.append(
#             decoded_student_strong[0.5]
#         )

#         decoded_teacher_strong = batched_decode_preds(
#             strong_preds_teacher,
#             weak_preds_teacher,
#             filenames,
#             self.encoder,
#             median_filter=self.config["training"]["median_window"],
#             thresholds=[0.5],
#             decode_weak=self.config["training"]["decode_weak_test"],
#         )

#         self.decoded_teacher_05_buffer = self.decoded_teacher_05_buffer.append(
#             decoded_teacher_strong[0.5]
#         )

    def on_test_epoch_end(self):
        self.prediction_dfs.to_csv('unlabel_in_domain_pseudo_weak.tsv', index=False, sep="\t")
        # pub eval dataset
#         try:
#             log_dir = self.logger.log_dir
#         except Exception as e:
#             log_dir = self.config["log_dir"]
#         save_dir = os.path.join(log_dir, "metrics_test")

#         # only save the predictions
#         save_dir_student = os.path.join(save_dir, "student")
#         os.makedirs(save_dir_student, exist_ok=True)
#         self.decoded_student_05_buffer.to_csv(
#             os.path.join(save_dir_student, f"predictions_05_student.tsv"),
#             sep="\t",
#             index=False
#         )

#         for k in self.test_psds_buffer_student.keys():
#             self.test_psds_buffer_student[k].to_csv(
#                 os.path.join(save_dir_student, f"predictions_th_{k:.2f}.tsv"),
#                 sep="\t",
#                 index=False,
#             )
#         print(f"\nPredictions for student saved in: {save_dir_student}")

#         save_dir_teacher = os.path.join(save_dir, "teacher")
#         os.makedirs(save_dir_teacher, exist_ok=True)

#         self.decoded_teacher_05_buffer.to_csv(
#             os.path.join(save_dir_teacher, f"predictions_05_teacher.tsv"),
#             sep="\t",
#             index=False
#         )

#         for k in self.test_psds_buffer_student.keys():
#             self.test_psds_buffer_student[k].to_csv(
#                 os.path.join(save_dir_teacher, f"predictions_th_{k:.2f}.tsv"),
#                 sep="\t",
#                 index=False,
#             )
#         print(f"\nPredictions for teacher saved in: {save_dir_teacher}")

        


    def configure_optimizers(self):
        return [self.opt], [self.scheduler]

    def train_dataloader(self):

        self.train_loader = torch.utils.data.DataLoader(
            self.train_data,
            batch_sampler=self.train_sampler,
            num_workers=self.num_workers,
        )

        return self.train_loader

    def val_dataloader(self):
        self.val_loader = torch.utils.data.DataLoader(
            self.valid_data,
            batch_size=self.config["training"]["batch_size_val"],
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )
        return self.val_loader

    def test_dataloader(self):
        self.test_loader = torch.utils.data.DataLoader(
            self.test_data,
            batch_size=self.config["training"]["batch_size_val"],
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )
        return self.test_loader
