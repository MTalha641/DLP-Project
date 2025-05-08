#!/usr/bin/env python3
""" Training for E2E-Spot """
import os
import argparse  # Concept 20: Hyperparameter Tuning (Arguments define hyperparameters)
from contextlib import nullcontext
import random
import numpy as np
from tabulate import tabulate
import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import (
    ChainedScheduler, LinearLR, CosineAnnealingLR) # Concept 7: Learning Rate Scheduling
from torch.utils.data import DataLoader
import torchvision
import timm # Used for loading pre-trained CNN models
from tqdm import tqdm
import traceback

from model.common import step, BaseRGBModel
from model.shift import make_temporal_shift # Concept 4: GSM (Implementation likely here)
from model.modules import * # Concept 5: GRU Decoder (GRUPrediction likely defined here)
# Ensure ActionSpotVideoDataset is imported correctly
from dataset.frame import ActionSpotDataset, ActionSpotVideoDataset, DEFAULT_PAD_LEN # Concept 12: Data Augmentation (Mixup), Concept 13: Label Dilation (Applied here)
from util.eval import process_frame_predictions # Concept 18: Post-processing (NMS - implied)
from util.io import load_json, store_json, store_gz_json, clear_files
from util.dataset import DATASETS, load_classes
from util.score import compute_mAPs # Concept 17: Evaluation Metrics (mAP)

EPOCH_NUM_FRAMES = 500000
BASE_NUM_WORKERS = 4
BASE_NUM_VAL_EPOCHS = 20
INFERENCE_BATCH_SIZE = 1
MAX_GRU_HIDDEN_DIM = 768 # Prevent the GRU params from going too big

def get_args():
    # Concept 20: Hyperparameter Tuning (This function defines all tunable hyperparameters)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, choices=DATASETS)
    parser.add_argument('frame_dir', type=str, help='Path to extracted frames')
    parser.add_argument('--modality', type=str, choices=['rgb', 'bw', 'flow'], default='rgb')
    parser.add_argument(
        '-m', '--feature_arch', type=str, required=True, choices=[
            'rn18', 'rn18_tsm', 'rn18_gsm', 'rn50', 'rn50_tsm', 'rn50_gsm',
            'rny002', 'rny002_tsm', 'rny002_gsm', 'rny008', 'rny008_tsm', 'rny008_gsm',
            'convnextt', 'convnextt_tsm', 'convnextt_gsm'
        ], help='CNN architecture for feature extraction') # Determines CNN Encoder (#2) and GSM usage (#4)
    parser.add_argument(
        '-t', '--temporal_arch', type=str, default='gru',
        choices=['', 'gru', 'deeper_gru', 'mstcn', 'asformer'],
        help='Spotting architecture, after spatial pooling') # Determines Decoder (#5)
    parser.add_argument('--clip_len', type=int, default=100)
    parser.add_argument('--crop_dim', type=int, default=224)
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('-ag', '--acc_grad_iter', type=int, default=1, help='Use gradient accumulation') # Concept 10: Gradient Accumulation
    parser.add_argument('--warm_up_epochs', type=int, default=3) # Concept 7: Learning Rate Scheduling (Warm-up part)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001) # Concept 7: Learning Rate Scheduling (Base LR)
    parser.add_argument('-s', '--save_dir', type=str, required=True, help='Dir to save checkpoints and predictions') # Concept 11: Best Model Saving
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint in <save_dir>')
    parser.add_argument('--start_val_epoch', type=int) # Related to #11 (When to start checking for best model via mAP)
    parser.add_argument('--criterion', choices=['map', 'loss'], default='map') # Related to #11 (Criterion for choosing best model)
    parser.add_argument('--dilate_len', type=int, default=0, help='Label dilation when training') # Concept 13: Label Dilation
    parser.add_argument('--mixup', type=bool, default=True) # Concept 12: Data Augmentation (Mixup)
    parser.add_argument('-j', '--num_workers', type=int, help='Base number of dataloader workers (overrides default)')
    parser.add_argument('--fg_upsample', type=float) # Concept 14: Handling Class Imbalance (Upsampling part)
    parser.add_argument('-mgpu', '--gpu_parallel', action='store_true') # Concept 16: Multi-GPU Training
    return parser.parse_args()

def worker_init_fn(worker_id):
    """Initialize random seed for each worker."""
    random.seed(worker_id)
    np.random.seed(worker_id)

# Concept 1: Encoder-Decoder Architecture (Overall class structure)
class E2EModel(BaseRGBModel):
    class Impl(nn.Module):
        def __init__(self, num_classes, feature_arch, temporal_arch, clip_len, modality):
            super().__init__()
            is_rgb = modality == 'rgb'
            in_channels = {'flow': 2, 'bw': 1, 'rgb': 3}[modality]

            # === CNN ENCODER SETUP (Feature Extractor Backbone) ===
            # Concept 2: CNN Encoder (Feature Extractor)
            # Concept 19: Batch Normalization (Implicitly used inside these CNNs)
            # This section loads a pre-trained CNN model (e.g., ResNet, RegNetY, ConvNeXt)
            # which will act as the primary feature extractor for individual frames.
            # The goal is to convert raw pixel data into meaningful feature vectors.
            if feature_arch.startswith(('rn18', 'rn50')):
                # Example: Using a ResNet model
                resnet_name = feature_arch.split('_')[0].replace('rn', 'resnet')
                # Concept 3: Transfer Learning (Loading pre-trained weights)
                features = getattr(torchvision.models, resnet_name)(weights='DEFAULT' if is_rgb else None)
                feat_dim = features.fc.in_features # Get the dimension of features before the original classifier
                features.fc = nn.Identity() # Remove the original classification layer
                if not is_rgb:
                    # Reinitialize conv1 for different input channels if not using RGB
                    features.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            elif feature_arch.startswith(('rny002', 'rny008')):
                # Example: Using a RegNetY model (as per your config 'rny002_gsm')
                model_name_map = {
                    'rny002': 'regnety_002', 'rny008': 'regnety_008',
                }
                timm_model_name = model_name_map[feature_arch.rsplit('_', 1)[0]]
                # Concept 3: Transfer Learning (Loading pre-trained weights)
                features = timm.create_model(timm_model_name, pretrained=is_rgb)
                feat_dim = features.head.fc.in_features # Get feature dimension
                features.head.fc = nn.Identity() # Remove original classification layer
                if not is_rgb:
                    features.stem.conv = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            elif 'convnextt' in feature_arch:
                # Example: Using a ConvNeXt model
                # Concept 3: Transfer Learning (Loading pre-trained weights)
                features = timm.create_model('convnext_tiny', pretrained=is_rgb)
                feat_dim = features.head.fc.in_features
                features.head.fc = nn.Identity()
                if not is_rgb:
                    features.stem[0] = nn.Conv2d(in_channels, 96, kernel_size=4, stride=4)
            else:
                raise NotImplementedError(feature_arch)

            # === GSM (Gate Shift Module) INTEGRATION (Part of Encoder) ===
            # Concept 4: GSM (Gate Shift Module)
            # If the feature_arch name ends with '_gsm' (or '_tsm'), this section
            # modifies the CNN backbone to include temporal shifting capabilities.
            # GSM helps the 2D CNN capture short-term motion cues between adjacent frames
            # by shifting parts of the feature maps along the time dimension, controlled by gates.
            self._require_clip_len = -1
            if feature_arch.endswith('_tsm') or feature_arch.endswith('_gsm'):
                make_temporal_shift(features, clip_len, is_gsm=feature_arch.endswith('_gsm'))
                self._require_clip_len = clip_len

            self._features = features # This is the complete CNN+GSM Feature Extractor / Encoder
            self._feat_dim = feat_dim # Dimension of the features extracted per frame

            # === GRU DECODER SETUP (Temporal Sequence Modeler) ===
            # Concept 5: GRU Decoder (Sequence Modeler)
            # This section sets up the GRU (Gated Recurrent Unit) based decoder.
            # The GRU will take the sequence of frame features from the encoder
            # and model their temporal relationships to predict the action.
            if 'gru' in temporal_arch: # Your config uses 'gru'
                hidden_dim = min(feat_dim, MAX_GRU_HIDDEN_DIM) # Hidden dimension for GRU
                if hidden_dim != feat_dim:
                     print(f'Clamped GRU hidden dim: {feat_dim} -> {hidden_dim}')
                if temporal_arch in ('gru', 'deeper_gru'):
                    # GRUPrediction is a module (likely in model/modules.py) that wraps nn.GRU
                    # and adds a final classification layer.
                    self._pred_fine = GRUPrediction(feat_dim, num_classes, hidden_dim, num_layers=3 if temporal_arch.startswith('d') else 1)
                else:
                    raise NotImplementedError(temporal_arch)
            elif temporal_arch == 'mstcn':
                self._pred_fine = TCNPrediction(feat_dim, num_classes, 3)
            elif temporal_arch == 'asformer':
                self._pred_fine = ASFormerPrediction(feat_dim, num_classes, 3)
            elif temporal_arch == '': # No explicit temporal model, just a fully connected layer
                self._pred_fine = FCPrediction(feat_dim, num_classes)
            else:
                raise NotImplementedError(temporal_arch)

        def forward(self, x):
            # x is the input video clip: (batch_size, true_clip_len, channels, height, width)
            batch_size, true_clip_len, channels, height, width = x.shape
            clip_len = true_clip_len # Effective clip length being processed

            # Pad if the input clip is shorter than what the GSM module might require
            if self._require_clip_len > 0:
                assert true_clip_len <= self._require_clip_len, f'Expected {self._require_clip_len}, got {true_clip_len}'
                if true_clip_len < self._require_clip_len:
                    # Pad with zeros at the end of the temporal dimension
                    x = F.pad(x, (0,) * 7 + (self._require_clip_len - true_clip_len,))
                    clip_len = self._require_clip_len
            
            # === ENCODER (CNN+GSM) FORWARD PASS ===
            # Reshape to process all frames in the batch as individual images:
            # (batch_size * clip_len, channels, height, width)
            # Then pass through the CNN+GSM feature extractor (_features).
            im_feat = self._features(x.view(-1, channels, height, width))
            # Reshape back to sequence format: (batch_size, clip_len, feat_dim)
            im_feat = im_feat.reshape(batch_size, clip_len, self._feat_dim)

            # If padding was applied, remove the features for the padded frames
            if true_clip_len != clip_len:
                im_feat = im_feat[:, :true_clip_len, :]

            # === DECODER (GRU) FORWARD PASS ===
            # Pass the sequence of frame features through the GRU decoder (_pred_fine).
            # This will output action scores for each frame in the clip.
            return self._pred_fine(im_feat)

        def print_stats(self):
            print('Model params:', sum(p.numel() for p in self.parameters()))
            print('   CNN features:', sum(p.numel() for p in self._features.parameters()))
            print('   Temporal:', sum(p.numel() for p in self._pred_fine.parameters()))

    def __init__(self, num_classes, feature_arch, temporal_arch, clip_len, modality, device='cuda', multi_gpu=False):
        self.device = device
        self._multi_gpu = multi_gpu
        self._model = E2EModel.Impl(num_classes, feature_arch, temporal_arch, clip_len, modality)
        self._model.print_stats()
        # Concept 16: Multi-GPU Training (DataParallel wrapper)
        if multi_gpu:
            self._model = nn.DataParallel(self._model)
        self._model.to(device)
        self._num_classes = num_classes # Expected total classes including background

    # === OPTIMIZER SETUP (AdamW) ===
    # Concept 6: Optimization (AdamW)
    # Concept 15: Regularization (Implicit via AdamW's weight decay)
    # AdamW: An optimizer that adapts learning rates for each parameter and correctly handles weight decay.
    # It's generally a good choice for deep learning models.
    def get_optimizer(self, opt_args):
        optimizer = torch.optim.AdamW(self._get_params(), **opt_args)
        
        # === MIXED PRECISION TRAINING (GradScaler) ===
        # Concept 8: Mixed Precision Training (GradScaler part)
        # GradScaler is used for mixed precision training. It helps prevent underflow issues
        # with small gradients when using half-precision (FP16) computations, by scaling losses up and down.
        # This speeds up training and reduces memory usage on compatible GPUs.
        scaler = torch.amp.GradScaler('cuda') if self.device == 'cuda' else None
        return optimizer, scaler

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None, acc_grad_iter=1, fg_weight=5):
        # This function defines one epoch of training or validation.
        is_train = optimizer is not None # If optimizer is provided, it's a training epoch
        if is_train:
            optimizer.zero_grad() # Clear previous gradients before new computations
            self._model.train()   # Set the model to training mode (enables dropout, batchnorm updates, etc.)
        else:
            self._model.eval()    # Set the model to evaluation mode (disables dropout, uses running stats for batchnorm)

        # === LOSS FUNCTION SETUP (Cross-Entropy with Class Weighting) ===
        # Concept 9: Loss Function (Cross-Entropy)
        # Concept 14: Handling Class Imbalance (Loss Weighting part)
        ce_kwargs = {}
        # weights: Used to give more importance to foreground (action) classes than the background class.
        # This can help if there's a class imbalance (many background frames, few action frames).
        weights = [1.0] + [float(fg_weight)] * (self._num_classes - 1) # Background class weight 1.0, others fg_weight
        if len(weights) != self._num_classes:
             print(f"Warning: Weight list size {len(weights)} does not match expected num_classes {self._num_classes}. Check class count.")
             weights = [1.0] * self._num_classes # Fallback
        if fg_weight != 1:
             ce_kwargs['weight'] = torch.tensor(weights, dtype=torch.float32).to(self.device)

        epoch_loss = 0.
        # torch.no_grad() disables gradient calculations during validation, saving memory and computation.
        # nullcontext() does nothing, so gradients are computed during training.
        with torch.no_grad() if not is_train else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = loader.dataset.load_frame_gpu(batch, self.device)
                if frame is None: continue
                frame = frame.float() # Ensure input is float32
                label = batch['label'].to(self.device)

                if len(label.shape) == 2: label_flat = label.flatten()
                elif len(label.shape) == 3: label_flat = label.view(-1, self._num_classes)
                else: raise ValueError(f"Unexpected label shape: {label.shape}")

                # === MIXED PRECISION TRAINING (autocast) ===
                # Concept 8: Mixed Precision Training (autocast part)
                # torch.amp.autocast automatically casts operations to FP16 (half-precision)
                # where appropriate, and FP32 (full-precision) for sensitive operations.
                # This speeds up computation and reduces memory, used in conjunction with GradScaler.
                with torch.amp.autocast(device_type='cuda', enabled=scaler is not None):
                    pred = self._model(frame) # Forward pass: Get model predictions

                    if pred.shape[-1] != self._num_classes:
                        raise RuntimeError(f"Model output last dimension size mismatch. Expected {self._num_classes}, got {pred.shape[-1]}. Output shape: {pred.shape}")

                    pred_flat = pred.view(-1, self._num_classes) # Reshape for loss calculation

                    # Calculate loss using Cross-Entropy (Concept 9)
                    if label_flat.dtype == torch.int64:
                         loss = F.cross_entropy(pred_flat, label_flat, **ce_kwargs)
                    elif label_flat.dtype == torch.float32: # Handles mixed labels if Mixup (#12) is used
                         loss = F.cross_entropy(pred_flat, label_flat, **ce_kwargs)
                    else:
                         raise TypeError(f"Unexpected label_flat dtype: {label_flat.dtype}")

                if is_train:
                    # === GRADIENT ACCUMULATION & OPTIMIZATION STEP ===
                    # Concept 10: Gradient Accumulation
                    # Gradients are accumulated over `acc_grad_iter` mini-batches before an optimizer step.
                    # This allows for a larger effective batch size than what might fit in GPU memory.
                    # `scaler.scale(loss).backward()`: Scales loss for mixed precision, then computes gradients.
                    # `scaler.step(optimizer)`: Unscales gradients and performs optimizer step.
                    # `scaler.update()`: Updates the scaler for the next iteration.
                    # `lr_scheduler.step()`: Updates the learning rate according to the schedule (Concept 7).
                    step(optimizer, scaler, loss / acc_grad_iter,
                         lr_scheduler=lr_scheduler, # Applies Learning Rate Schedule (#7)
                         backward_only=(batch_idx + 1) % acc_grad_iter != 0)

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)

    def predict(self, seq, use_amp=True):
        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)

        initial_seq_shape = seq.shape # This is a torch.Size object
        processed_seq_for_model = seq

        # These will store characteristics of the input for potential multi-crop averaging later
        self.B_for_avg_logic = 1
        self.num_crops_for_avg_logic = 1
        self.perform_multicrop_avg_logic = False # Flag to indicate if averaging should be done on output

        # Check dimensions using len()
        if len(initial_seq_shape) == 4: # Input is (T, C, H, W)
            processed_seq_for_model = seq.unsqueeze(0) # Shape becomes (1, T, C, H, W)
            self.B_for_avg_logic = 1
            self.num_crops_for_avg_logic = 1
            self.perform_multicrop_avg_logic = False
        elif len(initial_seq_shape) == 5:
            # Assuming this is (B_eval, T, C, H, W) for validation/testing without dataset multicrop
            self.B_for_avg_logic = initial_seq_shape[0] # This is B_eval
            self.num_crops_for_avg_logic = 1
            self.perform_multicrop_avg_logic = False
            # No change to processed_seq_for_model, it's already (B_eval, T, C, H, W)
        elif len(initial_seq_shape) == 6: # Input is (B_eval, num_crops, T, C, H, W)
            B, num_crops, T_dim, C_dim, H_dim, W_dim = initial_seq_shape # Unpack 6D shape
            processed_seq_for_model = seq.view(B * num_crops, T_dim, C_dim, H_dim, W_dim)
            self.B_for_avg_logic = B
            self.num_crops_for_avg_logic = num_crops
            self.perform_multicrop_avg_logic = True
        else:
            raise ValueError(f"Unexpected input sequence ndim for predict: {len(initial_seq_shape)}. Shape: {initial_seq_shape}")

        if processed_seq_for_model.device != self.device:
            processed_seq_for_model = processed_seq_for_model.to(self.device)

        self._model.eval()
        with torch.no_grad():
            # Updated autocast context manager
            with torch.amp.autocast(device_type='cuda', enabled=use_amp) if use_amp else nullcontext():
                pred = self._model(processed_seq_for_model) # Pass the correctly shaped sequence

            if isinstance(pred, tuple):
                pred = pred[0]

            if pred.ndim != 3 or pred.shape[-1] != self._num_classes: # _num_classes already includes background
                 raise RuntimeError(f"Unexpected model output shape in predict. Expected (B', T, {self._num_classes}), got {pred.shape}.")

            pred = torch.softmax(pred, axis=2)

            if self.perform_multicrop_avg_logic and self.num_crops_for_avg_logic > 1:
                 T_pred = pred.shape[1] # Get temporal dimension from current pred
                 num_actual_classes_output = pred.shape[2] # Get class dimension

                 pred = pred.view(self.B_for_avg_logic, self.num_crops_for_avg_logic, T_pred, num_actual_classes_output)
                 pred = torch.mean(pred, dim=1) # Average over the num_crops dimension

            pred_cls = torch.argmax(pred, axis=2)
            return pred_cls.cpu().numpy(), pred.cpu().numpy()


def evaluate(model, dataset, split, classes, save_pred, calc_stats=True, save_scores=True):
    num_actual_classes = len(classes)
    pred_dict = {}
    # Iterate using the dataset.videos property which yields processed keys
    for video_key, num_strided_frames, _ in dataset.videos:
        # Ensure dictionary uses correct number of classes (model._num_classes includes background)
        # Use video_key (which is already processed) for the dictionary
        pred_dict[video_key] = (
            np.zeros((num_strided_frames, model._num_classes), np.float32), # Scores shape (T, num_classes_incl_bg)
            np.zeros(num_strided_frames, np.int32)) # Support shape (T,)

    batch_size = 1 if dataset.augment else INFERENCE_BATCH_SIZE
    eval_num_workers = BASE_NUM_WORKERS * 2
    eval_prefetch_factor = 2 if eval_num_workers > 0 else None
    eval_loader = DataLoader(dataset, shuffle=False, batch_size=batch_size, pin_memory=True,
                           num_workers=eval_num_workers, prefetch_factor=eval_prefetch_factor,
                           worker_init_fn=worker_init_fn)

    for clip in tqdm(eval_loader):
        if clip is None or 'frame' not in clip or 'video' not in clip or 'start' not in clip: continue
        if isinstance(clip['frame'], (list, np.ndarray)): clip['frame'] = torch.as_tensor(clip['frame'])
        if clip['frame'] is None or clip['frame'].nelement() == 0: continue

        # Concept 8: Mixed Precision Training (autocast usage during evaluation/prediction)
        with torch.amp.autocast(device_type='cuda', enabled=True):
            _, batch_pred_scores = model.predict(clip['frame'])

        for i in range(batch_pred_scores.shape[0]):
            # The key from the dataloader should already be the processed one
            video_key = clip['video'][i]
            if video_key not in pred_dict:
                # This might happen if dataset.videos and the dataloader somehow yield different keys
                print(f"Warning: Video key '{video_key}' from DataLoader not found in initial pred_dict. Skipping.")
                continue

            # Check prediction shape consistency
            if batch_pred_scores[i].ndim != 2 or batch_pred_scores[i].shape[-1] != model._num_classes:
                print(f"Warning: Skipping prediction for {video_key} due to unexpected score shape {batch_pred_scores[i].shape}. Expected (T, {model._num_classes}).")
                continue

            scores, support = pred_dict[video_key]
            original_clip_start_frame = clip['start'][i].item()
            # Use max(1, ...) for stride to avoid division by zero if stride is 0
            safe_stride = max(1, dataset._stride)
            strided_start_idx = max(0, original_clip_start_frame) // safe_stride
            num_frames_in_clip_pred = batch_pred_scores[i].shape[0]
            strided_end_idx = strided_start_idx + num_frames_in_clip_pred

            # Trim predictions or skip if they fall outside the expected range
            if strided_start_idx >= scores.shape[0]: continue # Clip starts after video ends
            overlap = scores.shape[0] - strided_start_idx
            if overlap <= 0: continue

            # Adjust end index and trim prediction scores if needed
            if strided_end_idx > scores.shape[0]:
                 num_frames_to_use = overlap
                 strided_end_idx = scores.shape[0]
            else:
                 num_frames_to_use = num_frames_in_clip_pred

            # Ensure shapes match before adding
            if scores[strided_start_idx:strided_end_idx, :].shape[0] == batch_pred_scores[i][:num_frames_to_use, :].shape[0]:
                 scores[strided_start_idx:strided_end_idx, :] += batch_pred_scores[i][:num_frames_to_use, :]
                 support[strided_start_idx:strided_end_idx] += 1
            else:
                 print(f"Warning: Shape mismatch during score accumulation for {video_key}.")
                 print(f"  Target slice shape: {scores[strided_start_idx:strided_end_idx, :].shape}")
                 print(f"  Source slice shape: {batch_pred_scores[i][:num_frames_to_use, :].shape}")
                 # Decide how to handle: skip, pad, or adjust indices further? Skipping for now.
                 continue


    # Average scores where multiple predictions overlapped
    for video_key in pred_dict:
         scores, support = pred_dict[video_key]
         zero_support_mask = (support == 0)
         support[zero_support_mask] = 1 # Avoid division by zero
         scores = scores / support[:, None]
         pred_dict[video_key] = (scores, support) # Store averaged scores

    # Convert dict to only contain averaged scores needed by process_frame_predictions
    averaged_scores_dict = {video_key: scores for video_key, (scores, support) in pred_dict.items()}

    # Concept 18: Post-processing (Implied - NMS, thresholding likely happen inside this function)
    err, f1, pred_events, pred_events_high_recall, processed_pred_scores = \
        process_frame_predictions(dataset, classes, averaged_scores_dict)

    avg_mAP = None
    if calc_stats:
        print(f'\n=== Results on {split} (w/o NMS) ===')
        frame_error = err.get() * 100 if hasattr(err, 'get') else float('nan')
        print(f'Error (frame-level): {frame_error:0.2f}\n')

        def get_f1_tab_row(str_k):
            k = classes.get(str_k) if isinstance(classes, dict) else None
            f1_score = f1.get(k) * 100 if hasattr(f1, 'get') and k is not None else float('nan')
            tp, fp, fn = f1.tp_fp_fn(k) if hasattr(f1, 'tp_fp_fn') and k is not None else (0,0,0)
            return [str_k, f1_score, tp, fp, fn]

        rows = []
        if hasattr(f1, '_results') and 'any' in f1._results: rows.append(get_f1_tab_row('any'))
        class_names = sorted([k for k in classes.keys() if k != 'any']) if isinstance(classes, dict) else []
        for c in class_names: rows.append(get_f1_tab_row(c))

        print(tabulate(rows, headers=['Exact frame', 'F1', 'TP', 'FP', 'FN'], floatfmt='0.2f'))
        print()

        gt_labels = dataset.labels if hasattr(dataset, 'labels') else []
        # Ensure pred_events_high_recall is a list (compute_mAPs expects list of dicts)
        if isinstance(pred_events_high_recall, dict): # Check if it's dict {video_key: events}
            # Convert dict to list of dicts with 'video' key if necessary
            pred_events_list = [{'video': k, **v} if isinstance(v, dict) else {'video':k, 'events':v} for k, v in pred_events_high_recall.items()]
            # Example conversion assuming value is list of event dicts:
            # pred_events_list = [{'video': k, 'events': v} for k, v in pred_events_high_recall.items()]
        elif isinstance(pred_events_high_recall, list):
             pred_events_list = pred_events_high_recall # Already a list
        else:
             print("Warning: pred_events_high_recall is neither list nor dict. Setting to empty list.")
             pred_events_list = []

        if gt_labels and pred_events_list:
             try:
                  # Calls the function to compute mAP score
                  mAPs, _ = compute_mAPs(gt_labels, pred_events_list)
                  valid_mAPs = [m for m in mAPs if not np.isnan(m)]
                  avg_mAP = np.mean(valid_mAPs[1:]) if len(valid_mAPs) > 1 else (valid_mAPs[0] if valid_mAPs else 0.0)
                  print(f'\nValidation mAP: {avg_mAP:.4f}')
             except Exception as map_err:
                  print(f"Error during mAP calculation: {map_err}")
                  avg_mAP = 0.0
        else:
             print("Could not calculate mAP: Missing ground truth labels or processed predictions.")
             avg_mAP = 0.0


    if save_pred is not None:
        os.makedirs(os.path.dirname(save_pred), exist_ok=True)
        # Ensure variables are lists or dicts before saving
        if isinstance(pred_events, (list, dict)): store_json(save_pred + '.json', pred_events)
        if isinstance(pred_events_high_recall, (list, dict)): store_gz_json(save_pred + '.recall.json.gz', pred_events_high_recall)
        if save_scores and isinstance(processed_pred_scores, (list, dict)): store_gz_json(save_pred + '.score.json.gz', processed_pred_scores)

    return avg_mAP


def get_last_epoch(save_dir):
    if not os.path.exists(save_dir): return -1
    max_epoch = -1
    checkpoint_files = [f for f in os.listdir(save_dir) if f.startswith('checkpoint_') and f.endswith('.pt')]
    for file_name in checkpoint_files:
            try:
                epoch = int(os.path.splitext(file_name)[0].split('_')[-1])
                if epoch > max_epoch: max_epoch = epoch
            except (ValueError, IndexError): pass
    return max_epoch

def get_best_epoch_and_history(save_dir, criterion):
    loss_history_path = os.path.join(save_dir, 'loss.json')
    if not os.path.exists(loss_history_path): return [], None, 0.0 if criterion == 'map' else float('inf')
    try: data = load_json(loss_history_path)
    except Exception as e: print(f"Error loading loss history: {e}"); return [], None, 0.0 if criterion == 'map' else float('inf')
    if not isinstance(data, list) or not data: return [], None, 0.0 if criterion == 'map' else float('inf')

    best_epoch, best_criterion_value = None, 0.0 if criterion == 'map' else float('inf')
    key = 'val_mAP' if criterion == 'map' else 'val'
    valid_entries = [x for x in data if isinstance(x, dict) and key in x and x[key] is not None]

    if valid_entries:
        if criterion == 'map': best_entry = max(valid_entries, key=lambda x: x[key])
        else: best_entry = min(valid_entries, key=lambda x: x[key])
        best_epoch = best_entry.get('epoch')
        best_criterion_value = best_entry[key]

    return data, best_epoch, best_criterion_value

def store_config(file_path, args, num_epochs, classes):
    num_actual_classes = len(classes) if isinstance(classes, dict) else 0
    config = {k: getattr(args, k, None) for k in vars(args)}
    config.update({
        'num_classes': num_actual_classes, # Store actual number of classes (excluding background)
        'num_epochs': num_epochs,
        'epoch_num_frames': EPOCH_NUM_FRAMES,
        'num_workers': get_num_train_workers(args) # Store calculated workers
    })
    try: store_json(file_path, config, pretty=True)
    except Exception as e: print(f"Warning: Failed to store config to {file_path}: {e}")

def get_num_train_workers(args):
    if args.num_workers is not None and args.num_workers >= 0: return args.num_workers
    n = BASE_NUM_WORKERS * 2
    cpu_count = os.cpu_count()
    return min(cpu_count, n) if cpu_count is not None else n

# === LEARNING RATE SCHEDULER SETUP (Warm-up + Cosine Annealing) ===
# Concept 7: Learning Rate Scheduling
# This function sets up the learning rate scheduler, which adjusts the learning rate during training.
# Warm-up: Gradually increases LR from a small value at the beginning of training to stabilize learning.
# Cosine Annealing: After warm-up, LR decreases following a cosine curve, helping to fine-tune the model.
def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    cosine_epochs = max(0, args.num_epochs - args.warm_up_epochs)
    print(f'Using Linear Warmup ({args.warm_up_epochs}) + Cosine Annealing LR ({cosine_epochs})')
    schedulers = []
    # Add Linear Warm-up phase if warm_up_epochs > 0
    if args.warm_up_epochs > 0 and num_steps_per_epoch > 0:
         schedulers.append(LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=args.warm_up_epochs * num_steps_per_epoch))
    # Add Cosine Annealing phase if cosine_epochs > 0
    if cosine_epochs > 0 and num_steps_per_epoch > 0:
        schedulers.append(CosineAnnealingLR(optimizer, num_steps_per_epoch * cosine_epochs))

    # Chain the schedulers if both warm-up and cosine annealing are used
    if len(schedulers) > 1: return args.num_epochs, ChainedScheduler(schedulers)
    elif schedulers: return args.num_epochs, schedulers[0]
    else: print("Warning: No LR schedulers created."); return args.num_epochs, None

def get_datasets(args):
    classes = load_classes(os.path.join('data', args.dataset, 'class.txt'))
    dataset_len = EPOCH_NUM_FRAMES // args.clip_len
    # Dictionary to pass arguments to the dataset class constructor
    dataset_kwargs = {'crop_dim': args.crop_dim}
    
    # Concept 13: Label Dilation (Applied within dataset if dilate_len > 0)
    if args.dilate_len > 0:
        dataset_kwargs['dilate_len'] = args.dilate_len
    
    # Concept 12: Data Augmentation (Mixup is enabled/disabled here for training dataset)
    # Concept 15: Regularization (Mixup acts as a regularizer)
    dataset_kwargs['mixup'] = args.mixup
    
    # Concept 14: Handling Class Imbalance (Foreground Upsampling applied within dataset)
    if args.fg_upsample is not None:
        assert args.fg_upsample > 0, "fg_upsample must be positive"
        dataset_kwargs['fg_upsample'] = args.fg_upsample

    print('Dataset size:', dataset_len)
    # Instantiate training dataset
    train_data = ActionSpotDataset(
        classes, os.path.join('data', args.dataset, 'train.json'),
        args.frame_dir, args.modality, args.clip_len, dataset_len,
        is_eval=False, **dataset_kwargs)
    train_data.print_info()
    
    # Instantiate validation dataset (for loss calculation during training)
    # Note: Augmentations like Mixup are typically disabled for validation loss dataset
    val_dataset_kwargs = dataset_kwargs.copy()
    val_dataset_kwargs['mixup'] = False # Disable mixup for validation loss calculation
    val_data = ActionSpotDataset(
        classes, os.path.join('data', args.dataset, 'val.json'),
        args.frame_dir, args.modality, args.clip_len, dataset_len // 4,
        is_eval=True, **val_dataset_kwargs)
    val_data.print_info()

    val_data_frames = None
    # Instantiate validation dataset specifically for mAP evaluation (if criterion is mAP)
    # This dataset might load frames differently (e.g., ActionSpotVideoDataset vs ActionSpotDataset)
    if args.criterion == 'map':
        val_data_frames = ActionSpotVideoDataset(
            classes, os.path.join('data', args.dataset, 'val.json'),
            args.frame_dir, args.modality, args.clip_len,
            crop_dim=args.crop_dim, overlap_len=0, pad_len=DEFAULT_PAD_LEN)
        val_data_frames.print_info()

    return classes, train_data, val_data, val_data_frames

def load_from_save(args, model, optimizer, scaler, lr_scheduler):
    assert args.save_dir is not None
    last_completed_epoch = get_last_epoch(args.save_dir)
    if last_completed_epoch < 0:
        print(f"No previous checkpoints found in {args.save_dir}. Starting training from epoch 0.")
        return -1, [], None, 0.0 if args.criterion == 'map' else float('inf')

    print(f'Loading model from epoch {last_completed_epoch}')
    checkpoint_path = os.path.join(args.save_dir, f'checkpoint_{last_completed_epoch:03d}.pt')
    if not os.path.exists(checkpoint_path):
         print(f"Error: Checkpoint file not found: {checkpoint_path}. Starting from epoch 0.")
         return -1, [], None, 0.0 if args.criterion == 'map' else float('inf')
    try:
        # Load model state dict with map_location to handle potential device mismatches
        model.load(torch.load(checkpoint_path, map_location=model.device))
        print(f"Successfully loaded model state from {checkpoint_path}")
    except Exception as e:
         print(f"Error loading model state from {checkpoint_path}: {e}. Starting from epoch 0.")
         traceback.print_exc() # Print traceback for loading error
         return -1, [], None, 0.0 if args.criterion == 'map' else float('inf')

    losses, best_epoch, best_criterion_value = get_best_epoch_and_history(args.save_dir, args.criterion)

    if args.resume:
        opt_path = os.path.join(args.save_dir, f'optim_{last_completed_epoch:03d}.pt')
        if os.path.exists(opt_path):
            try:
                # Load optimizer state dict with map_location
                opt_data = torch.load(opt_path, map_location=model.device)
                if 'optimizer_state_dict' in opt_data: optimizer.load_state_dict(opt_data['optimizer_state_dict']); print("Loaded optimizer state.")
                if scaler and 'scaler_state_dict' in opt_data: scaler.load_state_dict(opt_data['scaler_state_dict']); print("Loaded scaler state.")
                if lr_scheduler and 'lr_state_dict' in opt_data: lr_scheduler.load_state_dict(opt_data['lr_state_dict']); print("Loaded LR scheduler state.")
            except Exception as e: print(f"Error loading optimizer/scaler/scheduler state from {opt_path}: {e}."); traceback.print_exc()
        else: print(f"Optimizer state file {opt_path} not found. Resuming training without optimizer state.")
    else: print("Resume flag not set. Loading model weights only.")

    return last_completed_epoch, losses, best_epoch, best_criterion_value

def main(args):
    assert args.batch_size % args.acc_grad_iter == 0, "Batch size must be divisible by acc_grad_iter"
    if args.start_val_epoch is None:
        args.start_val_epoch = max(0, args.num_epochs - BASE_NUM_VAL_EPOCHS)
    if args.crop_dim is not None and args.crop_dim <= 0: args.crop_dim = None

    classes, train_data, val_data, val_data_frames = get_datasets(args)
    num_actual_classes = len(classes) # Number of foreground classes
    model_num_classes = num_actual_classes + 1 # Model outputs include background

    loader_batch_size = args.batch_size // args.acc_grad_iter
    actual_num_workers = get_num_train_workers(args)
    train_prefetch_factor = 2 if actual_num_workers > 0 else None
    print(f"Using DataLoader with num_workers={actual_num_workers}, prefetch_factor={train_prefetch_factor}")
    train_loader = DataLoader(train_data, shuffle=False, batch_size=loader_batch_size, pin_memory=True,
                              num_workers=actual_num_workers, prefetch_factor=train_prefetch_factor,
                              worker_init_fn=worker_init_fn)

    val_num_workers = BASE_NUM_WORKERS
    val_prefetch_factor = 2 if val_num_workers > 0 else None
    print(f"Using Validation DataLoader with num_workers={val_num_workers}, prefetch_factor={val_prefetch_factor}")
    val_loader = DataLoader(val_data, shuffle=False, batch_size=loader_batch_size, pin_memory=True,
                            num_workers=val_num_workers, prefetch_factor=val_prefetch_factor,
                            worker_init_fn=worker_init_fn)

    model = E2EModel(model_num_classes, args.feature_arch, args.temporal_arch,
                     clip_len=args.clip_len, modality=args.modality,
                     multi_gpu=args.gpu_parallel)
    optimizer, scaler = model.get_optimizer({'lr': args.learning_rate})

    num_steps_per_epoch = len(train_loader) // args.acc_grad_iter
    if num_steps_per_epoch == 0 and len(train_loader) > 0:
        print("Warning: num_steps_per_epoch calculated as 0. Setting to 1.")
        num_steps_per_epoch = 1

    num_epochs, lr_scheduler = get_lr_scheduler(args, optimizer, num_steps_per_epoch)

    losses = []
    best_epoch = None
    initial_best_criterion = 0.0 if args.criterion == 'map' else float('inf')
    best_criterion = initial_best_criterion
    start_epoch = -1 

    # === RESUME TRAINING LOGIC ===
    # If resuming or checkpoints exist, load the last state and best historical performance.
    if args.resume or get_last_epoch(args.save_dir) >= 0:
        try:
            start_epoch, losses, history_best_epoch, history_best_criterion = load_from_save(
                args, model, optimizer, scaler, lr_scheduler)
            print(f"Resuming from epoch {start_epoch + 1}. History best epoch: {history_best_epoch}, criterion value: {history_best_criterion:.4f}")
            best_epoch = history_best_epoch
            best_criterion = history_best_criterion
        except Exception as e:
             print(f"Error loading from save directory {args.save_dir}: {e}. Starting training from epoch 0.")
             traceback.print_exc()
             start_epoch = -1; losses = []; best_epoch = None; best_criterion = initial_best_criterion

    # === MAIN TRAINING LOOP ===
    for current_epoch in range(start_epoch + 1, num_epochs):
        print(f"\nStarting Epoch {current_epoch}/{num_epochs}")
        try:
            # --- Training Step for one epoch ---
            train_loss = model.epoch(train_loader, optimizer, scaler, lr_scheduler=lr_scheduler, acc_grad_iter=args.acc_grad_iter)
            print(f'[Epoch {current_epoch}] Train loss: {train_loss:.5f}')

            # --- Validation Step for one epoch ---
            val_loss = model.epoch(val_loader, acc_grad_iter=args.acc_grad_iter) # No optimizer, so it's validation
            print(f'[Epoch {current_epoch}] Val loss: {val_loss:.5f}')

            current_mAP = None 
            is_best = False
            
            # === BEST MODEL SELECTION (Related to Concept 11: Early Stopping Logic) ===
            # Monitors validation performance (loss or mAP) and saves the best model checkpoint.
            if args.criterion == 'loss': 
                if val_loss < best_criterion:
                    best_criterion = val_loss
                    best_epoch = current_epoch
                    is_best = True
                    print('New best epoch found based on validation loss!')
            elif args.criterion == 'map': 
                if current_epoch >= args.start_val_epoch:
                    pred_file = os.path.join(args.save_dir, f'pred-val.{current_epoch:03d}') if args.save_dir else None
                    if val_data_frames is None:
                         print("Warning: criterion is 'map' but val_data_frames is None. Cannot calculate mAP.")
                    else:
                         print(f'Starting mAP evaluation on validation set for epoch {current_epoch}...')
                         current_mAP = evaluate(model, val_data_frames, 'VAL', classes, pred_file, save_scores=False)
                         if current_mAP is not None and current_mAP > best_criterion:
                             best_criterion = current_mAP
                             best_epoch = current_epoch
                             is_best = True
                             print('New best epoch found based on validation mAP!')
                else:
                    print(f'Skipping mAP evaluation for epoch {current_epoch} (before start_val_epoch {args.start_val_epoch}).')

            losses.append({'epoch': current_epoch, 'train': train_loss, 'val': val_loss, 'val_mAP': current_mAP})

            # === SAVING CHECKPOINTS ===
            # Concept 11: Best Model Saving (Saves checkpoint_best.pt if is_best)
            if args.save_dir is not None:
                os.makedirs(args.save_dir, exist_ok=True)
                # Save history of losses and mAP
                store_json(os.path.join(args.save_dir, 'loss.json'), losses, pretty=True)
                
                # Save checkpoint for the current epoch
                checkpoint_path = os.path.join(args.save_dir, f'checkpoint_{current_epoch:03d}.pt')
                try: torch.save(model.state_dict(), checkpoint_path); print(f"Saved checkpoint: {checkpoint_path}")
                except Exception as e: print(f"Error saving checkpoint {checkpoint_path}: {e}")
                
                # Save optimizer, scaler, and lr_scheduler state (useful for resuming)
                clear_files(args.save_dir, r'optim_\d+\.pt') # Keep only latest optimizer state
                optim_path = os.path.join(args.save_dir, f'optim_{current_epoch:03d}.pt')
                opt_dict = {'optimizer_state_dict': optimizer.state_dict()}
                if scaler: opt_dict['scaler_state_dict'] = scaler.state_dict()
                if lr_scheduler: opt_dict['lr_state_dict'] = lr_scheduler.state_dict()
                try: torch.save(opt_dict, optim_path); 
                except Exception as e: print(f"Error saving optimizer state {optim_path}: {e}")
                
                # Save config file
                store_config(os.path.join(args.save_dir, 'config.json'), args, num_epochs, classes)
                
                # If the current model is the best one so far (based on criterion), save it separately
                if is_best:
                     best_cp_path = os.path.join(args.save_dir, 'checkpoint_best.pt')
                     # Saves the weights of the model that performed best on validation set
                     try: torch.save(model.state_dict(), best_cp_path); print(f"Saved best checkpoint: {best_cp_path}")
                     except Exception as e: print(f"Error saving best checkpoint {best_cp_path}: {e}")

        except Exception as e: # Catch errors during an epoch
             print(f"\nAn error occurred during epoch {current_epoch}: {e}\n")
             print("Full traceback:")
             traceback.print_exc()
             break # Stop training if a major error occurs

    print('\nTraining finished.')
    if best_epoch is not None: print(f'Best epoch found: {best_epoch} with validation {args.criterion} = {best_criterion:.4f}')
    else: print('No best epoch determined.')

    # === FINAL EVALUATION USING THE BEST MODEL ===
    # Uses the best checkpoint saved during training (Concept 11)
    if args.save_dir is not None and best_epoch is not None:
        print(f'\nEvaluating best model from epoch {best_epoch} on test/challenge splits...')
        best_checkpoint_path = os.path.join(args.save_dir, f'checkpoint_{best_epoch:03d}.pt')
        # Fallback to generic best checkpoint if specific epoch file missing
        if not os.path.exists(best_checkpoint_path):
             best_checkpoint_path = os.path.join(args.save_dir, 'checkpoint_best.pt')

        if os.path.exists(best_checkpoint_path):
             try:
                 print(f'Loading best model state_dict from {best_checkpoint_path}')
                 # Load model state dict with map_location to handle potential device mismatches
                 model.load(torch.load(best_checkpoint_path, map_location=model.device))
             except Exception as e:
                 print(f'Error loading best model checkpoint {best_checkpoint_path}: {e}. Skipping final evaluation.')
                 traceback.print_exc() # Print loading error traceback
                 return # Exit if best model cannot be loaded

             eval_splits = ['test', 'challenge']
             # Optionally re-evaluate on validation set with the best model
             # if args.criterion != 'map' and val_data_frames: eval_splits.insert(0, 'val')

             for split in eval_splits:
                 split_path = os.path.join('data', args.dataset, f'{split}.json')
                 if os.path.exists(split_path):
                      print(f'\nLoading data for {split.upper()} split...')
                      try:
                          split_data = ActionSpotVideoDataset(
                              classes, split_path, args.frame_dir, args.modality,
                              args.clip_len, overlap_len=args.clip_len // 2,
                              crop_dim=args.crop_dim, pad_len=DEFAULT_PAD_LEN,
                              flip=False, multi_crop=False) # Use standard eval settings
                          split_data.print_info() # Print info which might include debug prints
                          pred_file = os.path.join(args.save_dir, f'pred-{split}.best_{best_epoch:03d}')
                          perform_stats = split not in ['challenge'] # Don't calc stats for challenge split typically
                          print(f'Starting evaluation on {split.upper()} split...')
                          evaluate(model, split_data, split.upper(), classes, pred_file,
                                   calc_stats=perform_stats, save_scores=True)
                          # evaluate now prints results internally
                      except Exception as e:
                           print(f'Error during evaluation on {split.upper()} split: {e}. Skipping.')
                           traceback.print_exc() # Print traceback for evaluation error
                 else: print(f'Data file not found: {split_path}. Skipping evaluation on {split.upper()}.')
        else: print(f'Best checkpoint ({best_checkpoint_path}) not found. Skipping final evaluation.')
    elif args.save_dir is not None and best_epoch is None: print('\nSkipping final evaluation: No best epoch determined.')
    else: print('\nSkipping final evaluation: Save directory not specified.')


if __name__ == '__main__':
    # Concept 20: Hyperparameter Tuning (get_args() reads command line args/hyperparameters)
    main(get_args())