#!/usr/bin/env python3
""" Inference for E2E-Spot """

import os
import argparse
import re
import torch
import traceback
from dataset.frame import ActionSpotVideoDataset
from util.io import load_json, store_json, store_gz_json # Added store_gz_json just in case evaluate saves in that format too
from util.dataset import load_classes
from train_e2e import E2EModel, evaluate



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', help='Path to the model dir (should contain checkpoint and author\'s config.json)')
    parser.add_argument('frame_dir', help='Path to the frame dir')
    parser.add_argument('-s', '--split',
                        choices=['train', 'val', 'test', 'challenge'],
                        required=True)
    parser.add_argument('--no_overlap', action='store_true', help="Disable overlapping windows during evaluation inference.")

    save = parser.add_mutually_exclusive_group()
    save.add_argument('--save', action='store_true',
                      help='Save predictions with default names in model_dir')
    save.add_argument('--save_as', help='Save predictions with a custom prefix (e.g., path/to/my_preds)')

    parser.add_argument('-d', '--dataset',
                        help='Dataset name (e.g., soccernetv2) if not inferrable from the config in model_dir')
    # --- Add an explicit checkpoint file argument ---
    parser.add_argument('--checkpoint_file', type=str, default=None,
                        help='Name of the checkpoint file within model_dir (e.g., checkpoint_best.pt). If None, tries to infer.')
    # --- End Add ---
    return parser.parse_args()


def find_checkpoint_file(model_dir, specified_filename=None):
    if specified_filename:
        chkpt_path = os.path.join(model_dir, specified_filename)
        if os.path.isfile(chkpt_path):
            print(f"Using specified checkpoint file: {chkpt_path}")
            return chkpt_path
        else:
            print(f"Warning: Specified checkpoint file '{specified_filename}' not found in '{model_dir}'. Trying to infer.")

    # Try common names first
    common_names = ['checkpoint_best.pt', 'model_best.pt', 'checkpoint.pt', 'model.pth']
    for name in common_names:
        chkpt_path = os.path.join(model_dir, name)
        if os.path.isfile(chkpt_path):
            print(f"Found checkpoint: {chkpt_path}")
            return chkpt_path

    # Fallback to epoch-based discovery (get_best_epoch or get_last_epoch)
    loss_json_path = os.path.join(model_dir, 'loss.json')
    best_epoch = -1
    if os.path.isfile(loss_json_path):
        try:
            data = load_json(loss_json_path)
            # Try to find best epoch based on val_mAP, then val loss, then just epoch
            # This logic might need adjustment based on what's in the author's loss.json
            if data and isinstance(data, list):
                best_val_map_entry = max([e for e in data if e.get('val_mAP') is not None], key=lambda x: x['val_mAP'], default=None)
                if best_val_map_entry:
                    best_epoch = best_val_map_entry['epoch']
                else:
                    best_val_loss_entry = min([e for e in data if e.get('val') is not None], key=lambda x: x['val'], default=None)
                    if best_val_loss_entry:
                        best_epoch = best_val_loss_entry['epoch']
            if best_epoch != -1:
                 print(f"Inferred best epoch from loss.json: {best_epoch}")
        except Exception as e:
            print(f"Warning: Could not parse loss.json to find best epoch: {e}")

    if best_epoch == -1: # If loss.json didn't help, try finding latest numbered checkpoint
        regex = re.compile(r'checkpoint_(\d+)\.pt')
        last_epoch = -1
        for file_name in os.listdir(model_dir):
            m = regex.match(file_name)
            if m:
                epoch = int(m.group(1))
                last_epoch = max(last_epoch, epoch)
        if last_epoch != -1:
            best_epoch = last_epoch
            print(f"Found last numbered checkpoint epoch: {best_epoch}")

    if best_epoch != -1:
        chkpt_path = os.path.join(model_dir, f'checkpoint_{best_epoch:03d}.pt')
        if os.path.isfile(chkpt_path):
            print(f"Using inferred checkpoint: {chkpt_path}")
            return chkpt_path

    print(f"Error: Could not find a suitable checkpoint file in '{model_dir}'.")
    print("Please ensure the author's checkpoint .pt file is in this directory and consider using --checkpoint_file argument.")
    return None


def main(model_dir, frame_dir, split, no_overlap, save, save_as, dataset, checkpoint_file):
    config_path = os.path.join(model_dir, 'config.json')
    if not os.path.isfile(config_path):
        print(f"Error: config.json not found in model directory: {model_dir}")
        return
    
    print("--- Loaded Author's Configuration (config.json) ---")
    with open(config_path) as fp:
        print(fp.read())
    print("----------------------------------------------------")
    config = load_json(config_path)

    # Determine the checkpoint file to load
    model_checkpoint_path = find_checkpoint_file(model_dir, checkpoint_file)
    if not model_checkpoint_path:
        return # Exit if no checkpoint found

    if dataset is None:
        dataset = config.get('dataset')
        if dataset is None:
            print("Error: Dataset name not found in config.json and not provided as an argument.")
            return
        print(f"Inferred dataset from config.json: {dataset}")
    else:
        if dataset != config.get('dataset'):
            print(f"Warning: Provided dataset '{dataset}' differs from config's dataset '{config.get('dataset')}'. Using provided '{dataset}'.")

    # Load local class definitions for mapping labels in the final output
    local_class_file = os.path.join('data', dataset, 'class.txt')
    if not os.path.isfile(local_class_file):
        print(f"Error: Your local class.txt not found at: {local_class_file}")
        return
    local_classes_for_eval_mapping = load_classes(local_class_file)
    print(f"Loaded {len(local_classes_for_eval_mapping)} classes from your local '{local_class_file}' for final evaluation label mapping.")

    # --- Use num_classes from the AUTHOR'S config.json for model initialization ---
    # config['num_classes'] typically refers to foreground classes in these configs.
    # The model's output layer needs +1 for the background class.
    author_foreground_classes = config.get('num_classes')
    if author_foreground_classes is None:
        print(f"Error: 'num_classes' not found in author's config.json from {model_dir}")
        return
    num_model_outputs_from_author_config = author_foreground_classes + 1
    print(f"Initializing E2EModel with {author_foreground_classes} foreground classes + 1 background = {num_model_outputs_from_author_config} total outputs, based on author's config.json.")

    # --- Initialize Model based on Author's Config ---
    try:
        model = E2EModel(
            num_model_outputs_from_author_config,
            config['feature_arch'],
            config['temporal_arch'],
            clip_len=config['clip_len'],
            modality=config['modality'],
            # Use .get() for optional parameters for robustness
            multi_gpu=config.get('gpu_parallel', False)
        )
    except KeyError as e:
        print(f"Error: Missing key {e} in author's config.json needed for model initialization.")
        return

    # --- Load Pre-trained Weights ---
    try:
        print(f"Attempting to load pre-trained weights from: {model_checkpoint_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict_to_load = torch.load(model_checkpoint_path, map_location=device)
        
        # Handle different ways state_dict might be saved in the checkpoint
        if 'model_state_dict' in state_dict_to_load:
            model.load(state_dict_to_load['model_state_dict'])
        elif 'state_dict' in state_dict_to_load: # Another common pattern
            model.load(state_dict_to_load['state_dict'])
        else: # Assume the checkpoint file is directly the model's state_dict
            model.load(state_dict_to_load)
        print("Successfully loaded pre-trained weights.")
        model._model.to(device) # Ensure model is on the correct device after loading
        model.device = device   # Update the device attribute in the wrapper
    except FileNotFoundError:
        print(f"ERROR: Checkpoint file not found at {model_checkpoint_path}")
        return
    except RuntimeError as e:
        print(f"ERROR loading state_dict: {e}")
        print("This usually indicates a mismatch between the model architecture defined by author's config.json and the weights, or a corrupted file.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading weights: {e}")
        traceback.print_exc()
        return

    # --- Prepare Dataset for Evaluation ---
    split_json_path = os.path.join('data', dataset, f'{split}.json')
    if not os.path.isfile(split_json_path):
        print(f"Error: Split data file not found: {split_json_path}")
        return
    
    # Use author's config for dataset parameters during evaluation if they affect how model sees data
    eval_clip_len = config.get('clip_len', 100)
    eval_modality = config.get('modality', 'rgb')
    eval_crop_dim = config.get('crop_dim', None) # crop_dim might be null

    try:
        split_data = ActionSpotVideoDataset(
            local_classes_for_eval_mapping, # Use local classes for this dataset object, primarily for label iteration
            split_json_path,
            frame_dir,
            eval_modality, # Use modality from author's config
            eval_clip_len, # Use clip_len from author's config
            overlap_len=0 if no_overlap else eval_clip_len // 2,
            crop_dim=eval_crop_dim # Use crop_dim from author's config
        )
        print(f"Created ActionSpotVideoDataset for split '{split}' with {len(split_data.videos)} videos.")
    except Exception as e:
        print(f"Error creating ActionSpotVideoDataset for split '{split}': {e}")
        traceback.print_exc()
        return

    # --- Determine Save Path for Predictions ---
    pred_file_prefix = None
    if save_as is not None:
        pred_file_prefix = save_as
    elif save: # Note: README implies --save saves in model_dir, test_e2e.py also has this logic.
        # Extract a base name from the checkpoint file for uniqueness
        checkpoint_basename = os.path.splitext(os.path.basename(model_checkpoint_path))[0]
        pred_file_prefix = os.path.join(model_dir, f'pred-{split}.{checkpoint_basename}')
    
    if pred_file_prefix is not None:
        print(f'Predictions will be saved with prefix: {pred_file_prefix}')
        # Ensure directory exists for save_as if it's a full path
        if os.path.dirname(pred_file_prefix):
             os.makedirs(os.path.dirname(pred_file_prefix), exist_ok=True)


    # --- Run Evaluation ---
    print(f"\nStarting evaluation on {split.upper()} split...")
    # The evaluate function from train_e2e.py is used.
    # It needs the model, the dataset for the current split, the split name,
    # your local class mapping (for outputting human-readable labels in the prediction JSON),
    # and the prefix for saving predictions.
    # calc_stats=False in test_e2e.py usually means just generate predictions,
    # and then use a separate script like eval_soccernetv2.py for official metrics.
    try:
        evaluate(model, split_data, split.upper(), local_classes_for_eval_mapping, pred_file_prefix,
                 calc_stats=False, save_scores=True) # save_scores=True to get the .score.json.gz
        print(f"Evaluation complete. Predictions saved with prefix: {pred_file_prefix} (if saving was enabled)")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    args = get_args()
    try:
        main(**vars(args))
    except Exception as main_err:
        print(f"\n !!! An error occurred in main execution of test_e2e.py: {main_err} !!!")
        traceback.print_exc()