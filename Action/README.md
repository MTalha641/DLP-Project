# Football Action Recognition

A deep learning system for recognizing and classifying 12 distinct action classes in soccer videos.

## Overview

This system employs state-of-the-art deep learning to automatically identify key events in soccer videos. The pipeline processes video frames and classifies important football actions using a combination of spatial (CNN) and temporal (GRU) models.

## Action Classes

The system is trained to recognize 12 football action classes:

1. **Goals**: When the ball crosses the goal line
2. **Cards**: Yellow or red card shown to a player
3. **Substitutions**: Player replacement during the match
4. **Kick-offs**: Start or restart of play from the center
5. **Throw-ins**: Ball re-entry after going out of bounds
6. **Corner kicks**: Kick from the corner arc
7. **Free kicks**: Direct or indirect free kicks
8. **Goal kicks**: Goal keeper kick after ball goes out
9. **Offsides**: Player in offside position
10. **Shots on target**: Attempts directly at the goal
11. **Penalties**: Penalty kicks awarded
12. **Ball possession**: Tracking which team controls the ball

## Technical Implementation

### Model Architecture

The system uses a two-stage architecture:

1. **Feature Extraction**: CNN-based spatial feature extractor
   - ResNet variants (ResNet18, ResNet50)
   - RegNetY efficient architectures
   - ConvNeXT modern architecture

2. **Temporal Modeling**: Sequence analysis for action recognition
   - GRU (Gated Recurrent Units)
   - Optional: Multi-Stage TCN or ASFormer

### Temporal Shift Module (TSM)
- Zero-parameter temporal modeling by shifting channel features
- Enables temporal reasoning without 3D convolutions

### Network Diagram
```
Input Video Frames
   ↓
CNN Backbone (ResNet/RegNetY/ConvNeXT)
   ↓
[Optional] Temporal Shift Module
   ↓
Global Average Pooling
   ↓
GRU Sequence Model
   ↓
12-Class Classification
```

## Performance Metrics

- **Average mAP**: 40-60% on SoccerNet-v2 benchmark
- **Per-class Performance**:
  - Goals: 60-70% mAP
  - Cards: 50-60% mAP
  - Other actions: Varying based on frequency and visual distinctiveness
- **Real-time Inference**: >25 FPS on modern GPUs with ResNet18 backbone

## Requirements

```
Python 3.x
PyTorch
torchvision
numpy (<2.0)
timm
tqdm
tabulate
```

## Installation

```bash
# Create a virtual environment
python -m venv playerspot

# Activate the environment (Windows)
.\playerspot\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# For CUDA 12.1 support
pip uninstall numpy
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2"
```

## Dataset

This project uses the SoccerNet dataset for training and evaluation:

1. Download the SoccerNet dataset from [https://www.soccer-net.org/](https://www.soccer-net.org/)
2. Process the dataset using:
   ```bash
   python parse_soccernet.py
   python frames_as_jpg_soccernet.py
   ```

## Usage

### Testing with Pre-trained Model

```bash
python test_e2e.py <model_weights_dir> <frame_dir> -s test --save
```

Example:
```bash
python test_e2e.py ./author_training_weights ./dataset/frame_dir -s test --save
```

### Validation

```bash
python test_e2e.py <model_weights_dir> <frame_dir> -s val --save
```

### Creating Annotated Videos

After generating prediction JSON files with the test script, run:
```bash
python create_annotated_video.py
```

This script visualizes detected actions by:
- Reading prediction JSON files generated during testing
- Overlaying action labels on the original video
- Displaying confidence scores for each detected action
- Generating a new video with annotations for easier analysis

### Training a Custom Model

```bash
python train_e2e.py <dataset> <frame_dir> -m <feature_arch> -t <temporal_arch> -s <save_dir>
```

Example:
```bash
python train_e2e.py soccernetv2 ./dataset/frame_dir -m rn18_tsm -t gru -s ./model_output
```

#### Key Training Parameters

- `-m, --feature_arch`: CNN architecture for feature extraction
  - Options: `rn18`, `rn18_tsm`, `rn18_gsm`, `rn50`, `rn50_tsm`, etc.
- `-t, --temporal_arch`: Temporal model architecture
  - Options: `gru`, `deeper_gru`, `mstcn`, `asformer`
- `--clip_len`: Number of frames per clip (default: 100)
- `--batch_size`: Batch size for training (default: 8)
- `--num_epochs`: Total number of training epochs (default: 50)
- `--learning_rate`: Initial learning rate (default: 0.001)
- `--dilate_len`: Label dilation length when training (default: 0)
- `--mixup`: Use mixup augmentation (default: True)

## Optimization Techniques

- **Mixed Precision Training**: Faster training with FP16/FP32 mix
- **Gradient Accumulation**: Support for larger effective batch sizes
- **Learning Rate Scheduling**: Warmup and cosine annealing
- **Checkpointing**: Saving best models based on validation metrics

## Key Files

- `train_e2e.py`: Main training script
- `test_e2e.py`: Inference and evaluation script
- `create_annotated_video.py`: Visualization tool
- `parse_soccernet.py`: Dataset preparation
- `frames_as_jpg_soccernet.py`: Frame extraction for dataset

## Citation

If using SoccerNet dataset, please cite:
```
@inproceedings{Deliège2020SoccerNetv2AM,
  title={SoccerNet-v2: A Dataset and Benchmarks for Holistic Understanding of Broadcast Soccer Videos},
  author={Adrien Deliège and Anthony Cioppa and Silvio Giancola and Meisam J. Seikavandi and Jacob V. Dueholm and Kamal Nasrollahi and Bernard Ghanem and Thomas B. Moeslund and Marc Van Droogenbroeck},
  year={2020}
}
``` 