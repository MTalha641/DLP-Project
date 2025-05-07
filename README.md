# Football Video Analysis System

A comprehensive system for football video analysis combining action recognition and player tracking technologies.

## Overview

This project integrates two core functionalities:
1. **Action Recognition**: Deep learning system that identifies football action classes
2. **Player Detection & Tracking**: Computer vision system for player detection and analysis

## Key Components

### Action Recognition
- **Classification**: 17 football action classes:
  1. Ball out of play
  2. Clearance
  3. Corner
  4. Direct free-kick
  5. Foul
  6. Goal
  7. Indirect free-kick
  8. Kick-off
  9. Offside
  10. Penalty
  11. Red card
  12. Shots off target
  13. Shots on target
  14. Substitution
  15. Throw-in
  16. Yellow card
  17. Background (no action)
- **Model Architecture**: 
  - Feature extraction with CNN backbones (ResNet)
  - Temporal modeling with GRU
  - Support for Temporal Shift Module (TSM)
  - GSM for feature extraction

### Player Detection & Tracking
- **Detection**: YOLO-based detection of players and ball
- **Tracking**: ByteTrack algorithm for multi-object tracking
- **Team Assignment**: Jersey color-based team classification
- **Field Analysis**: Transform from pixel to field coordinates
- **Ball Tracking**: Detection and possession assignment
- **Performance Metrics**: Speed and distance calculation

## Project Structure

```
├── Action/                          # Action Recognition
│ 
│   ├── data/                        # Dataset configuration
│   ├── model/                       # Model definitions
│   ├── test_e2e.py                  # Inference script
│   ├── train_e2e.py                 # Training script
│   ├── parse_soccernet.py           # Data preprocessing
│   ├── frames_as_jpg_soccernet.py   # Frame extraction
│   └── create_annotated_video.py    # Visualization
│
├── Football_Substitution_Planning/  # Player Detection & Tracking
│   ├── main.py                      # Main execution script
│   ├── models/                      # YOLO models
│   ├── camera_movement_estimator/   # Camera movement compensation
│   ├── player_ball_assigner/        # Ball possession detection
│   ├── speed_and_distance_estimator/# Player metrics calculation
│   ├── team_assigner/               # Team classification
│   ├── trackers/                    # Object tracking
│   ├── utils/                       # Utility functions
│   └── view_transformer/            # 2D to 3D coordinate mapping
```

## Technical Implementation

### Action Recognition
- **Input**: Video frames from football matches
- **Model Parameters**:
  ```json
  {
    "batch_size": 8,
    "clip_len": 100,
    "crop_dim": null,
    "dataset": "soccernetv2",
    "dilate_len": 0,
    "feature_arch": "rny002_gsm",
    "gpu_parallel": false,
    "learning_rate": 0.001,
    "mixup": true,
    "modality": "rgb",
    "num_classes": 17,
    "num_epochs": 100,
    "start_val_epoch": 80,
    "temporal_arch": "gru",
    "warm_up_epochs": 3
  }
  ```
- **Model Architecture**: 
  - Feature Extraction: GSM (Gate Shift Module)
  - Temporal Modeling: GRU sequence model
  - Clip Length: 100 frames per segment
  - Output: 16 action classes 
- **Training Configuration**: 
  - Dataset: SoccerNet v2
  - Optimization: AdamW with learning rate scheduling

### Player Detection & Tracking
- **Detection**: YOLOv5 for player and ball detection
- **Tracking**: ByteTrack algorithm for reliable multi-object tracking
- **Team Assignment**: Color-based team classification

## Requirements

Core dependencies:
```
Python 3.x
PyTorch
torchvision
numpy
OpenCV
timm
```

## Usage

### Action Recognition
```bash
# Create a virtual environment
python -m venv playerspot
.\playerspot\Scripts\Activate.ps1
pip install -r requirements.txt

# Testing
python Action/test_e2e.py <model_weights_dir> <frame_dir> -s test --save

# Create annotated videos
python Action/create_annotated_video.py
```

### Player Detection & Tracking
```bash
# Run player tracking
python Football_Substitution_Planning/main.py
```

## Dataset

The system uses the SoccerNet dataset for action recognition, which must be downloaded separately:
1. Download from [https://www.soccer-net.org/](https://www.soccer-net.org/)
2. Process using the provided scripts:
   ```
   python parse_soccernet.py
   python frames_as_jpg_soccernet.py
   ```

## Input Videos

For player detection and tracking, place football match videos in:
`Football_Substitution_Planning/input_videos/`

## Known Limitations
- Action recognition performance depends on camera angles and video quality
- Player tracking may lose accuracy in crowded scenes
- Processing speed depends on hardware capabilities

## Contributors
- 21K-3288
- 21K-4660 