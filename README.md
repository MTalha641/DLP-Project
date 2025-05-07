# Football Analysis System

This repository contains two complementary systems for football (soccer) video analysis:

1. **Player Action Spotting** - A deep learning system to detect and classify player actions in soccer videos.
2. **Football Substitution Planning** - A comprehensive tracking and analysis system to help make informed substitution decisions.

## Player Action Spotting

The Player Action Spotting system uses deep learning to automatically detect key events in soccer videos.

### Features
- End-to-end deep learning pipeline for action detection
- Support for various CNN architectures (ResNet, RegNetY, ConvNeXt)
- Temporal modeling with GRU, TCN, or ASFormer
- Support for RGB, grayscale, and optical flow inputs
- Optimized inference with checkpoint management

### Requirements
```
Python 3.x
PyTorch
torchvision
numpy
timm
tqdm
tabulate
```

### Installation
```bash
# Create a virtual environment
python -m venv playerspot
pip install -r requirements.txt

# For CUDA 12.1 support
pip uninstall numpy
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2"
```

### Dataset
**Note:** The dataset is not included in this repository due to its large size. To run the code, you need to:

1. Download the SoccerNet dataset from [https://www.soccer-net.org/](https://www.soccer-net.org/)
2. Process the dataset using the provided scripts:
   ```bash
   python parse_soccernet.py
   python frames_as_jpg_soccernet.py
   ```
3. Place the processed dataset in the appropriate directory structure before training or testing.

### Usage

#### Testing
```bash
.\playerspot\Scripts\Activate.ps1
python test_e2e.py <model_weights_dir> <frame_dir> -s test --save
```

#### Validation
```bash
.\playerspot\Scripts\Activate.ps1
python test_e2e.py <model_weights_dir> <frame_dir> -s val --save
```

#### Creating Annotated Videos
```bash
python create_annotated_video.py
```

#### Training a New Model
```bash
python train_e2e.py <dataset> <frame_dir> -m <feature_arch> -t <temporal_arch> -s <save_dir>
```

#### Preprocessing SoccerNet Dataset
```bash
python parse_soccernet.py
python frames_as_jpg_soccernet.py
```

## Football Substitution Planning

A computer vision-based system that tracks players and the ball to provide data-driven insights for substitution decisions.

### Features
- Player and ball tracking
- Team assignment
- Ball possession tracking
- Camera movement estimation
- Player speed and distance calculations
- 2D to 3D view transformation
- Visualizations for tactical analysis

### Components
- **Trackers**: YOLO-based object detection and tracking
- **Team Assigner**: Assigns players to teams based on jersey colors
- **Player-Ball Assigner**: Determines which player possesses the ball
- **Camera Movement Estimator**: Accounts for camera movements
- **View Transformer**: Converts pixel coordinates to field coordinates
- **Speed and Distance Estimator**: Calculates player metrics

### Input Data
**Note:** Sample input videos are not included in this repository. To run the Football Substitution Planning system:

1. Place your football match videos in the `Football_Substitution_Planning/input_videos/` directory
2. The expected format is standard video files (MP4, AVI, etc.) of football matches with a clear view of the field

### Usage
```bash
python main.py
```

## Project Structure

```
├── Action/                          # Player Action Spotting
│   ├── author_training_weights/     # Pretrained model weights
│   ├── data/                        # Dataset configuration
│   ├── model/                       # Model definitions
│   ├── test_e2e.py                  # Inference script
│   ├── train_e2e.py                 # Training script
│   ├── parse_soccernet.py           # Data preprocessing
│   ├── frames_as_jpg_soccernet.py   # Frame extraction
│   ├── create_annotated_video.py    # Visualization
│   └── how_to_run.txt               # Quick start guide
│
├── Football_Substitution_Planning/  # Substitution Planning System
│   ├── camera_movement_estimator/   # Camera movement compensation
│   ├── input_videos/                # Source videos
│   ├── models/                      # Pretrained models
│   ├── output_videos/               # Generated outputs
│   ├── player_ball_assigner/        # Ball possession detection
│   ├── speed_and_distance_estimator/# Player metrics calculation
│   ├── stubs/                       # Cached data
│   ├── team_assigner/               # Team classification
│   ├── trackers/                    # Object tracking
│   ├── utils/                       # Utility functions
│   ├── view_transformer/            # 2D to 3D coordinate mapping
│   └── main.py                      # Main execution script
```

## License
This project is for educational purposes.

## Contributors
- 21K-3288
- 21K-4660 