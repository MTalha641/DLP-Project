import os
import json
import cv2 # OpenCV for image/video processing
import shutil # For managing temporary directories
from collections import defaultdict

# --- Configuration ---
PREDICTION_FILE_PATH = r'E:\fahad\player_action_spotting\author_weights\pred-test.checkpoint_088.json'
YOUR_TEST_SET_JSON_PATH = r'E:\fahad\player_action_spotting\data\soccernetv2\test.json'
FRAME_DIR_ROOT = r'E:\fahad\player_action_spotting\dataset\frame_dir'
# CLASS_FILE_PATH = r'E:\fahad\player_action_spotting\data\soccernetv2\class.txt' # Not strictly needed if labels are in preds
ANNOTATED_FRAMES_OUTPUT_DIR = r'E:\fahad\player_action_spotting\temp_annotated_frames'
OUTPUT_VIDEO_DIR = r'E:\fahad\player_action_spotting'

# Annotation settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_COLOR = (255, 255, 255)  # White
BG_COLOR = (0, 0, 0) # Black background for text
TEXT_THICKNESS = 1
LINE_TYPE = cv2.LINE_AA
Y_OFFSET_START = 30  # Starting Y position for text annotations
Y_OFFSET_STEP = 25   # Y step for each new line of text

# --- Helper Functions ---

def load_json_data(file_path):
    """Loads data from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None

def normalize_video_id_from_truth(video_id_from_truth_json):
    """Converts slash-separated ID (from truth JSON) to underscore-separated ID (like in pred JSON)."""
    return video_id_from_truth_json.replace('/', '_')

# --- Main Script Logic ---

if __name__ == "__main__":
    # 1. Load prediction data
    predictions_data = load_json_data(PREDICTION_FILE_PATH)
    if not predictions_data:
        exit()

    # 2. Load your specific test set video IDs, their FPS, and normalize IDs for matching predictions
    your_test_set_data = load_json_data(YOUR_TEST_SET_JSON_PATH)
    if not your_test_set_data:
        exit()
    
    your_target_videos_details = {}  # original_truth_id -> {'normalized_pred_id': str, 'fps': float}
    for video_entry_in_truth in your_test_set_data:
        original_truth_id = video_entry_in_truth['video'] # e.g., 'england_epl/2016-2017/...'
        normalized_id_for_pred_matching = normalize_video_id_from_truth(original_truth_id)
        # Get FPS from test.json (parse_soccernet.py adds it), default if missing
        fps = float(video_entry_in_truth.get('fps', 25.0)) 
        your_target_videos_details[original_truth_id] = {
            'normalized_pred_id': normalized_id_for_pred_matching,
            'fps': fps
        }
    
    target_ids_normalized_for_pred_lookup = {details['normalized_pred_id'] for details in your_target_videos_details.values()}
    print(f"Identified {len(your_target_videos_details)} target videos from your test set.")
    # print("Normalized IDs for matching predictions:", target_ids_normalized_for_pred_lookup)


    # 3. Filter predictions to only include target videos (using normalized IDs) and map them for easy access
    video_predictions_map = defaultdict(lambda: defaultdict(list)) # normalized_pred_video_id -> frame_num -> [events]
    unique_pred_videos_matched = set()

    for video_entry_in_preds in predictions_data:
        # Video ID in prediction file is already in underscore format
        pred_video_id_underscore_format = video_entry_in_preds.get('video') 
        if pred_video_id_underscore_format in target_ids_normalized_for_pred_lookup:
            unique_pred_videos_matched.add(pred_video_id_underscore_format)
            for event in video_entry_in_preds.get('events', []):
                frame_num = event.get('frame')
                label = event.get('label', 'N/A')
                score = event.get('score', 0.0)
                if frame_num is not None:
                    video_predictions_map[pred_video_id_underscore_format][frame_num].append({'label': label, 'score': score})
    
    if not unique_pred_videos_matched:
        print(f"Error: The prediction file '{PREDICTION_FILE_PATH}' does not contain predictions for any of your target videos (after ID normalization).")
        print("Normalized target video IDs (from your test.json, for matching preds) were:", target_ids_normalized_for_pred_lookup)
        print("Video IDs found in prediction file were:", {entry.get('video') for entry in predictions_data})
        exit()
    else:
        print(f"Found predictions for {len(unique_pred_videos_matched)} of your target videos in the prediction file.")


    # 4. Create or clear the temporary directory for annotated frames
    if os.path.exists(ANNOTATED_FRAMES_OUTPUT_DIR):
        print(f"Clearing existing temporary annotated frames directory: {ANNOTATED_FRAMES_OUTPUT_DIR}")
        shutil.rmtree(ANNOTATED_FRAMES_OUTPUT_DIR)
    os.makedirs(ANNOTATED_FRAMES_OUTPUT_DIR, exist_ok=True)
    print(f"Created temporary directory for annotated frames: {ANNOTATED_FRAMES_OUTPUT_DIR}")

    # 5. Process each target video
    for original_truth_video_id, details in your_target_videos_details.items():
        normalized_pred_id = details['normalized_pred_id'] # This is 'league_season_game_half'
        video_fps = details['fps']

        if normalized_pred_id not in video_predictions_map:
            print(f"Skipping video {original_truth_video_id} (normalized: {normalized_pred_id}): No predictions found for it.")
            continue

        print(f"\nProcessing video: {original_truth_video_id} (using prediction key: {normalized_pred_id})")
        
        # Frame path should use the format that matches your actual directory names.
        # This is the underscore-separated format, which is stored in `normalized_pred_id`.
        video_frame_path_root = os.path.join(FRAME_DIR_ROOT, normalized_pred_id)

        if not os.path.isdir(video_frame_path_root):
            # Updated error message for clarity
            print(f"  Error: Frame directory not found for video ID '{normalized_pred_id}' at '{video_frame_path_root}'. Skipping video '{original_truth_video_id}'.")
            continue

        # Use the filesystem-friendly normalized_pred_id for output directory and file names
        safe_video_name_for_output = normalized_pred_id 
        video_specific_annotated_frames_dir = os.path.join(ANNOTATED_FRAMES_OUTPUT_DIR, safe_video_name_for_output)
        os.makedirs(video_specific_annotated_frames_dir, exist_ok=True)

        frame_files = sorted([f for f in os.listdir(video_frame_path_root) if f.endswith('.jpg')])
        if not frame_files:
            print(f"  Error: No .jpg frames found in {video_frame_path_root}. Skipping.")
            continue
            
        # Get frame dimensions from the first frame
        first_frame_path = os.path.join(video_frame_path_root, frame_files[0])
        first_frame_img = cv2.imread(first_frame_path)
        if first_frame_img is None:
            print(f"  Error: Could not read first frame {first_frame_path}. Skipping video {original_truth_video_id}.")
            continue
        height, width, _ = first_frame_img.shape # Use underscore for unused 'layers' variable
        frame_size = (width, height)

        output_video_file_path = os.path.join(OUTPUT_VIDEO_DIR, f"{safe_video_name_for_output}_annotated.mp4")
        try:
            # Initialize VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Common codec for .mp4
            video_writer = cv2.VideoWriter(output_video_file_path, fourcc, video_fps, frame_size)
            print(f"  Outputting annotated video to: {output_video_file_path} (FPS: {video_fps}, Size: {frame_size})")
        except Exception as e:
            print(f"  Error initializing VideoWriter for {output_video_file_path}: {e}. Skipping video.")
            continue

        for frame_idx, frame_filename in enumerate(frame_files):
            frame_path = os.path.join(video_frame_path_root, frame_filename)
            frame_img = cv2.imread(frame_path)
            if frame_img is None:
                print(f"    Warning: Could not read frame {frame_path}. Skipping this frame.")
                continue

            # Annotate frame if predictions exist for this frame_idx
            # Frame filenames (000000.jpg, 000001.jpg) correspond to frame_idx (0, 1, ...)
            current_frame_predictions = video_predictions_map[normalized_pred_id].get(frame_idx, [])

            if current_frame_predictions:
                y_offset = Y_OFFSET_START
                for event_pred in current_frame_predictions:
                    label = event_pred['label']
                    score = event_pred['score']
                    text = f"{label}: {score:.2f}"
                    
                    # Simple text background for better readability
                    (text_width, text_height), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, TEXT_THICKNESS)
                    text_x_pos = 10
                    text_y_pos = y_offset
                    
                    rect_start_point = (text_x_pos - 5, text_y_pos - text_height - baseline + 2)
                    rect_end_point = (text_x_pos + text_width + 5, text_y_pos + baseline -2 + 5) # Adjusted for better fit
                    
                    cv2.rectangle(frame_img, rect_start_point, rect_end_point, BG_COLOR, -1) # -1 for filled rectangle
                    cv2.putText(frame_img, text, (text_x_pos, text_y_pos), FONT, FONT_SCALE, FONT_COLOR, TEXT_THICKNESS, LINE_TYPE)
                    y_offset += Y_OFFSET_STEP
            
            # Save annotated frame to temporary directory
            annotated_frame_save_path = os.path.join(video_specific_annotated_frames_dir, frame_filename)
            cv2.imwrite(annotated_frame_save_path, frame_img)

            # Write frame to video
            video_writer.write(frame_img)
            
            if (frame_idx + 1) % 100 == 0: # Print progress
                 print(f"    Processed and wrote frame {frame_idx + 1}/{len(frame_files)}")

        video_writer.release()
        print(f"  Finished creating annotated video: {output_video_file_path}")

    # 6. Optionally, clean up the temporary annotated frames directory
    # print(f"\nAnnotated frames saved in: {ANNOTATED_FRAMES_OUTPUT_DIR}")
    # print("You can delete this directory manually if you no longer need the individual annotated frames.")
    # Or, to automatically delete:
    # try:
    #     shutil.rmtree(ANNOTATED_FRAMES_OUTPUT_DIR)
    #     print(f"Successfully removed temporary directory: {ANNOTATED_FRAMES_OUTPUT_DIR}")
    # except Exception as e:
    #     print(f"Error removing temporary directory {ANNOTATED_FRAMES_OUTPUT_DIR}: {e}")

    print("\nAll processing complete.")