import os
import json
import numpy as np
import pandas as pd
from scipy.signal import uniform_filter1d
from scipy.fft import fft

# Configurable constants
ANALYSIS_BASE_DIR = 'path/to/analysis'  # user to edit
SESSIONS_BASE_DIR = 'path/to/sessions'  # user to edit

PATTERN_DURATION = 23.0  # seconds
PHASE_DURATIONS = {'groundtruth': 300, 'pasat': 150}  # seconds
FPS = 13

# Function to process a single session

def process_session(session):
    # Paths
    usrp_data_path = os.path.join(ANALYSIS_BASE_DIR, session, 'usrp', 'data.csv')
    xenics_frames_path = os.path.join(SESSIONS_BASE_DIR, session, 'xenics', 'npy')
    markers_path = os.path.join(ANALYSIS_BASE_DIR, session, 'analysis_manual_markers.json')
    output_path = os.path.join(ANALYSIS_BASE_DIR, session, 'ml_features')

    # Load data
    usrp_data = pd.read_csv(usrp_data_path)
    with open(markers_path, 'r') as f:
        markers = json.load(f)['markers']

    # Define phases based on markers
    phases = {
        'baseline': (markers['GROUNDTRUTH_START'] + PATTERN_DURATION, markers['GROUNDTRUTH_START'] + PATTERN_DURATION + PHASE_DURATIONS['groundtruth']),
        'pasat1': (markers['PASAT1_START'] + PATTERN_DURATION, markers['PASAT1_START'] + PATTERN_DURATION + PHASE_DURATIONS['pasat']),
        'pasat2': (markers['PASAT2_START'] + PATTERN_DURATION, markers['PASAT2_START'] + PATTERN_DURATION + PHASE_DURATIONS['pasat']),
        'pasat3': (markers['PASAT3_START'] + PATTERN_DURATION, markers['PASAT3_START'] + PATTERN_DURATION + PHASE_DURATIONS['pasat']),
        'recovery': (markers['GROUNDTRUTH_FINAL_START'] + PATTERN_DURATION, markers['GROUNDTRUTH_FINAL_START'] + PATTERN_DURATION + PHASE_DURATIONS['groundtruth'])
    }

    # Prepare output
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    feature_df = pd.DataFrame()

    for phase_name, (t_start, t_end) in phases.items():
        # Define subwindows
        windows = np.arange(t_start, t_end - 30, 15)  # 30s windows with 50% overlap
        windows = windows[windows > t_start + 30]  # Discard first and last if not enough
        if len(windows) < 3:
            continue

        # Create features for each window
        for idx, window_start in enumerate(windows):
            window_end = window_start + 30  # 30 seconds window
            # Compute features here...
            # Placeholder for feature extraction logic
            features = {'session': session,
                        'phase': phase_name,
                        'label_stress': 1 if 'pasat' in phase_name else 0,
                        't_start': window_start,
                        't_end': window_end,
                        'window_idx': idx}
            feature_df = feature_df.append(features, ignore_index=True)

    # Save features to CSV
    feature_df.to_csv(os.path.join(output_path, 'features.csv'), index=False)

# Main entry for script execution
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate ML features per session')
    parser.add_argument('--session', type=str, help='specific session to process')
    args = parser.parse_args()

    if args.session:
        process_session(args.session)
    else:
        for session in os.listdir(ANALYSIS_BASE_DIR):
            process_session(session)