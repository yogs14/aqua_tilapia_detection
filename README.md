# Aqua Tilapia Detection and Tracking

![Example Tracking](images/example_tracking.png)

This project provides a high-performance solution for detecting and tracking Tilapia fish in video streams using **YOLO11** and **ByteTrack**. Unlike standard trackers, this system leverages **ByteTrack** to robustly handle occlusions (when fish cross each other) and utilizes **GPU acceleration** for real-time processing.

## Key Features

- **Advanced Detection:** Uses the latest **YOLO11** model (`yolo11s.pt` or `yolo11n.pt`) for high-accuracy fish detection.
- **Robust Multi-Object Tracking:** Implements **ByteTrack** (via `supervision`) instead of SORT. This allows the system to maintain consistent fish IDs even when detection confidence is low or when fish temporarily overlap.
- **GPU Acceleration:** Fully optimized to run on NVIDIA GPUs (`device=0`) for both training and inference, ensuring smooth playback and fast processing.
- **Speed & Trajectory Analysis:** Calculates instantaneous/average speed (px/s) and visualizes movement paths.
- **Data Logging:** Automatically saves tracking data (Timestamp, ID, Coordinates) to CSV for further analysis.

## Technologies Used

- **Python 3.x**
- **Ultralytics YOLO:** For object detection.
- **Supervision & ByteTrack:** For state-of-the-art multi-object tracking.
- **OpenCV (`cv2`):** For video manipulation and drawing.
- **NumPy & Pandas:** For data handling.
- **Matplotlib:** For generating analytics graphs.
- **CUDA (Optional but Recommended):** For GPU acceleration.

## Project Structure

```text
aqua_tilapia_detection/
├── .gitignore
├── README.md
├── requirements.txt              # Project dependencies
├── archive/                      # Legacy files (SORT, centroids, etc.)
├── assets/                       # Input videos
│   └── video_shorter.mp4
├── images/                       # Documentation images
├── models/                       # Trained models
│   └── best.pt                   # Place your trained YOLO model here
├── result/                       # Output metrics & graphs
│   ├── fish_positions.csv
│   ├── average_speed_over_time.png
│   └── fish_trajectories.png
├── src/                          # Source code
│   ├── bytetrack.py              # ByteTrack wrapper implementation
│   ├── data_handler.py           # Data plotting and saving logic
│   ├── main.py                   # Entry point script
│   ├── utils.py                  # Helper functions (distance calc)
│   └── video_processor.py        # Core processing logic (YOLO + ByteTrack)
└── train/                        # Training module
    └── train.py                  # Script to train YOLO with Roboflow data
````

## Setup and Installation

### 1\. Clone the Repository

```bash
git clone <repository_url>
cd aqua_tilapia_detection
```

### 2\. Set Up Virtual Environment (Recommended)

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3\. Install Dependencies

Ensure you have the required libraries. Note that `supervision` and `lapx` are required for ByteTrack.

```bash
pip install -r requirements.txt
```

*(Note: `lapx` is recommended for faster linear assignment on Windows/Linux)*

### 4\. GPU Setup (Highly Recommended)

To enable GPU acceleration (as configured in the code), uninstall the CPU version of PyTorch and install the CUDA version matching your driver (e.g., CUDA 11.8):

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
```

## Usage

### Running Detection & Tracking

1.  Place your trained model in `models/best.pt`.
2.  Run the main script:

<!-- end list -->

```bash
python src/main.py
```

The system will:

  * Load the video from `assets/`.
  * Process frames using **GPU (device=0)**.
  * Apply **ByteTrack** to associate detections.
  * Save outputs to the `result/` folder.

### Configuration (`src/video_processor.py`)

You can tweak tracking sensitivity in `src/video_processor.py`:

  * **`conf`**: Confidence threshold for YOLO (default `0.1` - `0.25`).
  * **`track_thresh`**: Minimum confidence for ByteTrack to start tracking.
  * **`imgsz`**: Inference image size (ensure this matches your training size, e.g., `640` or `480`).

## Training the Model

The `train/train.py` script has been updated to support **GPU training** and optimized worker allocation.

### Standard Training Command

To train a model (e.g., YOLO11 Small) using your GPU:

```bash
python train/train.py --api_key YOUR_ROBOFLOW_API_KEY --batch 4 --imgsz 640 --model_base yolo11s.pt
```

### Optimized Parameters

The script is pre-configured with the following optimizations:

  * **`device=0`**: Forces training on the first NVIDIA GPU.
  * **`workers=4`**: Optimized for systems with decent CPU cores but limited RAM, preventing memory overflows during data loading.

**Custom Arguments:**

  * `--epochs`: Number of training cycles (default: 200).
  * `--batch`: Batch size (reduce to 4 or 2 if you encounter VRAM errors).
  * `--model_base`: Choose `yolo11n.pt` (Nano) for speed or `yolo11s.pt` (Small) for accuracy.

-----

**Note:** If you encounter `AMP checks failed` warnings on GTX 16xx series cards, this is normal and training will continue safely in FP32 mode.

```
