**Real-Time 3D Human Pose Estimation and Action Recognition POC**

**Objective:** To develop a robust, multi-stage Deep Learning pipeline capable of translating 2D video data into sequential 3D human pose estimates for real-time action classification (e.g., fall detection, specific activity analysis).

**Tech Stack:** Python, PyTorch (Conceptual Architecture), MediaPipe, OpenCV, NumPy, CSV.

**Implementation Highlights (Phase 1 Completed):**

**Real-Time CV Pipeline:** Implemented an optimized pipeline using OpenCV to stream video and preprocess frames for robust model inference, demonstrating proficiency in managing high-throughput visual data.

**2D Feature Extraction:** Integrated the high-performance MediaPipe Pose model to accurately detect and track 33 normalized 2D joint coordinates across video sequences (the Human Skeleton Extraction stage).

**Structured Data Preparation:** Designed a script to log sequential keypoint data into a standardized CSV format, effectively creating the ground-truth input tensor required for the deep learning training phases.

**Architectural Planning (Future Work):** Defined conceptual PyTorch classes (GCN and LSTM/RNN) to guide the future implementation of the 2D-to-3D Lifting Model and the Action Recognition Classifier, demonstrating a complete technical roadmap aligning with the mentor's research.
