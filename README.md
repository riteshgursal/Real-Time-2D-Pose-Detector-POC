**Real-Time 3D Pose Estimation & Action Recognition POC**

This repository serves The objective is to efficiently bridge real-time video capture with structured data preparation for advanced 3D Human Pose Estimation and sequential Fine-Grained Action Recognition modules.

🚀 **Execution Instructions**

**A. Setup and Installation**

Clone the Repository:

**git clone** [https://github.com/riteshgursal/Real-Time-2D-Pose-Detector-POC.git]
cd 3d-pose-poc


**Install Dependencies:** (Requires Python 3.9+ for MediaPipe/PyTorch compatibility)

**pip install -r requirements.txt**


**B. Running the Tracker**

Run the main Python script. The script will automatically open your default webcam and start the real-time detection. Press 'q' to exit the application.

**python pose_tracker.py**


**C. Output Artifact**

A file named 2d_keypoints_log.csv is created. This file contains the sequential, normalized 2D coordinates (x, y, z, visibility) for all 33 body landmarks, structured perfectly for ingestion by the downstream PyTorch models.

💡 **Technical Highlights & Research Value**

Real-Time CV Pipeline

Utilizes OpenCV for optimized video stream handling and visualization, ensuring high frame rates (FPS).

Core skill required for any complex Computer Vision Application and model integration.

Robust 2D Feature Extraction

Integrates MediaPipe Pose, a high-performance, pre-trained Deep Learning model for reliable Human Skeleton Extraction.

Demonstrates proficiency in using industry-standard tools to solve the various foundation problem.

**DL Structure & Architecture**

Includes two conceptual PyTorch classes (ThreeDLiftingModel and ActionRecognizer) that define the architecture for the subsequent research phases.

Shows a clear technical roadmap using a GCN for lifting and LSTM/RNN for sequential action classification, matching the mentor's research field.

**ML Data Preparation**

Logs keypoint data in a standardized CSV format, effectively preparing the input tensor for Deep Learning (DL) training in Phase 2.

Bridges the gap between raw video data (Phase 1) and neural network training (Phase 2), a critical step in DL research.

📐**Future Research: Deep Learning Phases**

The project is structured for immediate expansion into research based on the collected data:

**Phase 2:** 3D Lifting Implementation: Replace the ThreeDLiftingModel placeholder with a trained GCN (Graph Convolutional Network) to convert the logged 2D keypoints into estimated 3D body coordinates, essential for accurate action analysis.

**Phase 3:** Action Recognition Implementation: Train the ActionRecognizer (using LSTM/RNN) on sequences of the estimated 3D data to achieve fine-grained action recognition (e.g., classifying specific athletic movements or health-related actions).
