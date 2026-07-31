# BasketballAnalytics

🏀 Basketball AI Analytics

Basketball AI Analytics is a computer vision project designed to automatically extract tactical and performance insights from basketball game footage.

Using YOLO26 Pose, ByteTrack, homography-based court calibration, and AI-powered analytics, the system transforms broadcast video into structured basketball data that can assist coaches, analysts, and player development staff.

Features 🎥 YouTube video ingestion 👥 Player detection and multi-object tracking 🏀 Ball detection 🎽 Jersey color classification 🔢 Jersey number recognition (EasyOCR) 📐 Dynamic court calibration and homography 📍 Player positioning in real-world court coordinates ⚡ Player speed estimation 🏃 Offensive and defensive spacing analysis 🛡️ Closest defender pressure metrics 🔄 Possession and turnover detection 📈 Transition defense timing 🎯 Shot detection and event logging 💾 SQLite data storage 📺 Real-time tactical HUD visualization Technology Stack Python OpenCV Ultralytics YOLO26 Pose ByteTrack EasyOCR NumPy SQLite Project Goals

The long-term objective is to build an automated basketball analytics platform capable of producing advanced tactical metrics directly from broadcast footage without requiring wearable sensors or dedicated tracking cameras.

Future developments include:

Heatmaps Offensive set recognition Defensive scheme classification Player trajectory prediction Pass network analysis Expected Shot Quality (xShot) Automatic possession segmentation AI-assisted coaching recommendations Architecture

The project is progressively evolving from a monolithic MVP into a modular processing pipeline:

Video Session │ ▼ Frame Loader │ ▼ Gameplay Filter │ ▼ Camera Tracker │ ▼ Detector │ ▼ Analytics Engine │ ▼ Database Service │ ▼ Renderer

Each processing stage operates on a shared FrameState object, making the system modular, testable, and extensible.

Status

🚧 Active Development

1.Install python 2.Install GIT 2. Install Requirements 2.1 run pip install -r requirements.txt 3. To run analytics in terminal - python3 services/analytics/analyzer.py
3. To run dashboards in terminal - streamlit run ui/dashboard.py

This repository is currently undergoing a major architectural refactor to improve maintainability, scalability, and support future AI analytics modules.
