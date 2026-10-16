# AI-Powered Tree Carbon Assessment System

## Project Vision & Overall Architecture

This is a production-ready AI-powered Tree Carbon Assessment System built as a B.Tech Computer Science Engineering capstone project. It is an end-to-end AI application consisting of multiple deep learning models, a backend server, a mobile application, cloud storage, and a database.

The objective is to help farmers estimate the amount of carbon stored in their trees using two images captured from a smartphone.

---

# Project Goal

Develop a mobile application where a farmer can:

1. Register/Login.
2. Upload two images of a tree.
3. Receive an AI-generated report containing:
   * Tree species
   * Tree health
   * Estimated biomass
   * Carbon stored
   * CO₂ equivalent
   * Estimated carbon credit (educational estimate only)
4. Download a professional PDF report.
5. View previous analyses.

The application should be scalable and production-ready.

---

# Why Two Images?

The AI system will require **two different images** because each serves a different purpose.

### Image 1: Full Tree
The farmer captures the entire tree from the base to the top.
* Used for: Tree Detection, Tree Segmentation, Biomass Estimation

### Image 2: Close-up Image
The farmer captures a close-up showing leaves and a portion of the trunk.
* Used for: Tree Species Classification, Tree Health Classification

---

# AI Pipeline

* **Module 1**: Tree Species Classification (Input: Close-up image, Output: Species name, Confidence score)
* **Module 2**: Tree Health Classification (Input: Close-up image, Output: Healthy, Diseased, Moderate)
* **Module 3**: YOLOv11 Tree Detection (Input: Full tree image, Output: Detect primary tree, Crop tree)
* **Module 4**: Tree Segmentation (Input: Detected tree, Output: Tree mask, Canopy mask, Trunk mask, Polygon, Metadata)
* **Module 5 + 6**: Integrated Biomass Prediction Engine (Input: Segmented tree, Species, Health, Output: Predicted Above Ground Biomass)
* **Module 7**: Carbon Analytics Engine (Input: Biomass prediction, Output: Carbon stored, CO₂ equivalent, Estimated carbon credit, PDF report, JSON report, CSV report)

---

# Complete AI Workflow

Farmer -> Full Tree Image & Close-up Image
Full -> Mod 3 -> Mod 4 -> Mod 5/6 -> Mod 7 -> Final Report!
Close-up -> Mod 1 -> Mod 2 -> Mod 5/6!

---

# Mobile Application (Flutter)
The mobile application **must NOT run AI models locally.**
The Flutter app handles: Login/Register, Camera, Uploads, Receiving API Responses, PDF Downloads, History.

# Backend (FastAPI with PostgreSQL & AWS S3)
* **FastAPI** will expose REST APIs.
* **PostgreSQL** will handle Users, Predictions, Reports, History, Downloads.
* **AWS S3** will store Image files and PDF links, keeping the Database light. 
* **Auth**: JWT via FastAPI.

# Cloud & Deployment
* **AWS EC2**: Runs FastAPI and AI Models over Docker.
* **AWS RDS/Local PG**: PostgreSQL database.
* **AWS S3**: Storage.
* **Deploy**: Docker, Docker Compose, Nginx.

# Endpoints
POST /register, POST /login, POST /analyze, GET /prediction/{id}, GET /history/{user_id}, GET /download/report/{id}, GET /download/csv/{id}, DELETE /prediction/{id}

# Important Scientific Note
**Estimated Carbon Credit (Educational Estimate Only)**
The system must never claim to generate official carbon credits. Actual carbon credits require field verification, certification, approved methodologies, and regulatory approval.

# Future Enhancements
Multi-tree analysis, Satellite imagery, GPS mapping, Growth monitoring, Time-series, Web app, IoT sensor integration, Drone image support.
