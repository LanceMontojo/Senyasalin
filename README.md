# Senyasalin System Overview
Offline Filipino Sign Language Recognition with Real-Time Corrective and Adaptive Feedback

Senyasalin is an interactive platform for learning Filipino Sign Language (FSL) through real-time gesture recognition and feedback. Using an LSTM-based model and hand landmark detection, the system allows users to practice, receive corrections, and track progress in a fully offline environment.

## Home Section
The Home section introduces the system and its core features. It presents Senyasalin as an interactive learning tool that combines gesture recognition with real-time feedback, allowing users to quickly understand how the platform works and begin their learning experience.

## Learn Section 
The Learn section provides instructional content through demonstration videos organized by category, including Numbers, Family, Colors, Relationship, and Survival Signs. Each module shows the correct execution of gestures—for example, the Numbers module covers signs from One to Twenty. This section helps users build a foundation before moving to interactive practice.

## Select Section
The Select section allows users to practice specific gestures by choosing a category and a corresponding sign. The system captures the user’s movement, evaluates it using the recognition model, and immediately provides feedback. If the gesture is incorrect, corrective guidance is shown to help improve accuracy through repetition.

An example of the feedback provided by the system is shown below.

https://github.com/user-attachments/assets/7c2a0b6f-f316-4565-9ee4-bbaea706bf8b

## Activity Section
The Activity section introduces a challenge mode where gestures are randomly assigned instead of selected by the user. This encourages recall-based learning while maintaining real-time feedback to help users identify and correct mistakes.

## Auto Section
The Auto section enables continuous, free-flow recognition. Users can perform signs naturally without selecting a category, and the system predicts gestures in real time. This supports more fluid and spontaneous practice.

## Results Section
The Results section summarizes user performance, including total attempts, correct and incorrect predictions, and usage streaks. It also provides a category-based assessment for the current session, highlighting areas that need improvement and best-performing categories.

## Installation and Setup
1. Download or clone the repository
2. Open the project folder
3. Run the application: python app.py
4. Go to SIGN UP and create an account/ go to LOGIN and enter credentials
5. After login, access Learn, Select, Activity, Auto, Results

The system runs fully offline. Once launched, users can register and log in to access all features.

## System Requirements
- Webcam-enabled device
- Camera permission enabled
- Sufficient CPU/GPU for real-time processing
