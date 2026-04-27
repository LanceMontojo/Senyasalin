# Senyasalin System Overview
Offline Filipino Sign Language Recognition with Real-Time Corrective and Adaptive Feedback

Senyasalin is an interactive learning platform designed to help users learn Filipino Sign Language (FSL) through real-time gesture recognition and feedback. Powered by an LSTM-based model and hand landmark detection, the system allows users to practice, receive corrections, and track their progress—all in an offline environment.

## Home Section
The Home section serves as the entry point of the system, introducing users to Senyasalin’s purpose and core functionality. It presents the platform as an interactive learning tool that combines gesture recognition with real-time feedback. Users are given an overview of how the system works, along with options to begin learning or explore more about its features. The goal of this section is to orient users and prepare them for guided and interactive sign language practice.

## Learn Section 
The Learn section provides users with instructional content through organized demonstration videos. Each category, such as Numbers, Family, Colors, Relationship, and Survival Signs, contains short visual demonstrations that show the correct execution of each gesture. For instance, the Numbers module introduces signs from One to Twenty through step-by-step demonstrations. This section is designed to help users build a foundational understanding of gestures before attempting recognition-based interaction.

## Select Section
The Select section allows users to actively practice specific gestures of their choice. Users begin by selecting a category and then choosing a corresponding sign, such as a number or a family-related gesture. Once a gesture is selected, the system captures the user’s hand movements and evaluates them using the trained recognition model. Immediate feedback is then provided, informing the user whether the gesture was performed correctly. In cases of incorrect execution, the system offers corrective guidance to help users adjust and improve their performance through repetition.

An example of the feedback provided by the system is shown below.

https://github.com/user-attachments/assets/7c2a0b6f-f316-4565-9ee4-bbaea706bf8b

## Activity Section
The Activity section introduces a more challenging learning approach by removing user-selected input. Instead of choosing gestures, the system randomly assigns them, requiring users to recall and perform the correct sign independently. This shifts the learning experience from guided practice to active recall, helping reinforce memory and retention. Despite the increased difficulty, the system continues to provide feedback after each attempt, allowing users to identify mistakes and refine their gestures.

## Auto Section
The Auto section enables continuous, free-flow gesture recognition. In this mode, users are not limited to a specific category or prompted gesture. Instead, they can perform signs naturally, and the system continuously detects and predicts each gesture in real time. This creates a more fluid and realistic practice environment, allowing users to transition between different signs without interruption and develop confidence in spontaneous signing.

## Results Section
The Results section tracks user performance and progress.

Metrics include:
- Total attempts
- Correct predictions
- Incorrect predictions
- Usage streak (number of consecutive days using the system)

Category Assessment (Current Session):
- Category needing the most practice
- Best-performing category
