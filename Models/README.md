# Senyasalin – Model Training Overview

The model was trained using Long Short-Term Memory (LSTM) networks to learn sequential patterns in sign language, particularly the temporal progression of gestures.

To extract spatial features, MediaPipe was implemented for real-time hand tracking. It detects hand landmarks (keypoints), which are then used as input sequences for the LSTM model.

The dataset consists of video recordings of Filipino Sign Language (FSL), where each clip contains a signer performing a specific word or phrase. These videos were sourced from the FSL-105 Dataset developed by Tupal, Isaiah Jassen (2023) from De La Salle University and made publicly available via Mendeley Data.

## Trained Categories and Classes
The current categories and classes the model was trained on are:

Family:
- Mother
- Father
- Son
- Daughter
- Grandmother
- Grandfather
- Cousin
- Auntie
- Uncle

Colors:
- Blue
- Green
- Red
- Yellow
- Orange
- Violet
- Black
- White
- Pink

Survival:
- Yes
- No
- Understand
- Wrong
- Correct
- Please
- Thank You

Relationship:
- Boy
- Girl

Numbers:
- One (1) to Twenty (20)

## Additional Vocabulary (Sentence Construction Support)
Additionally, extra signs were added to extend the system’s functionality and support the creation of simple sentences.

This additional category includes:
- Meat
- Eggs
- Juice
- Fish
- Coffee
- Chicken
- Rice
- Milk
- Drink
- Eat
- Cook
