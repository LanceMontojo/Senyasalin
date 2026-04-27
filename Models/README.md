# Senyasalin - Training 

The project was trained with Long-Short Term Memory (LSTM) for learning sequence of data such as the sequence of movement in sign language. Mediapipe was also implemnted as it helps with real-time inference on knowing the keypoints/location of the user's hand and movements. 

The data used consist of video recordings of FSL, each showing a signer performing a specific phrase or expression. The organized collection of these videos was obtained from the FSL-105 Dataset developed by Tupal, Isaiah Jassen (2023) of De La Salle University, and made publicly available through Mendeley Data. 

The current categories and classes the model was trained are:

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
- Values One through Twenty

Additionally, there were signs that are added that extend the system’s functionality to support the creation of simple sentences.
This is the Additional category that includes:
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