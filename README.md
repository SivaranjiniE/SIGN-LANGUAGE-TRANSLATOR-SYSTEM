# Sign Language Translator System 🤟

This is a real-time sign language recognition project built using Python. The system uses a laptop webcam to detect a hand gesture and predict the corresponding sign.

For the current version, the model is trained to recognize **five signs:**

* A
* B
* C
* OK
* I LOVE YOU

One of the main ideas behind this project is that the signs are not fixed. Users can collect their own hand gesture data, replace or extend the existing dataset, and train the model again according to the signs they want the system to recognize.

## How It Works

The system follows a simple process:

```text
Webcam
   ↓
Hand Detection
   ↓
MediaPipe Hand Landmarks
   ↓
Feature Extraction
   ↓
Random Forest Model
   ↓
Predicted Sign
```

The webcam captures the hand gesture using OpenCV. MediaPipe detects the hand and extracts 21 landmark points from it.

The landmark coordinates are then given to a trained Random Forest model, which predicts the sign.

## Signs Currently Supported

The model included in this repository was trained with:

| Sign       | Label      |
| ---------- | ---------- |
| A          | A          |
| B          | B          |
| C          | C          |
| OK         | OK         |
| I LOVE YOU | I_LOVE_YOU |

## Customizing the Dataset

The project includes a data collection script, so users don't have to use only the five signs provided in the current dataset.

Users can collect their own hand gestures and create a new dataset.

Run:

```bash
python collect_data.py
```

After collecting the required data, train the model again:

```bash
python train_model.py
```

The newly trained model will be saved as:

```text
model/model.pkl
```

This allows the system to be customized for different signs based on the user's own dataset.

## Technologies Used

* **Python** – Main programming language
* **OpenCV** – Webcam access and image processing
* **MediaPipe** – Hand detection and landmark extraction
* **NumPy** – Numerical operations
* **Pandas** – Dataset handling
* **Scikit-learn** – Machine learning
* **Random Forest** – Sign classification
* **Pickle** – Saving and loading the trained model

## Project Structure

```text
SIGN-LANGUAGE-TANSLATOR-SYSTEM/
│
├── dataset/
│   └── landmarks.csv
│
├── model/
│   └── model.pkl
│
├── collect_data.py
├── hand_dectect.py
├── landmark.py
├── main.py
├── predict.py
├── train_model.py
├── requirements.txt
└── .gitignore
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/SivaranjiniE/SIGN-LANGUAGE-TANSLATOR-SYSTEM.git
```

### 2. Open the project folder

```bash
cd SIGN-LANGUAGE-TANSLATOR-SYSTEM
```

### 3. Install the required packages

```bash
pip install -r requirements.txt
```

## Run the Prediction

To use the trained model with the webcam:

```bash
python predict.py
```

Make sure your laptop webcam is available before running the program.

The system will detect the hand gesture and display the predicted sign.

## Dataset

The current dataset is stored in:

```text
dataset/landmarks.csv
```

It contains the landmark coordinates collected for the five signs used in the current model.

The dataset can be replaced or extended using `collect_data.py`.

## Training Your Own Model

If you want the system to recognize different signs:

1. Collect data for the signs you want.
2. Save the collected data in the dataset.
3. Train the model using `train_model.py`.
4. The new model will be saved as `model/model.pkl`.
5. Run `predict.py` to test the new model.

This makes the project flexible instead of limiting it to the five signs included in the original dataset.

## Future Improvements

Some improvements I would like to add in the future are:

* Add more signs
* Recognize complete words and sentences
* Add text-to-speech
* Improve recognition for different hand positions
* Improve performance in different lighting conditions
* Add a simple user interface
* Make the system easier for users to create and manage their own datasets

## Author

**Sivaranjini**

GitHub:
https://github.com/SivaranjiniE

