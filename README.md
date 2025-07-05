# 🎸 Guitar Chord Recognition with CNN and GPT-4o

A deep learning project that recognizes guitar chords using hand gestures captured via webcam. This project combines:

- A **CNN model** trained on chord hand gesture images
- **MediaPipe** for hand tracking
- **OpenAI GPT-4o API** to display chord details in real time

## 📸 Overview

- Capture hand gesture images using a webcam and save them per chord class
- Train a CNN model to classify grayscale hand images
- Use a webcam to predict chords in real time
- Fetch and display chord-related information using GPT-4o API

---


## 🔧 Requirements

Install the dependencies before running the code:

```bash
pip install -r requirements.txt
```

Also make sure you have access to the OpenAI GPT-4o API, if you want info on chords too, and update your API key in .env file

## Steps:

### 🎼 Step 1: Collect Chord Images
Run the image capture script to record hand gestures for each chord. Make sure to move your fingers slightly while recording to avoid overfitting. To run:

``` bash
cd data_prep
python create_chords_dataset.py
```

* Input the chord name (e.g., "C Major", "G Major")
* Press c to toggle capture (pause/start) if required
* It will eventually stop after capturing 1000 images, you can also press q to quit in between

Captured images will be saved under chords_dataset/<chord_name>/.


### 🧠 Step 2: Train the Model
Once the dataset is ready, train the CNN model from the root directory using:

```
python train_model.py
```

This script:

* Applies data augmentation
* Trains a CNN on grayscale hand images
* Evaluates the model using a confusion matrix and classification report
* Saves the best model to guitar_model.h5

### 🎛️ Step 3: Real-Time Inference
Run the inference script to start webcam-based chord recognition:

```
python main.py
```
This will:

* Use MediaPipe to detect the hand
* Crop the region of interest (ROI)
* Predict the chord using your trained CNN model
* Query GPT-4o API for chord information
* Overlay the chord name and info onto the webcam feed

Press q to exit the window.


## 🧠 GPT-4o Integration
The chord info is retrieved in real time using GPT-4o. Check the following function::
```bash
# gpt.py
def get_chord_info(chord_label: str) -> str:
    # Use OpenAI API to get definition / tips / theory
    ...
```
If you don't have OpenAI key, you can comment out use of it, it will still detect chords 

## 🤖 Model Architecture

<pre> ```text Input: (200x200 grayscale image) ↓ Conv2D → MaxPooling ↓ Conv2D → MaxPooling ↓ Flatten → Dense(1024) → Dropout(0.6) ↓ Output: Softmax layer with N chord classes ``` </pre>


## ✅ Example Chords
For now, I have trained and detected the following chords:

* C major
* D major
* E minor
* F major
* G major

You can update the chord dataset and train the model on other chords as well.


## 📈 Results
Model evaluation includes:

* Accuracy & loss metrics
* Confusion matrix visualization
* Classification report

## 📌 Tips
* Use uniform lighting during image capture
* Capture from consistent angles and positions
* Balance number of samples per chord class
* As of 5th July 2025, Python 3.12 has some issues with the MediaPipe library. Use Python 3.10 or 3.11 if you encounter any version-related problems.

## 📝 License
MIT License. Feel free to use and extend.

## 💬 Credits
Created with 🧠 CNN + 🎥 OpenCV + 🖐️ MediaPipe + 🤖 GPT-4o + ❤️ for Guitars