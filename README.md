# AmericanHandSingnLanuage-ASL-Detection
Real-time Sign Language Detection system that converts hand gestures into text and speech using MediaPipe and Machine Learning.

---

## 🔍 Project Overview

LinguaSignAI detects hand gestures using a webcam, converts them into alphabets, allows users to build words, and finally converts the words into speech. It helps bridge communication between sign language users and others.

---

## ⚙️ Features

* Real-time hand tracking using MediaPipe
* 21 hand landmark extraction
* Machine Learning-based gesture classification
* Word formation with user confirmation
* Text-to-Speech output
* Easy keyboard controls

---

## 🧠 How It Works

```
Webcam → MediaPipe → 21 Landmarks → Feature Vector → ML Model → Alphabet → Word → Speech
```

---

## 🛠️ Tech Stack

* Python
* OpenCV
* MediaPipe
* Scikit-learn (Random Forest)
* NumPy, Pandas
* pyttsx3 (Text-to-Speech)

---

## 📦 Requirements

Install all dependencies using:

```bash
pip install opencv-python mediapipe scikit-learn numpy pandas pyttsx3 joblib
```

---

## 📁 Required Files

Make sure your project folder contains:

```
project-folder/
│
├── collect_data.py        # For dataset collection
├── train_model.py         # For training model
├── predict_sign.py        # Main detection program
├── dataset.csv            # Collected dataset (generated)
├── sign_model.pkl         # Trained model (generated)
├── README.md
```

---

## 🚀 How to Run the Project

### Step 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/LinguaSignAI.git
cd LinguaSignAI
```

---

### Step 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

(Or use the manual install command above)

---

### Step 3️⃣ Collect Dataset (Optional but Recommended)

Run:

```bash
python collect_data.py
```

* Enter label (A, B, C...)
* Show hand sign in webcam
* Data will be saved in `dataset.csv`

---

### Step 4️⃣ Train the Model

```bash
python train_model.py
```

Output:

```
sign_model.pkl
```

---

### Step 5️⃣ Run Real-Time Detection

```bash
python predict_sign.py
```

---

## 🎮 Controls

| Key | Action                  |
| --- | ----------------------- |
| C   | Confirm detected letter |
| T   | Retake letter           |
| R   | Speak final word        |
| X   | Clear word              |
| ESC | Exit                    |

---

## 🧪 Example Usage

```
Show H → Press C  
Show E → Press C  
Show L → Press C  
Show L → Press C  
Show O → Press C  
Press R → Output: HELLO (spoken)
```

---

## 📊 Model Details

* Model Used: Random Forest Classifier
* Input: 42 features (21 hand landmarks × x,y)
* Output: Alphabet (A–Z)
* Accuracy: ~90–97% (depends on dataset size)

---

## 📌 Applications

* Assistive communication for deaf/mute users
* Gesture-based interfaces
* Human-computer interaction

---

## ⚠️ Limitations

* Works best with single hand
* Requires proper lighting
* Accuracy depends on dataset quality

---

## 🔮 Future Improvements

* Sentence-level prediction
* Deep learning (CNN/LSTM)
* Multi-hand detection
* Mobile app deployment

---

## 👨‍💻 Author

Rahul Singh

---
