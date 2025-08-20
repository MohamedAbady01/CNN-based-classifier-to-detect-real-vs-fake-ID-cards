# Card Classifier (Real vs Fake ID Detection)

This project detects **real vs fake ID cards** by analyzing the **face region** and extracting **texture + edge features**.  
It uses **YOLO (OpenCV DNN)** for face detection, **LBP + edges** for feature enhancement, and a **PyTorch CNN** for classification.

---

## 🚀 Features
- Face detection & cropping using YOLO
- Extracts **RGB + LBP + Edge** → 9-channel input
- CNN classifier with dropout & batch normalization
- Training with early stopping & learning rate scheduling
- Misclassified images are saved for inspection
- Confusion matrix & classification report

---

## 📂 Project Structure
├── models/
│ ├── face_doc_1623.cfg
│ ├── face_doc_1623.weights
│ └── face_doc_1623.names
├── dataset/
│ ├── real/ # Real ID images
│ └── fake/ # Fake ID images
├── card_classifier.py # Main training & evaluation script
├── card_classifier_best.pth # Saved model (after training)
├── training_plot_improved.png # Training loss & accuracy
├── confusion_matrix.png # Confusion matrix plot
├── requirements.txt
└── README.md


---

## ⚙️ Installation

### 1. Clone repository
```bash
git clone https://github.com/MohamedAbady01/CNN-based-classifier-to-detect-real-vs-fake-ID-cards.git
cd CNN-based-classifier-to-detect-real-vs-fake-ID-cards
```
2. Install dependencies
pip install -r requirements.txt

3. Prepare dataset

Put your dataset in the following structure:

dataset/
 ├── real/
 └── fake/

4. Run training
python card_classifier.py


This will:

Train CNN

Save best model → card_classifier_best.pth

📊 Evaluation

After training, evaluation will:

Generate confusion matrix (confusion_matrix.png)

Print classification report

Save misclassified images → misclassified_images/

📈 Example Output

Training Loss & Accuracy plot

Confusion Matrix

Misclassified image samples

🔮 Next Steps

Experiment with deeper CNN architectures (ResNet, EfficientNet)

Add data augmentation (rotation, brightness, noise)

Deploy model as a REST API (FastAPI/Flask)

👨‍💻 Author

Developed by Mohamed Abady
