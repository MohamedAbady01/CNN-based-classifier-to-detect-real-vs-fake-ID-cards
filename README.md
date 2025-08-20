# Card Classifier (Real vs Fake ID Detection)

This project was developed as an **Upwork freelance project**, where I successfully delivered the solution and received **⭐ 5-star feedback** from the client.  

⚠️ **Important note**: The client required that **no pre-trained models** (e.g., ResNet, VGG, EfficientNet) be used.  
All the work was built **from scratch**, including:
- Custom dataset handling
- Feature extraction (LBP + Edge)
- CNN model architecture
- Training & evaluation pipeline

---

## 🚀 Features
- Face detection & cropping using YOLO (OpenCV DNN)
- Extracts **RGB + LBP + Edge** → 9-channel input
- CNN classifier built completely from scratch (no pretrained backbone)
- Training with early stopping & learning rate scheduling
- Misclassified images are saved for inspection
- Confusion matrix & classification report
- Delivered successfully with 5-star client rating ⭐⭐⭐⭐⭐

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
```bash
pip install -r requirements.txt
```
4. Prepare dataset

Put your dataset in the following structure:

dataset/
 ├── real/
 └── fake/

4. Run training
 ```bash
python card_classifier.py
```

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
