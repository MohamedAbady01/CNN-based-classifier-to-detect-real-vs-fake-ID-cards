import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline
import time
import copy
from skimage.feature import local_binary_pattern
from torchvision.utils import save_image 
from skimage import img_as_ubyte
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# Set the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the pre-trained face detection model
face_net = cv2.dnn.readNetFromDarknet("./models/face_doc_1623.cfg", "./models/face_doc_1623.weights")

# Class labels
classes = None
with open("./models/face_doc_1623.names", 'r') as f:
    classes = [line.strip() for line in f.readlines()]
def crop_face_area(image_path, face_net, margin=0.2):
    """
    Crop the region of interest (ROI) around the detected face with a margin.
    """
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    face_net.setInput(blob)
    outs = face_net.forward(get_output_layers(face_net))

    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    # Process each detection and filter out low confidence faces
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # If a face is detected, crop it with margin
    if len(indices) > 0:
        i = indices[0]
        if isinstance(i, (list, np.ndarray)):
            i = i[0]
        box = boxes[i]
        x, y, w, h = box

        x_margin = int(w * margin)
        y_margin = int(h * margin)

        x1 = max(0, x - x_margin)
        y1 = max(0, y - y_margin)
        x2 = min(width, x + w + x_margin)
        y2 = min(height, y + h + y_margin)

        cropped_face = image[y1:y2, x1:x2]
        return cropped_face
    else:
        return None
def get_output_layers(net):
    """
    Extract the output layers from the network.
    """
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def extract_lbp(image):
    """
    Extract Local Binary Pattern (LBP) features from the image and resize to match the input image size.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_resized = cv2.resize(lbp.astype(np.uint8), (image.shape[1], image.shape[0]))  # Resize to match image size
    lbp_resized = np.expand_dims(lbp_resized, axis=-1)  # Add channel dimension
    lbp_resized = np.repeat(lbp_resized, 3, axis=-1)  # Repeat channels to match RGB format
    return lbp_resized

def extract_edge_features(image):
    """
    Extract edge features from the image using Canny edge detector and resize to match the input image size.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_resized = cv2.resize(edges, (image.shape[1], image.shape[0]))  # Resize to match image size
    edges_resized = np.expand_dims(edges_resized, axis=-1)  # Add channel dimension
    edges_resized = np.repeat(edges_resized, 3, axis=-1)  # Repeat channels to match RGB format
    return edges_resized
class CardDataset(Dataset):
    """
    Dataset class for loading card images, applying transformations, and extracting face region.
    """
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Load images from 'real' and 'fake' folders
        for label, folder in enumerate(['real', 'fake']):
            paths = glob.glob(os.path.join(root_dir, folder, '*.*'))
            self.image_paths += paths
            self.labels += [label] * len(paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Crop face from image
        face_crop = crop_face_area(img_path, face_net)
        if face_crop is None:
            face_crop = np.zeros((128, 128, 3), dtype=np.uint8)

        # Resize the face region to 128x128 pixels
        face_crop = cv2.resize(face_crop, (128, 128))
        image = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        # Apply transformations if defined
        if self.transform:
            image = self.transform(image)

        # Convert PyTorch tensor to NumPy array for concatenation
        image = image.permute(1, 2, 0).numpy()  # Change from [C, H, W] to [H, W, C]

        # Extract LBP and edge features
        lbp_features = extract_lbp(face_crop)
        edge_features = extract_edge_features(face_crop)

        # Ensure all feature arrays are in the same format
        #print("Image shape:", image.shape)
        #print("LBP shape:", lbp_features.shape)
        #print("Edge Features shape:", edge_features.shape)

        # Concatenate original image, LBP, and edge features
        image = np.concatenate([image, lbp_features, edge_features], axis=-1)

        # Convert back to PyTorch tensor and apply normalization if required
        image = torch.tensor(image).float().permute(2, 0, 1)  # Change from [H, W, C] to [C, H, W]

        return image, label
class CardClassifier(nn.Module):
    """
    Convolutional Neural Network for classifying card images (real vs fake).
    """
    def __init__(self, num_classes=2):
        super(CardClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=3, padding=1),  # 9 input channels (RGB + LBP + Edge)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
def train():
    """
    Training loop to train the CardClassifier model.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = CardDataset(root_dir='./dataset', transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = CardClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    epochs = 20
    best_acc = 0.0
    best_los = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    early_stop_patience = 5
    patience_counter = 0

    all_train_loss = []
    all_val_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        all_train_loss.append(avg_train_loss)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        all_val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f} - Val Acc: {val_accuracy:.2f}%")

        scheduler.step()

        # Early stopping
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            best_los = avg_train_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered.")
                break

    # Save best model
    model.load_state_dict(best_model_wts)
    model_path = "card_classifier_best.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Training complete. Best Val Accuracy: {best_acc:.2f}%")
    print(f"Training complete. Best Loss: {best_los:.4f}")

    # Optional: Plot training loss and accuracy
    plt.figure(figsize=(12, 6))

    # Plot Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(all_train_loss, label="Train Loss", color='red', linewidth=2)
    plt.title("Training Loss", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()

    # Plot Validation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(all_val_accuracies, label="Validation Accuracy", color='green', linewidth=2)
    plt.title("Validation Accuracy", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.tight_layout()

    plt.suptitle("Training Process", fontsize=16)
    plt.savefig("training_plot_improved.png")
    plt.show()

    return all_train_loss, all_val_accuracies , model , val_loader , model_path

def evaluate_model(model, val_loader, device, misclassified_save_dir="misclassified_images"):
    model.eval()
    all_preds = []
    all_labels = []

    os.makedirs(misclassified_save_dir, exist_ok=True)
    try:
      with torch.no_grad():
          for batch_idx, (images, labels) in enumerate(val_loader):
              images = images.to(device)
              labels = labels.to(device)
              outputs = model(images)
              _, preds = torch.max(outputs, 1)

              all_preds.extend(preds.cpu().numpy())
              all_labels.extend(labels.cpu().numpy())
              real = "Real"
              fake = "Fake"
              for i in range(images.size(0)):
                  if preds[i] != labels[i]:
                      # Save misclassified image directly
                      if labels[i].item() == 0:
                          img_filename = f"_img{i}_true is {fake}_pred is {real}.png"
                      else:    
                        img_filename = f"_img{i}_true is {real}_pred is {fake}.png"
                      img_path = os.path.join(misclassified_save_dir, img_filename)
                      img_rgb = images[i][:3]  
                      save_image(img_rgb, img_path, normalize=True)

      # Report and plot
      cm = confusion_matrix(all_labels, all_preds)
      print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=["Real", "Fake"]))
      print("===============================================================================================")

      plt.figure(figsize=(6, 5))
      sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
      plt.xlabel('Predicted Label')
      plt.ylabel('True Label')
      plt.title('Confusion Matrix')
      plt.tight_layout()
      plt.savefig("confusion_matrix.png")
      plt.show()
    except Exception as e:
      print(f"Image {i} shape:", images[i].shape)
      raise

def ConfusionMatrix(model, val_loader, model_path, device,misclassified_save_dir="misclassified_images"):
  model.load_state_dict(torch.load(model_path))
  evaluate_model(model, val_loader, device,misclassified_save_dir)
if __name__ == "__main__":
    try:
        print(" Wait For Training.......")
        all_train_loss , all_val_accuracies , model , val_loader , model_path = train()
    except:
        print("Error On Training ")
    print("Training process completed.",model)
    try:
        print("===================================Model Evaluation ============================================================")
        print("Confusion Matrix:")
        ConfusionMatrix(model, val_loader, model_path, device,misclassified_save_dir="misclassified_images")
        print("===============================================================================================")
    except Exception as e:
        print("Error On Model Evaluation Process",e)