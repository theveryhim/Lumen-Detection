
import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image
import zipfile, os, shutil
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np

# Define both models
# Load both models at script start
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SqueezeNet
class SqueezeNetBinary(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.squeezenet1_1(pretrained=False)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(512, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.backbone.classifier(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

squeezenet_model = SqueezeNetBinary().to(device)
squeezenet_model.load_state_dict(torch.load("Models/SqueezeNet.pth", map_location=device))
squeezenet_model.eval()

mobilenet_model = models.mobilenet_v3_small(weights=True)
mobilenet_model.classifier[3] = nn.Linear(mobilenet_model.classifier[3].in_features, 1)
mobilenet_model = mobilenet_model.to(device)
mobilenet_model.load_state_dict(torch.load("Models/mobilenetv3small_bce_adam.pth", map_location=device))
mobilenet_model.eval()

# Model map for routing
model_map = {
    "SqueezeNet(700K params)": squeezenet_model,
    "MobileNetV3-Small(1.5M params)": mobilenet_model
}

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

def infer_image(img, model_choice):
    model = model_map[model_choice]
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(img)
        prob = torch.sigmoid(logit).item()
    label = "Lumen" if prob > 0.5 else "No Lumen"
    return {label: round(prob, 4), "Lumen" if label == "No Lumen" else "No Lumen": round(1 - prob, 4)}

def infer_batch(zip_file, model_choice):
    extract_dir = "temp_data"
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_file.name, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    dataset = datasets.ImageFolder(extract_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    model = model_map[model_choice]

    y_true, y_pred, y_scores = [], [], []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.numpy()
            logits = model(imgs).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()

            preds = (probs > 0.5).astype(int)
            y_true.extend(labels)
            y_pred.extend(preds)
            y_scores.extend(probs)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_scores)
    except:
        auc = float('nan')
    shutil.rmtree(extract_dir)
    return { "Accuracy": round(acc, 4), "F1 Score": round(f1, 4), "AUC": round(auc, 4) }

# Gradio UI
image_input = gr.Image(type="pil", label="Upload Image")
zip_input = gr.File(file_types=[".zip", ".rar"], label="Upload .zip or .rar")

with gr.Blocks() as demo:
    gr.Markdown("## 🔬 Lumen Detection Page")
    gr.Markdown("Select a model --- upload a single image or a dataset ZIP with `Lumen` and `NotLumen` folders.")

    with gr.Row():
        with gr.Column():
            model_dropdown = gr.Dropdown(
                choices=["SqueezeNet(700K params)", "MobileNetV3-Small(1.5M params)"],
                value="SqueezeNet(700K params)",
                label="Choose Model"
            )
            image_input = gr.Image(type="pil", label="Upload Image")
            single_output = gr.Label(label="Single Image Prediction")
            image_btn = gr.Button("Classify Image")
            image_btn.click(fn=infer_image, inputs=[image_input, model_dropdown], outputs=single_output)

        with gr.Column():
            zip_input = gr.File(file_types=[".zip"], label="Upload .zip Dataset")
            batch_output = gr.JSON(label="Batch Evaluation Metrics")
            zip_btn = gr.Button("Evaluate ZIP Dataset")
            zip_btn.click(fn=infer_batch, inputs=[zip_input, model_dropdown], outputs=batch_output)


if __name__ == "__main__":
    demo.launch(share=True)
