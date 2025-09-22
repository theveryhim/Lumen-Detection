# Lumen Detection

Lumen detection in colonoscopy images refers to the identification and location of the central region of the intestinal tract (lumens) in colonoscopy images. This is done to correct guidance, improve the quality of imaging, and
It is very important to avoid navigational errors. Also, accurate detection of lumens can be effective in increasing the accuracy of lesion detection and improving the segmentation of polyps.

In this repo, we represent our work on colonoscopy images by light weight models 
as binary classifiers

*Due to confidentiality matters, we are unable to share data set directly here.
If you need the acces contact any of us:
*
(Arman Yazdani)[]

## SqueezeNet
We trained **SqueezeNet v1.1** from scratch with a custom lightweight classification head:

- Final classifier:  
  `Conv2D(512 → 32) → BatchNorm → ReLU → AdaptiveAvgPool2d → Linear(32 → 1)`
- Output is passed to `BCEWithLogitsLoss` for binary classification.
- Trainable params: `739,009`
<p align="center">
    <img src="Images/1.png" alt="Descriptive Alt Text" class="fit-width-image">
</p>



## MobileNetV3small

We Fine-tuned MobileNetV3 Small, a lightweight convolutional neural network, pre-trained on ImageNet and fine-tuned for our binary classification task. Final layer adapted to output a single logit → Binary classification with BCEWithLogitsLoss

*   **Model:** MobileNet V3 small
*   **Weights:** Pretrained
*   **Trainable params:** `1,518,881`
*   **Loss Function:** BCEWithLogitsLoss for binary output
*   **Optimizer:**
    -  Adam optimizer with learning rate 1e-4
    -  Learning rate scheduling using ReduceLROnPlateau
    -  Early stopping based on validation loss
<p align="center">
    <img src="Images/2.png" alt="Descriptive Alt Text" class="fit-width-image">
</p>

## Gradio UI 😊

We also prepaired a user friendly interface where you can uplaod your colonoscpy image or data set and see our models inference
<p align="center">
    <img src="Images/Gradio.gif" alt="Descriptive Alt Text" class="fit-width-image">
</p>


Here's how to use it on your local host:
```bash
git clone https://github.com/theveryhim/Lumen-Detection.git your-directory
cd ...\your-directory\Lumen-Detection
python app.py
```
then follow to the given link and tadaa!


# Lumen detection+
Here we try to Improve our models to their best performance for lumen detection through systematic optimization and new training techniques. Further more, We focus on balancing accuracy with computational efficiency for practical deployment.
## SqueezeNet
Let's continue with the smaller/faster model!
### Hyperparameter Optimization: Ray
Used Ray Tune with ASHA scheduler for Bayesian optimization. Searched loguniform spaces for `lr (1e-4 to 1e-1)`, `weight_decay (1e-6 to 1e-2)`, and `batch_size [16,32,64]`. Maximized `val_acc` over 10 trials.
<p align="center">
    <img src="Images/3.png" alt="Descriptive Alt Text" class="fit-width-image">
    <figcaption> Hyperparameter optimization with Ray</figcaption>
</p>

### Training: Curriculum Learning
Our strategy combines a fixed curriculum of **Progressive Data Deblurring** with the **MixUp** regularization technique, executed using the previously optimized hyperparameters.
<p align="center">
    <img src="Images/4.png" alt="Descriptive Alt Text" class="fit-width-image">
    <figcaption> Enahnced models performance</figcaption>
</p>

### Production Prototype: ONNX
- Exported optimized models to ONNX format for production deployment.
<p align="center">
    <img src="Images/5.png" alt="Descriptive Alt Text" class="fit-width-image">
    <figcaption> ONNX/Enhanced models performance</figcaption>
</p>

```markdown
Avg PyTorch inference time per batch: 98.930 ms
Avg ONNXRuntime inference time per batch: 20.431 ms
```

- Employed quantization techniques for potential edge deployment scenarios.
```markdown
PyTorch int8 inference time per batch (CPU): 103.658 ms
ONNX int8 inference time per batch: 55.911 ms
```
```markdown
Evaluating models...
PyTorch INT8 Accuracy: 0.8627, F1_torch: 0.8574, AUC_torch: 0.8625
ONNX INT8 Accuracy: 0.8513, F1_onnx: 0.8465, AUC_onnx: 0.8513
```
## MobileNetV3small
In this phase, we've done below tasks:

1. Realistic colonoscopy-specific augmentations:
     Geometric (flip, rotation, affine transformations)
     Photometric (lighting variations, contrast, saturation)
     Noise and blur simulation to mimic motion artifacts
   
3. Systematic hyperparameter search for stable training:
    Used Ray Tune with ASHA scheduler for Bayesian optimization.
    BCE Loss:
    search_space = {
    "lr": tune.loguniform(1e-5, 1e-3),
    "batch_size": tune.choice([16, 32, 64]),
    "optimizer": tune.choice(["AdamW", "SGD"]),
    "weight_decay": tune.loguniform(1e-6, 1e-2),
    "momentum": tune.uniform(0.8, 0.99)
    }
   
    Focal Loss:
    
  
4. Loss function comparison:
    BCE Loss – standard binary classification.
    Focal Loss (α=0.5) – emphasizes hard-to-classify samples.
   
6. Hard example mining to focus on difficult cases.

The overall workflow:

<img width="893" height="1021" alt="image" src="https://github.com/user-attachments/assets/5a810a5f-b02d-4ae9-ba73-9ae49a50c3a7" />




### A Bigger Stride: Knowledge Distillation
Leveraged MobileNetV3 teacher to enhance SqueezeNet student performance. Combined hard and soft loss for effective knowledge transfer.
<p align="center">
    <img src="Images/6.png" alt="Descriptive Alt Text" class="fit-width-image">
    <figcaption> Student(SqueezeNet) model's performance after knowledge distillation</figcaption>
</p>

```markdown
Test Accuracy: 0.8738, F1: 0.8501, AUC: 0.9291
```
