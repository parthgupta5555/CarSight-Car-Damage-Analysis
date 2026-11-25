# CarSight --- Intelligent Car Damage Analysis

```{=html}
<p align="center">
```
`<strong>`{=html}A computer-vision proof of concept for classifying
visible car condition from images.`</strong>`{=html}
```{=html}
</p>
```
CarSight is a deep-learning application that analyzes an uploaded car
image and predicts one of six front/rear condition categories. It also
presents class-wise prediction scores and a Grad-CAM visualization to
help inspect which image regions influenced a selected model output.

The project combines model experimentation in PyTorch with an
interactive Streamlit application.

**GitHub Repository:** [https://github.com/parthgupta5555/CarSight-Car-Damage-Analysis](https://github.com/parthgupta5555/CarSight-Car-Damage-Analysis)

------------------------------------------------------------------------

## Contents

-   [Project Overview](#project-overview)
-   [Application Screenshots](#application-screenshots)
-   [Key Features](#key-features)
-   [Classification Categories](#classification-categories)
-   [How It Works](#how-it-works)
-   [Model Development](#model-development)
-   [Evaluation Snapshot](#evaluation-snapshot)
-   [Grad-CAM Explainability](#grad-cam-explainability)
-   [Project Structure](#project-structure)
-   [Installation](#installation)
-   [Run CarSight](#run-carsight)
-   [Technologies](#technologies)
-   [Limitations and Future
    Improvements](#limitations-and-future-improvements)

------------------------------------------------------------------------

## Project Overview

The goal of CarSight is to build a proof of concept (POC) that
classifies the visible condition of a car's **front or rear** using an
image.

The model predicts one of six predefined classes:

-   Front Breakage
-   Front Crushed
-   Front Normal
-   Rear Breakage
-   Rear Crushed
-   Rear Normal

The supplied project specification sets a target of at least 75%
classification accuracy and calls for a trained model integrated into a
Streamlit image-upload application.

> **Scope:** CarSight is an image-classification POC. It predicts an
> image-level category; it does not identify exact damaged pixels,
> estimate repair costs, or determine the full physical extent of
> damage.

## Application Screenshots

### CarSight Interface

![CarSight Homepage](screenshots/homepage.png)

### Car Damage Prediction

![Car Damage Prediction Results](screenshots/prediction.png)

### Grad-CAM Visual Explainability

![Grad-CAM Visualization](screenshots/grad-cam.png)

## Key Features

  -----------------------------------------------------------------------
  Capability                          Description
  ----------------------------------- -----------------------------------
  Image input                         Upload a car image in JPG, JPEG, or
                                      PNG format

  Six-class prediction                Classify the visible front/rear
                                      condition

  Prediction scores                   Compare the model's softmax score
                                      across all six categories

  Visual explainability               Generate a Grad-CAM overlay for the
                                      predicted or another selected class

  GPU/CPU inference                   Automatically use CUDA when
                                      available; otherwise use CPU

  Interactive interface               Explore the analysis through a
                                      Streamlit application
  -----------------------------------------------------------------------

## Classification Categories

    Class index Category         Meaning
  ------------- ---------------- ----------------------------------
              0 Front Breakage   Front view --- breakage
              1 Front Crushed    Front view --- crushed condition
              2 Front Normal     Front view --- normal condition
              3 Rear Breakage    Rear view --- breakage
              4 Rear Crushed     Rear view --- crushed condition
              5 Rear Normal      Rear view --- normal condition

The class order is important: it follows the alphabetical folder order
used by `torchvision.datasets.ImageFolder` and must remain consistent
with the trained checkpoint.

## How It Works

``` text
       Car image (JPG / JPEG / PNG)
                    |
                    v
          Image preprocessing
          Resize to 224 x 224
          Normalize (ImageNet stats)
                    |
                    v
          Fine-tuned ResNet50
                    |
                    v
             Six class scores
                    |
          +---------+----------+
          |                    |
          v                    v
     Top prediction       Class-wise scores
          |                    |
          +---------+----------+
                    |
                    v
             Streamlit results
                    |
                    v
           Grad-CAM explanation
```

1.  The user uploads a car image through the Streamlit interface.
2.  The image is converted to RGB, resized to 224 × 224, converted to a
    tensor, and normalized.
3.  The trained ResNet50 classifier produces six output scores.
4.  Softmax converts the outputs into scores that can be compared across
    the six classes.
5.  The application displays the predicted category, top score,
    inference device, and class-wise score chart.
6.  Grad-CAM can generate a heatmap overlay for the predicted category
    or a class selected by the user.

## Model Development

Model development and experimentation were carried out in PyTorch.

### Architectures explored

The training notebook includes experiments with:

-   **Custom CNN:** a convolutional network built from convolution,
    ReLU, and max-pooling layers.
-   **Regularized CNN:** a custom CNN variant incorporating batch
    normalization and dropout.
-   **EfficientNet-B0:** a pretrained architecture used for
    transfer-learning experimentation.
-   **ResNet50:** a pretrained architecture fine-tuned for the six
    car-condition classes.

The deployed application uses the **ResNet50** checkpoint. The other
architectures represent development experiments, not additional models
running inside the app.

### Deployed ResNet50 configuration

  -----------------------------------------------------------------------
  Component                           Configuration
  ----------------------------------- -----------------------------------
  Framework                           PyTorch

  Backbone                            ResNet50

  Initialization                      ImageNet-pretrained weights during
                                      training

  Fine-tuning                         Layer 4 unfrozen; earlier backbone
                                      layers frozen

  Classification head                 Dropout (0.2) + Linear layer (2048
                                      → 6)

  Loss function                       Cross-entropy loss

  Optimizer                           Adam

  Final training run settings         Learning rate 0.005; dropout 0.2
  recorded in notebook                

  Input resolution                    224 × 224

  Output classes                      6
  -----------------------------------------------------------------------

The inference helper reconstructs the same ResNet50 architecture with
`weights=None` and loads the trained `saved_model.pth` checkpoint. This
avoids downloading pretrained weights again at inference time.

### Image preprocessing

The inference pipeline uses:

-   Resize to `(224, 224)`
-   Convert image to tensor
-   Normalize with ImageNet mean: `[0.485, 0.456, 0.406]`
-   Normalize with ImageNet standard deviation: `[0.229, 0.224, 0.225]`

The training notebook also applies random horizontal flipping, random
rotation, and color jitter as augmentation.

### Hyperparameter experimentation

Optuna was used for a 20-trial search over learning rate and dropout
rate. The notebook records a best trial with a learning rate of
approximately `0.002107` and dropout of approximately `0.4451`. The
final training run documented separately in the training notebook used a
learning rate of `0.005` and dropout of `0.2`.

These are experiment records; the final checkpoint and its matching
model definition are the source of truth for inference.

## Evaluation Snapshot

The training notebook reports the following classification report for
**575 validation samples**:

  Metric                     Score
  ------------------- ------------
  Accuracy              **79.83%**
  Macro precision             0.80
  Macro recall                0.79
  Macro F1-score              0.79
  Weighted F1-score           0.80

### Per-class results

The report uses the six class indices in the order listed above.

  Category           Precision   Recall   F1-score   Support
  ---------------- ----------- -------- ---------- ---------
  Front Breakage          0.82     0.82       0.82       113
  Front Crushed           0.80     0.73       0.76       111
  Front Normal            0.84     0.93       0.89       105
  Rear Breakage           0.75     0.79       0.77        82
  Rear Crushed            0.70     0.78       0.74        88
  Rear Normal             0.90     0.70       0.79        76

**Important evaluation note:** these figures are validation results from
the supplied notebook, not results from a separate untouched test set.
The notebook creates a random 75/25 split from an `ImageFolder` dataset
that has random augmentation in its transform. Consequently, the
validation subset also receives random augmentation. A stronger
evaluation should use separate training and validation
datasets/transforms, a reproducible split, and an untouched test set.
Treat the reported score as an experimental validation snapshot rather
than a guarantee of real-world performance.

## Grad-CAM Explainability

CarSight integrates Grad-CAM using the final convolutional block of
ResNet50 (`layer4[-1]`).

The application lets the user select one of the six categories and view:

-   The original uploaded image
-   A Grad-CAM heatmap overlay for the selected class

Grad-CAM highlights regions that influence a class score. It is useful
for inspecting model behavior and spotting possible reliance on
backgrounds or unrelated visual cues.

> Grad-CAM is an interpretation aid, not a pixel-level damage
> segmentation mask. A heatmap does not prove that the model has learned
> the correct visual concept or establish the exact physical extent of
> damage.

## Project Structure

Keep the following runtime files in the same directory:

``` text
CarSight/
├── app.py                  # Streamlit user interface
├── model_helper.py         # Model architecture, loading, preprocessing, prediction
├── saved_model.pth         # Trained ResNet50 state-dict checkpoint
├── README.md
├── screenshots/
│   ├── homepage.png
│   ├── prediction.png
│   └── grad-cam.png
│
├── damage_prediction.ipynb       # Model training and evaluation experiments
├── hyperparameter_tunning.ipynb  # Optuna hyperparameter search
└── gradcam_car_damage.ipynb      # Grad-CAM experiment
```

The notebook filenames above describe their roles; use the actual
filenames in your local repository if they differ.

## Installation

### 1. Clone the repository

``` bash
git clone https://github.com/parthgupta5555/CarSight-Car-Damage-Analysis.git
cd CarSight-Car-Damage-Analysis
```

### 2. Create and activate a virtual environment (recommended)

**Windows --- Command Prompt**

``` cmd
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
python -m pip install torch torchvision streamlit pillow pandas plotly grad-cam
```

Install a PyTorch build appropriate for your machine from the official
PyTorch installation selector if you need a specific CUDA-enabled build.
CPU inference is supported as well.

## Run CarSight

Make sure `app.py`, `model_helper.py`, and `saved_model.pth` are in the
same directory, then run:

``` bash
python -m streamlit run app.py
```

Streamlit will print a local URL in the terminal (usually
`http://localhost:8501`). Open it in your browser.

### Using the application

1.  Upload a JPG, JPEG, or PNG car image.
2.  Select **Analyze car image**.
3.  Review the predicted category and top prediction score.
4.  Compare the six class-wise scores.
5.  In the Grad-CAM section, choose a class and select **Generate
    Grad-CAM**.
6.  Compare the original image with the heatmap overlay.

### Checkpoint requirement

The application expects a file named:

``` text
saved_model.pth
```

Place the trained checkpoint beside `app.py`. The checkpoint must match
the ResNet50 architecture, six-class ordering, and classifier-head
configuration used in `model_helper.py`.

## Technologies

-   **Python** --- application and model development
-   **PyTorch** --- deep-learning model training and inference
-   **Torchvision** --- pretrained ResNet50, image transforms, and
    dataset utilities
-   **Optuna** --- hyperparameter search
-   **Grad-CAM** --- visual explainability
-   **Streamlit** --- interactive application
-   **Pandas** --- score-table preparation
-   **Plotly** --- class-wise prediction visualization
-   **Pillow** --- image loading and conversion
-   **scikit-learn** --- classification metrics and confusion matrix in
    evaluation

## Limitations and Future Improvements

Potential next steps:

-   Build separate training and validation transforms so validation
    images are not randomly augmented.
-   Use a fixed random seed and stratified splitting to improve
    reproducibility and class balance.
-   Evaluate on a separate, untouched test set and report per-class
    metrics.
-   Inspect confusion-matrix errors, especially between breakage and
    crushed categories.
-   Compare candidate architectures using the same data split and
    evaluation protocol.
-   Test on varied real-world images, lighting conditions, viewpoints,
    and backgrounds.
-   Consider object detection or segmentation if the product requirement
    expands from image-level classification to locating damaged regions.

## Dataset

The dataset was provided as part of the guided Deep Learning
project by Codebasics.

It contains images belonging to six car-condition categories.
The original dataset is not included in this repository.

Please refer to the original course materials for dataset access.

------------------------------------------------------------------------

## Project Note

CarSight demonstrates an end-to-end computer-vision workflow: dataset
preparation, model experimentation, transfer learning, hyperparameter
search, evaluation, inference, and visual explainability through a
Streamlit interface.
