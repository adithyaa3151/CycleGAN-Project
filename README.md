# CycleGAN: Unpaired Image-to-Image Translation

This project explores the **CycleGAN** model, a deep learning framework used for unpaired image-to-image translation. The goal is to translate human faces into cat/dog faces and vice versa, showcasing CycleGAN’s ability to handle diverse datasets without paired images.

## Project Overview
CycleGAN is a **Generative Adversarial Network (GAN)** designed for unpaired image translation. Unlike traditional models requiring paired training images, CycleGAN employs a **cycle-consistency loss** to learn mappings between domains.

This project aims to:
- Translate human faces into cat/dog faces and vice versa.
- Demonstrate CycleGAN’s ability to learn style transfer from unpaired datasets.
- Assess the quality of translated images using evaluation metrics.

## Objectives
- **Implement CycleGAN using PyTorch** for unpaired image-to-image translation.
- **Train the model** on a dataset of human and animal faces.
- **Evaluate the effectiveness** of CycleGAN’s style transfer capability.

## Key Goals
- Train CycleGAN on an **unpaired dataset** for image transformation.
- Optimize the model using **cycle-consistency loss**.
- Assess CycleGAN’s **generalization ability** across different datasets.

## Tools & Technologies
- **Python**: Programming language for implementation.
- **PyTorch**: Deep learning framework.
- **CycleGAN**: Model architecture for style transfer.
- **OpenCV & PIL**: Image processing libraries.
- **Matplotlib & Seaborn**: Visualization tools.

## Workflow Highlights
1. **Cloning GitHub Repository**: Fetching CycleGAN's official implementation.
2. **Installing Dependencies**: Setting up the environment for PyTorch training.
3. **Dataset Preprocessing**:
   - Organizing images into `trainA`, `trainB`, `testA`, and `testB` directories.
   - Standardizing image dimensions and formats.
4. **Model Training**:
   - Training CycleGAN on the dataset.
   - Evaluating results using loss functions.
5. **Generating Translated Images**:
   - Comparing original and generated images.

## Insights
- **CycleGAN effectively learns mappings** between human and animal faces.
- **Cycle-consistency loss prevents mode collapse**, improving translation quality.
- **Training stability is crucial**, requiring careful hyperparameter tuning.

## Dataset Details
The dataset consists of:
- **Human face images** (`trainA`, `testA`)
- **Cat/Dog face images** (`trainB`, `testB`)
- Images are resized and normalized before training.

## Getting Started
### Prerequisites
- Python 3.x
- PyTorch
- OpenCV, PIL, Matplotlib, Seaborn
- CycleGAN official repository

### Steps to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/CycleGAN_Project.git
2. Navigate to the project directory:
   ```bash
   cd CycleGAN_Project
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
4. Structure the dataset into trainA, trainB, testA, and testB directories.
5. Train the CycleGAN model:
   ```bash
   python train.py --dataset dataset_name
6. Generate translated images:
   ```bash
   python test.py --dataset dataset_name

### Results
-**Successful translation** between human and animal faces.
-**CycleGAN generalizes well** across different datasets.
-**Generated images maintain identity preservation** while transferring styles.
