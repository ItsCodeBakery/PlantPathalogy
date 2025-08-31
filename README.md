
# Plant Pathology - Self-Supervised Learning 

This repository contains the complete implementation of our research paper on Scalable Plant Disease Detection using a hybrid learning framework that combines SimCLR-based contrastive pretraining with a custom CNN classifier. The proposed model first learns general-purpose visual representations in a self-supervised manner, and then fine-tunes these features for plant disease classification with minimal supervision. It supports dual operational modes—projection for SimCLR and classification for diagnosis—allowing for flexible experimentation. Our approach achieves high performance across diverse plant disease classes and includes robust visual analysis via Grad-CAM, t-SNE, ROC-AUC, and confusion matrix visualizations. This codebase is fully modular and intended for researchers, practitioners, and students exploring self-supervised learning and agricultural AI.


##  Project Structure

```
Code/
├── SimCLR_CNNClassifier.py         # Combined CNN model with SimCLR projection + classification head
├── data_loader.py                  # Data loading and augmentation (ImageFolder-based)
├── finetune_classifier.py          # Optional separate CNN model for classification only
├── generate_tsne.py                # t-SNE visualization of learned embeddings
├── gradcam.py                      # Grad-CAM visualization for model interpretability
├── plot_loss_accuracy_curves.py    # Training/validation loss and accuracy plots
├── plot_multiclass_roc.py          # Multi-class ROC and AUC curve plotting
├── test.py                         # Model evaluation and metric collection
├── train_model.py                  # Supervised training loop for classification
├── utils.py                        # Seed setting, accuracy calculation, and helpers
```

---

##  Model Overview

`SimCLR_CNNClassifier.py` defines a unified architecture:
- `mode='pretrain'`: Enables **SimCLR** contrastive learning using a projection head.
- `mode='classification'`: Switches to a **fully supervised classification head**.

Use `.set_mode('pretrain')` and `.set_mode('classification')` as needed.

---

##  Workflow

###  1. Setup

```bash
pip install -r requirements.txt
```

###  2. Data Format

Organize your dataset like this:
```
/data/
  ├── train/
  │     ├── class1/
  │     ├── class2/
  └── test/
        ├── class1/
        ├── class2/
```

###  3. Training


# Supervised fine-tuning
python train_model.py


###  4. Evaluation


python test.py


This will:
- Print classification report
- Save confusion matrix
- Save ROC-AUC plots
- Save t-SNE plot

---

##  Visualization Tools

| Module | Description |
|--------|-------------|
| `plot_loss_accuracy_curves.py` | Dual-axis plot of loss and accuracy |
| `plot_multiclass_roc.py`      | ROC-AUC curve for each class and macro average |
| `generate_tsne.py`            | t-SNE of final embeddings |
| `gradcam.py`                  | Grad-CAM attention visualization on test images |
| `utils.py`                    | Utility functions (seed, accuracy, etc.) |

---

##  Key Features

-  Self-supervised learning via **SimCLR projection head**
-  Custom CNN architecture
-  Classification with full visualization suite
-  Modular design for ease of extension
-  Plots are saved in `plots/` (ensure the directory exists)
-  Feel free to contact me if you face any problem in running the code. 

---


