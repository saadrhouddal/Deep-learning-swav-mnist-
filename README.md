SwAV on MNIST with only 100 labels

Deep Learning class project — reproduction of SwAV (Caron et al., NeurIPS 2020) for semi-supervised digit classification.

Goal

Classify MNIST digits using only 100 labeled images (10 per digit) out of 60,000.

Approach
Self-supervised pretraining (no labels): a CNN is trained with SwAV on the 60,000 train images. Each image is augmented into several views (multi-crop), and the network learns to predict the cluster (prototype) of one view from another view (swapped prediction). Sinkhorn-Knopp keeps the clusters balanced and prevents collapse.
Evaluation with 100 labels: linear probe, kNN, and fine-tuning on the learned features, tested on the 10,000 test images.
Comparison with baselines that only see the 100 labels (kNN on pixels, PCA + logistic regression, supervised CNN), and with a CNN trained on all 60,000 labels (upper bound).
Repository structure
preprocessing.py               shared data preprocessing (loading, normalization, 100-label split)
labeled_indices_seed42.json    the 100 labeled indices (same for everyone)
notebooks/                     experiments
src/                           models, augmentations, SwAV, evaluation
results/                       figures and result tables
Quick start
bash
pip install -r requirements.txt
python preprocessing.py        # downloads MNIST, prints the split checksum
python
import sys; sys.path.append("..")
from preprocessing import get_data, set_seed

set_seed()
data = get_data()   # X_unlabeled, X_lab / y_lab (100), X_test / y_test

All experiments must use preprocessing.py so that everyone works on the same 100 labeled images.
