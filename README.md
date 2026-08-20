Due to the file size limit for upload, the original absorbance file has been split into 4 separate files. Before conducting the model training, these 4 absorbance datasets need to be merged into a single xlsx file, and then the model training can be performed.

# SpectralExpertPINN

A deep-learning framework for simultaneous qualitative identification and quantitative prediction of groundwater contaminants using UV–Vis absorption spectra.

## Overview

This repository implements a multi-expert spectral analysis framework that combines:

* Convolutional Neural Networks (CNN)
* Transformer Encoder
* Physics-Informed Neural Networks (PINNs)

for the detection and quantification of:

* Heavy Metals (HMs)

  * As
  * Cr
  * Pb

* PFASs

  * PFBA
  * PFOA
  * PFOS

The framework performs:

1. Multi-label contaminant classification
2. Quantitative concentration prediction
3. Physics-constrained spectral reconstruction
4. Expert-model routing for different contaminant groups

---

## Model Architecture

### Classification Expert

CNN → Transformer → MLP

Predicts contaminant presence/absence.

### Heavy Metal Expert

CNN → Transformer → PINN Decoder

Predicts concentrations of heavy metals while enforcing spectral consistency.

### PFAS Expert

Lightweight CNN → Transformer → PINN Decoder

Designed for weak spectral signatures and low-concentration PFAS compounds.

---

## Key Features

* Multi-task learning framework
* Expert model architecture
* Physics-informed loss functions
* Spectral reconstruction constraints
* Automatic contaminant routing
* UV–Vis spectral analysis
* Groundwater quality assessment

---

## Dataset Format

### Spectral Data

Excel file containing absorbance spectra:

```text
Sample_ID | λ1 | λ2 | λ3 | ... | λn
```

### Concentration Data

```text
Sample_ID | As | Cr | Pb | PFBA | PFOA | PFOS
```

---

## Training

```bash
python main.py
```

The workflow includes:

1. Data preprocessing
2. Savitzky–Golay derivative filtering
3. Data normalization
4. Expert model training
5. Early stopping
6. Quantitative evaluation
7. Visualization generation

---

## Evaluation Metrics

### Classification

* Accuracy
* Confusion Matrix

### Regression

* R²
* RMSE
* MAE
* Relative Error (RE)

---

## Output

The framework automatically generates:

* Model checkpoints (.pth)
* Prediction results (.csv)
* Relative error statistics (.xlsx)
* Regression scatter plots
* Classification confusion matrices
* Error boxplots

---

## Applications

* Groundwater monitoring
* Environmental pollution assessment
* PFAS screening
* Heavy metal detection
* Water quality management
* Spectroscopy-based sensing

---

## Citation

If you use this code in your research, please cite the corresponding publication.

```bibtex
@article{Deep Learning for Rapid Analysis of PFAS-Heavy Metal Mixtures in Industrial Wastewater},

}

