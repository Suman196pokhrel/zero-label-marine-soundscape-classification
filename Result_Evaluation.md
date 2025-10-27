# 🌊 Automated Reef Soundscape Classification using Weakly-Supervised Learning

## 📊 Overview

This project implements an automated classification system for marine acoustic data using a weakly-supervised learning approach. The system classifies reef soundscapes into **biophonic (BIO)** , **ambient (AMBIENT)** and **human_made(boats, submarines etc) (HUMAN)** categories without requiring extensive manual labeling.

---

## 🗂️ Dataset

| Attribute | Details |
|-----------|---------|
| **Source** | Long-term acoustic monitoring of coral reef ecosystems |
| **Total Size** | 1,053,610 audio clips (10-second segments) |
| **Loggers** | 2 autonomous underwater recording units (2802, 2823) |
| **Monitoring Duration** | 271 days of continuous recording |
| **Training Set** | 15,392 clips (after deduplication) |
| **Test Set** | 20 independent clips (unseen temporal periods) |

### 📍 Key Dataset Characteristics:
- ⚠️ **Small test sample (n=20)** - Limited diversity for validation
- ⚠️ **Temporal sampling** - Test clips from same loggers, different time periods
- ⚠️ **No ground truth labels** - Pseudo-labels generated via automated rules

---

## 🔬 Methodology

### 1️⃣ Feature Extraction

**Combined Feature Set (1,041 dimensions):**

| Feature Type | Dimensions | Description |
|--------------|------------|-------------|
| **YAMNet Embeddings** | 1,024 | Pre-trained deep learning features from AudioSet |
| **Ecoacoustic Indices** | 17 | Domain-specific acoustic metrics |

**Ecoacoustic Features:**
- 🎵 **Spectral**: Centroid, bandwidth, rolloff, contrast, flatness
- ⏱️ **Temporal**: Zero-crossing rate, RMS energy
- 🎼 **Cepstral**: MFCC coefficients (first 2)

**Dimensionality Reduction:**
- 📉 PCA: 1,041 → 39 dimensions
- Captures essential variance while reducing computational cost

---

### 2️⃣ Unsupervised Clustering

**Algorithm Comparison:**

| Algorithm | K/Clusters | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|-----------|------------|------------|----------------|-------------------|
| **K-Means** | 3 | **0.769** ✅ | **0.729** ✅ | **47,437** ✅ |
| HDBSCAN | 4 | 0.271 | 1.025 | 7,413 |
| GMM | 3 | 0.219 | 1.807 | 5,817 |

**Selected Algorithm:** K-Means (K=3)
- ✅ Best silhouette score (0.769) - excellent cluster separation
- ✅ Lowest Davies-Bouldin index (0.729) - well-separated, compact clusters
- ✅ Highest Calinski-Harabasz score - best-defined clusters

**Final Cluster Distribution:**
```
Cluster 0: 14,524 clips (94.4%) → Low-energy, continuous patterns
Cluster 1:    325 clips  (2.1%) → Mid-energy, variable patterns
Cluster 2:    543 clips  (3.5%) → High-energy, impulsive patterns
```

---

### 3️⃣ Pseudo-Labeling Strategy

**Rule-Based Heuristics** (no manual annotation):

| Cluster | Assigned Label | Confidence | Acoustic Characteristics |
|---------|----------------|------------|-------------------------|
| **0** | AMBIENT | 0.70 | Low frequency, continuous, low variance |
| **1** | BIO | 0.75 | Variable frequency, moderate complexity |
| **2** | BIO | 0.80 | High frequency, impulsive, high complexity |

⚠️ **Note:** Confidence scores are heuristic-based, not validated against ground truth.

---

### 4️⃣ Two-Stage Classification System

#### **Stage 1: AMBIENT vs BIO Classifier**

**Model Configuration:**
- 🤖 **Algorithm**: Logistic Regression (multinomial)
- 🎯 **Features**: 39 PCA components
- ⚖️ **Sample Weighting**: Confidence scores used as weights

**Training Performance:**
```
Training Accuracy:  99.99%
Test Accuracy:      99.90%
```

**Per-Class Metrics:**
```
Category    Precision  Recall   F1-Score  Support
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AMBIENT     1.000      0.999    0.999     2,905
BIO         0.989      0.994    0.991       174
```

#### **Stage 2: HUMAN Sound Detector**

**Purpose:** Identify anthropogenic sounds (boats, machinery) within the AMBIENT category

**Approach:**
- Analyzes AMBIENT predictions from Stage 1
- Applies additional rules for low-frequency, engine-like patterns
- Creates three-class output: **AMBIENT | BIO | HUMAN**

**Note:** Stage 2 detector implementation in progress; current validation focuses on Stage 1.

---

## 🧪 Model Evaluation

### Evaluation Rationale

⚠️ **Challenge:** No ground truth labels available for comprehensive validation.

**Solution:** Two complementary validation approaches that assess model quality without manual annotation:

---

### 🧠 Method 1: Robustness Testing (Perturbation Analysis)

**Objective:** Test if predictions remain stable when features are slightly perturbed.

**Approach:**
- Applied 4 perturbation types to test features
- Tested at 3 strength levels (0.01, 0.05, 0.10)
- Repeated 5 times per configuration
- Measured prediction consistency: original vs. perturbed

**Results:**

| Perturbation | Strength 0.01 | Strength 0.05 | Strength 0.10 |
|--------------|---------------|---------------|---------------|
| 🎲 **Noise** | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| 📏 **Scale** | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| ↔️ **Shift** | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| 🕳️ **Dropout** | 1.000 ± 0.000 | 0.990 ± 0.020 | 0.980 ± 0.024 |

```
📊 Overall Robustness Score: 0.998 (99.8%)
```

**Interpretation:**
- ✅ Model predictions are extremely stable under perturbations
- ✅ Suggests learning of robust patterns rather than overfitting to noise

---

### 🔁 Method 2: Model Agreement Analysis (Ensemble Validation)

**Objective:** Verify that different model architectures converge to same predictions.

**Approach:**
- Trained 5 diverse classifiers on identical data
- Algorithms: Logistic Regression, Random Forest, Gradient Boosting, SVM, Neural Network
- Evaluated agreement on held-out test set (n=3,079)

**Individual Model Performance:**

| Model | Train Acc. | Test Acc. | Gap |
|-------|------------|-----------|-----|
| 🎯 **Logistic Regression** | 0.9999 | 0.9990 | 0.0009 |
| 🌳 **Random Forest** | 1.0000 | 0.9997 | 0.0003 |
| 🚀 **Gradient Boosting** | 1.0000 | 0.9990 | 0.0010 |
| 🎪 **SVM** | 0.9999 | 0.9997 | 0.0002 |
| 🧠 **Neural Network** | 0.9998 | 0.9981 | 0.0017 |

**Inter-Model Agreement Matrix:**

|  | LR | RF | GB | SVM | NN |
|--|----|----|----|----|-----|
| **LR** | 1.000 | 0.999 | 0.999 | 0.999 | 0.998 |
| **RF** | 0.999 | 1.000 | 0.999 | 1.000 | 0.998 |
| **GB** | 0.999 | 0.999 | 1.000 | 0.999 | 0.997 |
| **SVM** | 0.999 | 1.000 | 0.999 | 1.000 | 0.998 |
| **NN** | 0.998 | 0.998 | 0.997 | 0.998 | 1.000 |

```
📊 Mean Pairwise Agreement: 0.999 (99.9%)
📊 Mean Cohen's Kappa: 0.987 (almost perfect agreement)
```

**Consensus Analysis:**
```
✅ Unanimous (all 5 agree):  3,070 clips (99.7%)
⚠️ Majority (≥3 agree):         9 clips  (0.3%)
❌ No consensus:                0 clips  (0.0%)
```

**Interpretation:**
- ✅ Different model architectures converge to nearly identical predictions
- ✅ Suggests features are informative and pseudo-labels are reasonable

---

## 📈 Validation Summary

### Combined Scores

| Metric | Score | Interpretation |
|--------|-------|----------------|
| 🧠 **Robustness** | 0.998 | Predictions stable under perturbations |
| 🔁 **Model Agreement** | 0.999 | Multiple models strongly agree |
| 🎯 **Combined Confidence** | **0.998** | High overall validation confidence |

---

## ⚠️ Critical Analysis: Why Such High Scores?

### Possible Explanations for Near-Perfect Validation (0.998)

#### 1. 🎯 **Small & Non-Representative Test Set**
- Test sample: **n=20** (0.002% of available data)
- Insufficient diversity to capture real-world variability
- May have randomly selected "easy" cases that happen to match training distribution
- **Risk:** Performance likely to drop significantly on larger, more diverse test sets

#### 2. 🎲 **Pseudo-Label Confirmation Bias**
- All 5 models trained on identical pseudo-labels from same clustering algorithm
- High agreement may reflect learning the clustering logic rather than acoustic reality
- Validation tests internal consistency, not actual prediction accuracy
- **Risk:** Systematic errors in automated labeling would propagate consistently across all models

#### 3. 📊 **Task Simplicity in Current Dataset**
- 94% of data from single dominant cluster (AMBIENT)
- Well-separated clusters in feature space (Silhouette: 0.769)
- Binary classification on clearly distinct acoustic patterns may be trivial
- **Risk:** Real-world edge cases (dawn chorus, transitional sounds, degraded audio) may be significantly harder

---

## 🔮 Future Work & Recommendations

### Priority Validation Steps

#### 1. 📏 **Expand Test Set Size**
- Test on **500-1,000 clips** sampled across full temporal range (271 days)
- Include diverse conditions: dawn/dusk, various weather, different seasons
- Calculate confidence intervals and statistical significance

#### 3. 🎯 **Complete Three-Class System**
- Finalize Stage 2 HUMAN detector implementation
- Test anthropogenic sound detection on boat/engine clips
- Validate full AMBIENT | BIO | HUMAN classification pipeline

#### 4. 🌍 **Cross-Site & Cross-Equipment Validation**
- Test on different reef locations to assess geographic generalization
- Test on different recording equipment to assess robustness to hardware variation

---

## ✅ Current Findings (with Caveats)

### What We Can Confidently Say:

✅ **Model is internally consistent:**
- 99.8% robustness shows stable predictions
- 99.9% agreement shows reproducible results

✅ **Features capture patterns:**
- Multiple architectures learn similar decision boundaries
- PCA features contain discriminative information

✅ **Pseudo-labeling approach is feasible:**
- Automated pipeline successfully trains classifiers
- No manual annotation required for initial model

### What Requires Further Validation:

⚠️ **Actual classification accuracy:**
- High validation scores don't guarantee correct predictions
- Need manual verification on representative sample

⚠️ **Generalization to diverse conditions:**
- Current test set is small and potentially homogeneous
- Larger, more diverse test set needed

⚠️ **Pseudo-label quality:**
- Automated labels may contain systematic errors
- Domain expert review recommended

---

## 🎯 Conclusion

The validation results (**Combined Confidence: 0.998**) indicate the Stage 1 classifier (AMBIENT vs BIO) has learned internally consistent patterns. However, the **exceptionally high scores (>99.8%) warrant cautious interpretation** due to:

1. ⚠️ Small test sample (n=20, 0.002% of dataset)
2. ⚠️ Pseudo-label quality unverified against ground truth
3. ⚠️ Possible task simplicity in well-separated clusters

**Current Status:**
- ✅ **Stage 1 (AMBIENT vs BIO):** Validated using robustness & ensemble methods
- ⏳ **Stage 2 (HUMAN detection):** Implementation in progress

**Recommendation:** Treat current results as **proof-of-concept validation** demonstrating the weakly-supervised approach is promising. However, deployment readiness requires:
- Testing on 500+ manually-verified clips
- Expert validation of pseudo-labeling rules
- Completion of three-class (AMBIENT | BIO | HUMAN) system

The methodology successfully demonstrates automated classification without extensive manual labeling, but production deployment should be deferred until comprehensive validation on larger, expert-labeled test sets.
