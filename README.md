# Wearable Based Cardiovascular Strain Prediction with Early Warning System

A machine learning system for predicting cardiovascular strain from wearable sensor data collected by Apple Watch and Fitbit devices.

The project estimates **Karvonen exercise intensity**, compares multiple regression models, classifies observations into cardiovascular risk zones, and implements an **Early Warning System (EWS)** for detecting potentially abnormal cardiovascular strain during low-activity states.

---

## Overview

Wearable devices continuously collect physiological and activity data, creating opportunities for real-time cardiovascular monitoring.

This project investigates whether machine learning models can accurately estimate cardiovascular strain using wearable sensor measurements while maintaining strict subject-independent evaluation.

The system:

- Predicts Karvonen exercise intensity from wearable sensor data
- Compares linear and non-linear machine learning models
- Evaluates performance across devices and activity types
- Classifies observations into cardiovascular risk zones
- Generates early warning alerts for potentially abnormal cardiovascular strain
- Prevents subject leakage through subject-aware validation

---

## Dataset

The dataset contains wearable sensor observations collected from:

- Apple Watch
- Fitbit

Each observation includes physiological, demographic, and activity-related measurements.

### Demographic Features

- Age
- Gender
- Height
- Weight
- Resting Heart Rate

### Sensor Features

- Heart Rate
- Steps
- Distance
- Calories Burned
- Heart Rate Entropy
- Step Entropy
- Heart-Step Correlation
- Normalized Heart Rate
- Standard Deviation of Normalized Heart Rate
- Steps × Distance Interaction

### Target Variable

The prediction target is **Karvonen Exercise Intensity**, a physiological measure of exercise effort based on Heart Rate Reserve (HRR).


Intensity = {HeartRate - RestingHeartRate}/{MaxHeartRate - RestingHeartRate}

where:
MaxHeartRate = 220 - Age

---

## Data Cleaning

Several preprocessing steps were applied before training.

### Subject Identification

Participants were grouped using demographic attributes:

- Age
- Gender
- Height
- Weight

These groups were used to generate subject identifiers for leakage-free evaluation.

### Sensor Quality Filtering

Resting heart rate measurements outside the physiologically plausible range:

- Below 40 bpm
- Above 120 bpm

were treated as wearable sensor artifacts and removed.

### Device Encoding

Device type was converted into a binary feature:

| Device | Encoded Value |
|----------|----------|
| Apple Watch | 0 |
| Fitbit | 1 |

---

## Feature Engineering

Three physiologically motivated features were engineered.

### Heart Rate Reserve (HRR)

```text
HRR = (220 - Age) - RestingHeartRate
```

Represents the available range for heart rate increase during physical activity.

### Age–Heart Rate Interaction

```text
Age × HeartRate
```

Captures the fact that identical heart rates may represent different exercise intensities for individuals of different ages.

### Metabolic Proxy

```text
Weight × Steps
```

Approximates mechanical workload and energy expenditure.

---

## Train/Test Strategy

To prevent subject leakage:

- Subjects were split using `GroupShuffleSplit`
- 80% training subjects
- 20% testing subjects
- No participant appears in both training and testing sets

Hyperparameter tuning and validation used:

**5-Fold GroupKFold Cross Validation**

This ensures that every validation fold remains subject-independent.

---

## Models Evaluated

### ElasticNet Regression

Linear regression model combining:

- L1 regularization
- L2 regularization

Advantages:

- Interpretable coefficients
- Fast training
- Reduced overfitting

### Gradient Boosting Regressor (GBM)

Ensemble tree-based model capable of learning:

- Non-linear relationships
- Feature interactions
- Complex cardiovascular patterns

Advantages:

- Higher predictive power
- Robust handling of mixed feature types

---

## Hyperparameter Optimization

Hyperparameters were optimized using `GridSearchCV`.

### ElasticNet

Parameters tuned:

- Alpha
- L1 Ratio

### Gradient Boosting Regressor

Parameters tuned:

- Number of Estimators
- Tree Depth
- Learning Rate
- Minimum Samples per Leaf
- Subsample Ratio

All tuning was performed using **GroupKFold** cross-validation.

---

## Evaluation Metrics

Models were evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Mean Absolute Percentage Error (MAPE)

Additional evaluations included:

- Per-device performance
- Per-activity performance
- Per-subject generalization
- Risk-zone classification performance

---

## Cardiovascular Risk Zones

Predicted intensity values were mapped to cardiovascular training zones.

| Zone | Intensity Range |
|--------|--------|
| Zone 1 – Recovery | < 0.20 |
| Zone 2 – Light Aerobic | 0.20 – 0.40 |
| Zone 3 – Moderate Aerobic | 0.40 – 0.60 |
| Zone 4 – Anaerobic Threshold | 0.60 – 0.80 |
| Zone 5 – Maximum Effort | > 0.80 |

The best-performing model was evaluated as a risk-zone classifier using:

- Precision
- Recall
- F1 Score

for each cardiovascular zone.

---

## Early Warning System (EWS)

A lightweight cardiovascular alert system was implemented.

An alert is triggered when:

```text
Predicted Intensity ≥ 0.85
```

while the recorded activity is:

- Lying
- Sitting
- Self Pace Walk

These situations may indicate cardiovascular strain that is unusually high relative to activity level.

### Example Alert

```text
⚠ EWS ALERT

Predicted Intensity: 0.91
Activity: Lying

Potential cardiovascular over-exertion detected.
```

---

## Visualizations

The project generates multiple analytical visualizations, including:

- Intensity distribution plots
- Correlation heatmaps
- Resting heart rate vs age analysis
- Actual vs predicted scatter plots
- Residual analysis
- Feature importance visualizations
- Per-activity performance comparisons
- Per-device performance comparisons
- Risk-zone distributions
- Per-subject generalization analysis
- Overall model comparison charts

All generated figures are automatically saved to:

```text
outputs/figures/
```

---

## Example Use Case

The trained model can predict cardiovascular strain from a single wearable observation and provide:

- Predicted cardiovascular intensity
- Risk zone classification
- Early warning alerts
- Cardiovascular strain assessment

Example output:

```python
{
    "predicted_intensity": 0.873,
    "risk_zone": "Zone 5 – Maximum Effort",
    "ews_flag": True,
    "ews_message": "Potential cardiovascular over-exertion detected."
}
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Joblib

---

## Key Contributions

- Subject-aware machine learning pipeline
- Wearable-device cardiovascular monitoring
- Physiologically motivated feature engineering
- Device-agnostic evaluation across Apple Watch and Fitbit
- Cardiovascular risk zone classification
- Early Warning System for abnormal cardiovascular strain
- Comprehensive visualization and model analysis framework

---


## Authors

- Shalini Dubey
- Riya Gupta
