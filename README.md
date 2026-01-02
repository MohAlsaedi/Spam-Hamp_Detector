# SMS Spam Classification System

An end-to-end machine learning application for detecting spam SMS messages using a custom **Perceptron algorithm**.  
The system processes raw text messages, converts them into numerical features using **TF-IDF**, evaluates model performance with standard metrics, and provides an **interactive Gradio web interface** for real-time message analysis.

---

## Features

- Custom Perceptron implementation (built from scratch)
- TF-IDF text vectorization (unigrams and bigrams)
- Early stopping during training
- Model evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion matrix
- Interactive web interface using Gradio
- Confidence-based predictions
- Message classification history
- Clean, modular, and well-documented code

---

## Technologies Used

- Python 3
- NumPy
- Pandas
- Scikit-learn
- Gradio

---

## Dataset

The model is trained on the **SMS Spam Collection Dataset**, which contains 5,572 labeled SMS messages.

- Labels: `spam`, `ham`
- Dataset is loaded automatically from an online source during runtime.
- https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv

---

## Requirements

- Python **3.8 or higher**
- pip (Python package manager)

---

## How to Download and Install Dependencies

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/sms-spam-classifier.git
cd sms-spam-classifier
