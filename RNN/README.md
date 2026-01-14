# Simple RNN Sentiment Analysis on IMDB Dataset

## 📌 Project Overview

This project demonstrates a **Simple Recurrent Neural Network (RNN)** model built using **TensorFlow/Keras** to perform **binary sentiment analysis** (Positive / Negative) on the **IMDB movie reviews dataset**. The model learns from textual movie reviews and predicts whether a given review expresses positive or negative sentiment.

The project is designed as a **beginner-friendly Deep Learning / NLP implementation**, focusing on understanding how RNNs work with sequential text data.

---

## 🚀 Features

* Uses the **IMDB dataset** provided by Keras
* Text sequence padding for uniform input length
* **Embedding layer** for word representation
* **SimpleRNN** layer for sequence learning
* Binary classification using **Sigmoid activation**
* **EarlyStopping** to prevent overfitting
* Accuracy & Loss visualization
* Model saving and reloading
* Sample sentiment prediction

---

## 🧠 Model Architecture

```
Embedding Layer (1000 words, 32 dimensions)
        ↓
SimpleRNN (128 units, tanh activation)
        ↓
Dense (1 unit, sigmoid activation)
```

---

## 🗂️ Dataset Details

* Dataset: **IMDB Movie Reviews**
* Vocabulary Size: `1000` most frequent words
* Maximum Review Length: `200` tokens
* Labels:

  * `1` → Positive Review
  * `0` → Negative Review

---

## ⚙️ Tech Stack

* **Python 3.11+**
* **TensorFlow / Keras**
* **NumPy**
* **Matplotlib**
* **IMDB Dataset (Keras)**

---

## 📦 Installation

```bash
pip install tensorflow numpy matplotlib
```

---

## ▶️ How to Run

```bash
python project_one.py
```

The script will:

1. Load and preprocess the IMDB dataset
2. Train the Simple RNN model
3. Display training & validation accuracy/loss graphs
4. Save the trained model (`simple_rnn_imdb_model.h5`)
5. Reload the model and evaluate on test data
6. Predict sentiment for a sample review

---

## 📊 Output Examples

* Training vs Validation Accuracy Graph
* Training vs Validation Loss Graph
* Final Test Accuracy & Loss
* Sample Sentiment Prediction with Confidence Score

---

## 💾 Model Saving

The trained model is saved as:

```
simple_rnn_imdb_model.h5
```

You can reload it anytime using:

```python
from tensorflow.keras.models import load_model
model = load_model('simple_rnn_imdb_model.h5')
```

---

## 🧪 Sample Prediction Output

```
Predicted Sentiment: Positive
Prediction Confidence: 0.87
Actual Label: Positive
```

---

## 📈 Learning Outcomes

* Understanding text preprocessing for RNNs
* Working with sequence padding
* Building and training Simple RNN models
* Avoiding overfitting using EarlyStopping
* Visualizing model performance
* Saving and deploying trained models

---

## 🔗 Connect With Me

* **GitHub:** [Ranit Sautya](https://github.com/Ranitsautya)
* **LinkedIn:** [Ranit Sautya](https://linkedin.com/in/ranitsautya)
* **Email:** [ranitsautya@gmail.com](ranitsautya@gmail.com)

---

## ⭐ Future Improvements

* Replace SimpleRNN with **LSTM / GRU**
* Increase vocabulary size
* Add word embeddings like **GloVe**
* Build a **Streamlit web app** for live predictions

---

## 📄 License

This project is open-source and available for educational purposes.

---

**Happy Learning & Deep Learning! 🚀**
