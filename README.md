# 🍽️ Predictive Restaurant Recommender System

A machine learning-based recommendation engine that predicts which restaurants customers are most likely to order from, based on customer location, restaurant information, and historical order data.

## 🔗 Live Demo
👉 [Hugging Face Space](https://huggingface.co/spaces/rahulkryadavbit/Predictive_restaurant_recommender_system)

## 📊 Problem Statement
Build a recommendation engine to predict what restaurants customers are most likely to order from given:
- Customer location
- Restaurant information
- Customer order history

## 📈 Results
| Metric | Score |
|--------|-------|
| AUC-ROC | 0.9102 |
| F1 Score | 0.2255 |
| Best Threshold | 0.7132 |

## 🤖 Model
- Algorithm: LightGBM Classifier
- Class imbalance handling: scale_pos_weight = 19
- Train/Val split: Customer-level GroupShuffleSplit 80/20

## 🔍 Key EDA Findings
- 135,303 total orders in training data
- 100 unique vendors
- 1:19 class imbalance ratio
- Full cold start — test customers have zero order history
- 47% customers order from only 1 vendor

## ⚙️ Features Used
- Customer-Vendor Euclidean distance
- Vendor total orders (popularity)
- Vendor average rating
- Vendor popularity rank
- Vendor repeat customer ratio
- Vendor average preparation time
- Customer gender and language

## 🚀 How to Run
```bash
git clone https://github.com/poiuyfddjhgfd/restaurant-recommender.git
cd restaurant-recommender
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## 👤 Author
Rahul Kumar Yadav
Integrated M.Sc. Mathematics & Computing — BIT Mesra
BS Data Science — IIT Madras