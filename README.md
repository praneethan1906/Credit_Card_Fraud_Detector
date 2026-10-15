This is my project structure 

Credit_Card_Fraud_Detector/
│
├── data/
│   └── creditcard.csv
│
├── models/
│   └── model_pipeline.pkl
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── train.py
│   └── evaluate.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

I have downloaded the dataset from kaggle
then I have performed EDA
the dataset contains PCA value so I include all of them for model training
I trained Logistic regression ,Random forest classifier and XGboost
the accuracy of Random forest classifier is high
so I used it to classify whether the transaction is legitimate are fraud using streamlit.
