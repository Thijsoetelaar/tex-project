import os

import streamlit as st
import pandas as pd
from joblib import load

from clean import clean_string


# Data path's and model path's
current_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
model = load(os.path.join(current_dir,'models/model.joblib'))
feature_importance = pd.DataFrame(model["model"].coef_, columns=model["vectorizer"].get_feature_names_out())

# User input
text = st.text_input("Text to classify as HAM or SPAM", "sample text")

# Call model 
X = clean_string(text)
x = model['vectorizer'].transform([X])
y = model['model'].predict(x)
y_prob = model['model'].predict_proba(x).round(2)

# Results 
output = f" {'ham' if y[0] == 1 else 'spam'} with probability: {y_prob[0][y[0]]}"

st.text_area(label="Output Data:", value=output, height=35)

doc_term_matrix = pd.DataFrame(x.toarray(), columns=model["vectorizer"].get_feature_names_out())
a = doc_term_matrix.loc[:, (doc_term_matrix != 0).all(axis=0)]
b = pd.concat([a,feature_importance[a.columns]],ignore_index=True)
st.table(b)
