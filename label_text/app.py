import os

import streamlit as st
import pandas as pd
from joblib import load

from clean import clean_string

DEFAULT_TEXT = "SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info"

# Data path's and model path's
current_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
model = load(os.path.join(current_dir,'models/model.joblib'))
feature_importance = pd.DataFrame(model["model"].coef_, columns=model["vectorizer"].get_feature_names_out())

# User input
text = st.text_input("Text to classify as HAM or SPAM", DEFAULT_TEXT)

# Call model 
X = clean_string(text)
x = model['vectorizer'].transform([X])
y = model['model'].predict(x)
y_prob = model['model'].predict_proba(x).round(2)

# Results 
output = f" {'ham' if y[0] == 1 else 'spam'} with probability: {y_prob[0][y[0]]}"

st.text_area(label="Output Data:", value=output, height=35)

doc_term_matrix = pd.DataFrame(x.toarray(), columns=model["vectorizer"].get_feature_names_out())
word_count = doc_term_matrix.loc[:, (doc_term_matrix != 0).all(axis=0)]
word_importance = feature_importance[word_count.columns]

word_count = word_count.melt(ignore_index=True, var_name='word', value_name='count').sort_values(by='count', ascending=False)
word_importance = word_importance.melt(ignore_index=True, var_name='word', value_name='importance').sort_values(by='importance', ascending=True)

st.subheader('Word count table (top 5)')
st.table(word_count.head(5))
st.subheader('Word importance table')
st.table(word_importance.style.highlight_min('importance', axis=0))
