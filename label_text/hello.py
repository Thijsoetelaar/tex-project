import click
import pandas as pd
import os

from joblib import load
from clean import clean_string

# Load in model
current_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
model = load(os.path.join(current_dir,'models/model.joblib'))
feature_importance = pd.DataFrame(model["model"].coef_, columns=model["vectorizer"].get_feature_names_out())

@click.command()
@click.option('--text', default="Ok lar... Joking wif u oni...", help='Text to classify')
def label(text):
    """
    Return the label of the text provided
    """
    X = clean_string(text)
    x = model["vectorizer"].transform([X])

    doc_term_matrix = pd.DataFrame(x.toarray(), columns=model["vectorizer"].get_feature_names_out())
    a = doc_term_matrix.loc[:, (doc_term_matrix != 0).all(axis=0)]
    b = feature_importance[a.columns]

    click.echo(a.T)
    click.echo(b.T)

    y = model["model"].predict(x)
    y_prob = model["model"].predict_proba(x).round(2)

    click.echo(f" {'ham' if y[0] == 1 else 'spam'} with probability: {y_prob[0][y[0]]}")

if __name__ == '__main__':
    label()