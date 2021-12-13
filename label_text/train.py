import pandas as pd
import click

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer

from clean import clean_string
from joblib import dump

@click.command()
@click.option('--csv', default="train.csv", help='Csv input file')
def run(csv):
    data = pd.read_csv(f"data/{csv}")

    data.category = data.category.apply(lambda x: 1 if x == "ham" else 0)
    print(f"{data.head(5)}")
    y = data.category
    X = data.message.apply(clean_string)
    
    print(f"{X.head(5)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Dim of X_train: {X_train.shape}")
    print(f"Dim of X_test: {X_test.shape}")

    ctv = CountVectorizer(stop_words='english', max_features=5000)
    ctv.fit(X_train)

    x_train = ctv.transform(X_train)    
    x_test = ctv.transform(X_test)

    clf = LogisticRegression(random_state=0)
    clf.fit(x_train, y_train)
    
    y_pred = clf.predict(x_test)
    
    roc_auc = roc_auc_score(y_test, y_pred)
    print(f"ROC AUC score: {roc_auc}")
    #accuracy_score(y_true, y_pred, normalize=False)

    model = {"model": clf, "vectorizer": ctv}
    
    dump(model, 'models/model.joblib') 
    print("Saved model here: 'models/model.joblib")
    
if __name__ == "__main__":
    run()
