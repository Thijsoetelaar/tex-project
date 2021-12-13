import re

def clean_string(string):
    """
    Remove punctuation, numbers, and stopwords from a string.
    """
    # Remove numbers
    string = re.sub(r'\d+', '', string)  # remove numbers
    string = string.lower()  # lowercase
    string = string.split()
    # Remove stopwords
    string = ' '.join(string)
    return string
