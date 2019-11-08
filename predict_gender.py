from sklearn.externals import joblib
import re 

def preprocessing_name(name):
    '''
    preprocessing text: remove the excess character in the name
    '''
    text = name.lower()
    ls_error_key = ['~', '`', '!', '@', '#', '%', '$', '^', '&', '*', '(', ')', '_', '+', 
                    '<', '>', '?', ':', '"', '{', '}', ',', '.', ';', '=', '-', '0', '1', 
                    '2', '3', '4', '5', '6', '7', '8', '9']
    for key in ls_error_key:
        if key in text:
            text = text.replace(key, ' ')
    text = re.sub('\s+', ' ', text)
    return text.strip()


def remove_house_name(text):
    '''
    remove the house name in the name
    '''
    ls_text = text.split()
    if len(ls_text) > 2:
        ls_text = ls_text[1:]
        name = " ".join(ls_text)
        return name.strip()
    else:
        return text


def predict_gender(text):
    text = preprocessing_name(text)
    text = remove_house_name(text)
    # load feature text model
    vectorize = joblib.load('model/vectorizer_bow')
    vec_text = vectorize.transform([text])
    # load model
    model = joblib.load('model/logistic_BOW_logistic')
    y_pred = model.predict(vec_text)
    return y_pred[-1]

