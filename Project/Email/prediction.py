import pickle
def prediction(text, cv_path, cls_path):
    # convert into string
    text = str(text)
    
    # load "string to numeric model"
    cv_model = pickle.load(open(cv_path, 'rb'))

    # transform into numeric
    numeric_text = cv_model.transform([text])

    # load classifier model
    cls_model = pickle.load(open(cls_path, 'rb'))

    #predict
    pred = cls_model.predict(numeric_text)

    # prediction give two output, 0 and 1
    # 0 :--> harm
    # 1 :--> spam
    if(pred[0]==1):
        return "spam"
    else:
        return "harm"


cv_path="/content/cv.pkl"
cls_path="/content/model.pkl"
data=input("enter email: ")
pred = prediction(data, cv_path, cls_path)
print(pred)