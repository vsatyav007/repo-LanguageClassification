import joblib
#import pandas as pd
if __name__=='__main__':
    best_svm_clf = joblib.load('best_svm.pkl')
    text = [input("Enter your input ")]
    y_pred = best_svm_clf.predict(text)
    print(y_pred)

    # #Uncomment to read from csv file and predict
    #df= pd.read_csv("your file")
    #y_pred = best_svm_clf.predict(df)
    #print(y_pred)
