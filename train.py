
# Importing Libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn import svm
from sklearn.pipeline import Pipeline
import joblib
from sklearn.metrics import f1_score, confusion_matrix


# Reading Data to Pandas DataFrame
data = pd.read_csv("lang_data.csv")
data.head()

# droping null values
df1=data.dropna(axis=0)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df1['text'], df1['language'], test_size=0.25, random_state=1)

# Defining PipeLine
text_svm = Pipeline([('vect2', CountVectorizer()),('tfidf2', TfidfTransformer()),('clf2', svm.SVC())])
text_svm = text_svm.fit(X_train, y_train)

# Parameters for Grid Search
parameters_svm = {'vect2__ngram_range': [(1, 1), (1, 2), (1,3)],'tfidf2__use_idf': (True, False),'clf2__C': [0.1,1, 10, 100], 
                  'clf2__gamma': [1,0.1,0.01,0.001], 'clf2__kernel':["linear", "poly", "rbf", "sigmoid"]}
# Grid Search 
gs_clf_svm = GridSearchCV(text_svm, parameters_svm, n_jobs=-1,scoring='f1_weighted',cv=5)
gs_clf_svm = gs_clf_svm.fit(X_train, y_train)

# Best Score mean cross validated score
print(gs_clf_svm.best_score_)
#{'clf2__C': 10, 'clf2__gamma': 1, 'clf2__kernel': 'sigmoid', 
# 'tfidf2__use_idf': False, 'vect2__ngram_range': (1, 2)}
#Best Params
print(gs_clf_svm.best_params_)

#Best Estimator
best_svm_clf = gs_clf_svm.best_estimator_

#Saving the Model
joblib.dump(best_svm_clf, 'best_svm.pkl')

#prediction on test set
y_pred_test=best_svm_clf.predict(X_test)
y_pred_train=best_svm_clf.predict(X_train)

# f1 score on test set
print(f1_score(y_test, y_pred_test, average=None))
print(confusion_matrix(y_test, y_pred_test))

print(f1_score(y_train, y_pred_train, average=None))
print(confusion_matrix(y_train, y_pred_train))

