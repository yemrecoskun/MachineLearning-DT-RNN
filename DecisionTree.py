import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

col_names = ['Recency', 'Frequency', 'Monetary', 'Time', 'BloodinMarch2017']
# load dataset
pima = pd.read_csv("transfusion.data", header=None, names=col_names)
print(pima)

#split dataset in features and target variable
feature_cols = ['Recency', 'Frequency', 'Monetary', 'Time']
X = pima[feature_cols] # Features
y = pima.BloodinMarch2017 # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:",accuracy)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred,target_names=['1','0']))


dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes1.png')
Image(graph.create_png())
