import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

file_path = "C:/Users/semih/Desktop/sınıflandırma/Social_Network_Ads.csv"
data = pd.read_csv("C:/Users/semih/Desktop/sınıflandırma/Social_Network_Ads.csv")

print(data.head())

X = data.iloc[:, [2, 3]].values  
y = data.iloc[:, 4].values 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


classifier = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=4, min_samples_split=10, min_samples_leaf=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(f"Doğruluk Skoru: {accuracy_score(y_test, y_pred)}")

cm = confusion_matrix(y_test, y_pred)
print(f"Karışıklık Matrisi:\n{cm}")

plt.figure(figsize=(25, 10))
tree.plot_tree(classifier, filled=True, feature_names=['Age', 'EstimatedSalary'], class_names=['No', 'Yes'], rounded=True)
plt.title("Karar Ağacı Modeli")
plt.show()
