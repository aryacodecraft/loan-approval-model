from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

df = pd.read_csv('data.csv')

#drop unnecessary columns and values
df.dropna(inplace=True)
if 'Loan_ID' in df.columns:
    df.drop("Loan_ID", axis=1, inplace=True)


# Encode categorical features
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

x = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=x.columns, class_names=["No", "Yes"], filled=True)
plt.title("Loan Approval Decision Tree")
plt.show()