from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score,recall_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd


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

model = XGBClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision_score(y_test, y_pred) * 100:.2f}%')
print(f'Recall: {recall_score(y_test, y_pred) * 100:.2f}%')
