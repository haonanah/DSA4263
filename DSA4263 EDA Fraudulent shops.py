import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Display the first 5 rows to verify the data
#print(data.head())
#print(data.columns)

#Creating plots to show dataset is balanced
#print(data['Label'].value_counts())
#sns.countplot(x='Label', data=data)
#plt.title('Fraudulent vs Legitimate')
#plt.show()  # dataset is balanced 


# Load the dataset (Replace the file path with your actual path)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset (Replace the file path with your actual path)
data = pd.read_csv('C:/Users/jason/OneDrive/Desktop/Jason/Nus education tools/Y4S2/DSA4263/DSA4263/data/Fraudulent_online_shops_dataset.csv')

# Ensure all features are in the correct format, encoding categorical features if necessary
categorical_columns = [
    'Presence of crypto currency',
          'Indication of young domain ',
          'Presence of SiteJabber reviews',
          'Number  of hyphens (-)',
          'SSL certificate issuer organization list item',
          'SSL certificate issuer',
          'Presence of free contact emails',
          'Presence of money back payment',
          'Presence of credit card payment'
]

# Apply label encoding to categorical columns
label_encoder = LabelEncoder()
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Select the features (X) and target variable (y)
X = data[['Presence of crypto currency',
          'Indication of young domain ',
          'Presence of SiteJabber reviews',
          'Number  of hyphens (-)',
          'SSL certificate issuer organization list item',
          'SSL certificate issuer',
          'Presence of free contact emails',
          'Presence of money back payment',
          'Presence of credit card payment']]

y = data['Label']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the Random Forest model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Get feature importances
feature_importances = model.feature_importances_
feature_names = X.columns

# Print feature importances
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print(importance_df)
