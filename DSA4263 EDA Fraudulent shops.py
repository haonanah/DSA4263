import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Replace 'your_dataset.csv' with the path to your file
data = pd.read_csv('C:/Users/jason/OneDrive/Desktop/Jason/Nus education tools/Y4S2/DSA4263/DSA4263/data/Fraudulent_online_shops_dataset.csv')

# Display the first 5 rows to verify the data
#print(data.head())
#print(data.columns)

#Creating plots to show dataset is balanced
print(data['Label'].value_counts())
sns.countplot(x='Label', data=data)
plt.title('Fraudulent vs Legitimate')
plt.show()  # dataset is balanced

# 
