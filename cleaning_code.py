import pandas as pd

# Load the dataset
df = pd.read_csv('/content/fraud_oracle.csv')

print(df.columns)
print(df.dtypes)

# Check for missing values
missing_values = df.isnull().sum()
missing_summary = missing_values[missing_values > 0]
if missing_summary.empty:
    print("No missing values found in the dataset.")
else:
    print("Missing values detected:")
    print(missing_summary)

# Convert Yes-No columns to binary (1-0)
yes_no_columns = ['PoliceReportFiled', 'WitnessPresent']
df[yes_no_columns] = df[yes_no_columns].apply(lambda x: x.map({'Yes': 1, 'No': 0}))

# Convert ordinal categorical columns to numerical values
ordinal_mappings = {
    'AgeOfVehicle': {'less than 1': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4, 
                     '5 years': 5, '6 years': 6, '7 years': 7, 'more than 7': 8},
    'AgeOfPolicyHolder': {'16 to 17': 16, '18 to 20': 19, '21 to 25': 23, '26 to 30': 28, 
                          '31 to 35': 33, '36 to 40': 38, '41 to 50': 45, '51 to 65': 58, 
                          'over 65': 70},
    'NumberOfCars': {'1 vehicle': 1, '2 vehicles': 2, '3 to 4': 3, '5 to 8': 5, 'more than 8': 9},
    'NumberOfSuppliments': {'none': 0, '1 to 2': 1, '3 to 5': 3, 'more than 5': 5},
    'AddressChange_Claim': {'no change': 0, 'under 6 months': 1, '1 year': 2, 
                             '2 to 3 years': 3, '4 to 8 years': 4, 'more than 8 years': 5}
}

for col, mapping in ordinal_mappings.items():
    df[col] = df[col].map(mapping)

# Convert numerical columns to proper numeric types
numerical_cols = ['Days_Policy_Accident', 'Days_Policy_Claim', 'PastNumberOfClaims']
df[numerical_cols] = df[numerical_cols].apply(pd.to_numeric, errors='coerce')

# Drop unnecessary columns if needed (e.g., PolicyNumber, RepNumber)
# df.drop(columns=['PolicyNumber', 'RepNumber'], inplace=True)

# Convert binary categorical columns ('Sex' and 'AgentType') to numerical 1-0 mapping
binary_mappings = {
    'Sex': {'Male': 1, 'Female': 0},
    'AgentType': {'Internal': 1, 'External': 0}
}

df.replace(binary_mappings, inplace=True)

df = pd.get_dummies(df, columns=['MaritalStatus'], drop_first=True)

df.to_csv("cleaned_df.csv",encoding = 'utf-8')