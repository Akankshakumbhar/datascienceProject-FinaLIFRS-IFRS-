import pandas as pd
from typing import Optional, List, Tuple, Dict, Union

from sklearn import linear_model


def validate_dataframe(df: pd.DataFrame,
                       n_cols: Optional[int] = None,
                       n_rows: Optional[Tuple[int, int]] = None,
                       columns: Optional[List[str]] = None,
                       column_types: Optional[Dict[str, type]] = None,
                       check_duplicates: bool = False,
                       check_null_values: bool = False,
                       unique_columns: Optional[List[str]] = None,
                       column_ranges: Optional[Dict[str, Tuple[Union[int, float], Union[int, float]]]] = None,
                       date_columns: Optional[List[str]] = None,
                       categorical_columns: Optional[Dict[str, List[Union[str, int, float]]]] = None
                       ) -> Tuple[bool, str]:
    """
    Validates a Pandas DataFrame based on specified criteria.

    Parameters:
    - df (pd.DataFrame): The DataFrame to be validated.
    - n_cols (int, optional): Number of expected columns in the DataFrame.
    - n_rows (tuple, optional): Tuple (min_rows, max_rows) specifying the expected range of rows.
    - columns (list, optional): List of column names that should be present in the DataFrame.
    - column_types (dict, optional): Dictionary mapping column names to the expected data types.
    - check_duplicates (bool, optional): Check for the presence of duplicate rows in the DataFrame.
    - check_null_values (bool, optional): Check for the presence of null values in the DataFrame.
    - unique_columns (list, optional): List of columns that should have only unique values.
    - column_ranges (dict, optional): Dictionary mapping numeric columns to the allowed ranges.
    - date_columns (list, optional): List of columns containing date values to validate the format.
    - categorical_columns (dict, optional): Dictionary mapping categorical columns to allowed values.

    Returns:
    - tuple: (bool, str) indicating success or failure, and an optional description of the problem.
    """

    # Validate number of columns
    if n_cols is not None and len(df.columns) != n_cols:
        return False, f"Error: Expected {n_cols} columns, but found {len(df.columns)} columns."

    # Validate row range
    if n_rows is not None:
        min_rows, max_rows = n_rows
        if not (min_rows <= len(df) <= max_rows):
            return False, f"Error: Number of rows should be between {min_rows} and {max_rows}."

    # Validate columns
    if columns is not None and not set(columns).issubset(df.columns):
        missing_columns = set(columns) - set(df.columns)
        return False, f"Error: Missing columns: {missing_columns}."

    # Validate column types
    if column_types is not None:
        for col, expected_type in column_types.items():
            if col not in df.columns:
                return False, f"Error: Column '{col}' not found."
            if not df[col].dtype == expected_type:
                return False, f"Error: Column '{col}' should have type {expected_type}."

    # Validate duplicates in specific columns
    if check_duplicates and df.duplicated().any():
        return False, "Duplicates found in the DataFrame."

    # Validate null values in specific columns
    if check_null_values and df.isnull().any().any():
        return False, "DataFrame contains null values."

    # Validate unique values in specific columns
    if unique_columns is not None:
        for col in unique_columns:
            if col in df.columns and df[col].duplicated().any():
                return False, f"Column '{col}' should have only unique values."

    # Validate values in a specific range
    if column_ranges is not None:
        for col, value_range in column_ranges.items():
            if col in df.columns and not df[col].between(*value_range).all():
                return False, f"Values in '{col}' should be between {value_range[0]} and {value_range[1]}."

    # Validate date format (assuming 'date_columns' are date columns)
    if date_columns is not None:
        for col in date_columns:
            if col in df.columns:
                try:
                    pd.to_datetime(df[col], errors='raise')
                except ValueError:
                    return False, f"'{col}' should be in a valid date format."

    # Validate categorical values
    if categorical_columns is not None:
        for col, allowed_values in categorical_columns.items():
            if col in df.columns and not df[col].isin(allowed_values).all():
                return False, f"Values in '{col}' should be {allowed_values}."

    # If all validations pass, return True
    return True, "DataFrame has passed all validations."

# Usage example:
model_config = pd.read_csv(r"C:\Users\Star\Desktop\Stuffs(IMP)\Project Task\lending-club-data\model_config.csv")
is_valid, message = validate_dataframe(model_config, n_cols=4, check_duplicates=True)
print(is_valid, message)
# Usage example:
model_collateral = pd.read_csv(r"C:\Users\Star\Desktop\Stuffs(IMP)\Project Task\lending-club-data\model_collteral.csv")
is_valid, message = validate_dataframe(model_collateral, n_cols=78, check_duplicates=True)
print(is_valid, message)



import os
import pandas as pd

folder_path = r"C:\Users\Star\Desktop\Stuffs(IMP)\Project Task\lending-club-data\model_authorrep"
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
dfs = []

# Read CSV files and append dataframes to the list
for csv_file in csv_files:
    model_authorrep = pd.read_csv(os.path.join(folder_path, csv_file))
    dfs.append(model_authorrep)

# Validate each dataframe in the list
for df in dfs:
    is_valid, message = validate_dataframe(df, n_cols=14, check_duplicates=True)
    print(is_valid, message)

# Concatenate all dataframes into a single dataframe
model_authorrep= pd.concat(dfs, ignore_index=True)

#good_dataframe=pd.join([model_authorrep, model_config,model_collateral])

join_coll_config=pd.merge(model_config,model_collateral,on='id')
#print(join_coll_config);

join_coll_auth=pd.merge(model_collateral,model_authorrep,on='id')
#print(join_coll_auth);
#ead computation reports

#stage1 Ecl
import duckdb
stage1ecl=duckdb.query("Select  EAD*PD12*LGD as Stage1 from join_coll_auth ").df()
#print(stage1ecl)

#stage2
stage2ecl=duckdb.query("select EAD*PDLT*LGD  as Stage2 from  join_coll_auth").df()
#print(stage2ecl)
# stage 3
stage3ecl=duckdb.query("select EAD*LGD  as Stage3 from join_coll_auth").df()
#print(stage3ecl)
# step 3#
# stage 1,stage2,stage3,PD12,PDLT,EAD,LGD(conact in one datframe as final 1)

import duckdb
cols=duckdb.query("select PD12,PDLT,EAD,LGD from join_coll_auth ").df()

Ecl_computation=Ecl_computation=pd.concat([stage1ecl,stage2ecl,stage3ecl,cols], axis=1)
Ecl_computation=Ecl_computation.reset_index(drop=True)
print(Ecl_computation)

# finalpart1(writing in csv)
Ecl_computation_report=pd.DataFrame(Ecl_computation)
output_excel_path=r"C:\Users\Star\Desktop\Stuffs(IMP)\dataScience expotent files\Project (datascience)\Output_data.csv"
Ecl_computation_report.to_csv(output_excel_path,index=False)


#step3:ead varition reports

change_EAD=duckdb.query('select EAD-"Previous EAD" as Change_EAD from join_coll_auth  ').df()
#print(change_EAD)

percentage=duckdb.query('select ((EAD-"Previous EAD")/"Previous EAD")*100  as Percentage from join_coll_auth  ').df()
#print(percentage)

data2 = duckdb.query('SELECT "Reporting Date",EAD,"Previous EAD" from join_coll_auth  ').df()
#print(data2)

EAD_report=pd.concat([data2,percentage,change_EAD],axis=1)
EAD_report=EAD_report.reset_index(drop=True)
print(EAD_report)

EAD_reports=pd.DataFrame(EAD_report)
output_excel_path=r"C:\Users\Star\Desktop\Stuffs(IMP)\dataScience expotent files\Project (datascience)\A.csv"

EAD_reports.to_csv(output_excel_path,index=False)



#step 4 Stage varitions
stage_varation=duckdb.query('select Stage-"Previous Stage"  as Stagevaration from join_coll_auth ').df()
#print(stage_varation)
#Stage_varation=join_coll_auth['Stage']-join_coll_auth['Previous Stage']
#Stage_varation=Stage_varation.to_frame(name="Stage Variation")
#print(Stage_varation)


stage_varation_percentage=duckdb.query('select ((stage-"Previous Stage")/"Previous Stage")*100  as Stage_varation_Percentage from join_coll_auth').df()
#print(stage_varation_percentage)
#per=((join_coll_auth['Stage']-join_coll_auth['Previous Stage'])/join_coll_auth['Previous Stage'])*100
#per=per.to_frame(name='per')
#print(per)


stage=duckdb.query('select "Reporting Date","Previous Stage",Stage from join_coll_auth ').df()
#print(stage)

stage1=pd.concat([stage,stage_varation,stage_varation_percentage],axis=1)
stage1=stage1.reset_index(drop=True)
print(stage1)

Ecl_computation['Percent_ECL_Stage1'] = (Ecl_computation['Stage1'] / (Ecl_computation['EAD'] * Ecl_computation['LGD'])) * 100
Ecl_computation['Percent_ECL_Stage2'] = (Ecl_computation['Stage2'] / (Ecl_computation['EAD'] * Ecl_computation['LGD'])) * 100
Ecl_computation['Percent_ECL_Stage3'] = (Ecl_computation['Stage3'] / (Ecl_computation['EAD'] * Ecl_computation['LGD'])) * 100

# Step 5: Save the computed percentages to a DataFrame and export to Excel
Percent_ECL_report = Ecl_computation[['Percent_ECL_Stage1', 'Percent_ECL_Stage2', 'Percent_ECL_Stage3']]
print(Percent_ECL_report)

percentage_of_ECL = 10  # Adjust this percentage as needed

# Calculate the total ECL for all stages
Ecl_computation['Total_ECL'] = Ecl_computation['Stage1'] + Ecl_computation['Stage2'] + Ecl_computation['Stage3']

# Calculate the loan loss provision based on the specified percentage of the total ECL
Ecl_computation['Loan_Loss_Provision'] = (percentage_of_ECL / 100) * Ecl_computation['Total_ECL']

# Print or use the computed loan loss provision
print(Ecl_computation['Loan_Loss_Provision'])




import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

Ecl_computation=pd.read_csv(r"C:\Users\Star\Desktop\Stuffs(IMP)\dataScience expotent files\Project (datascience)\Output_data.csv")
Ead_report=pd.read_csv(r"C:\Users\Star\Desktop\Stuffs(IMP)\dataScience expotent files\Project (datascience)\A.csv")

merged_data = pd.merge(Ecl_computation, Ead_report, on='EAD')
print(merged_data)
print(merged_data['Reporting Date'])
merged_data.dropna()
# Assuming 'Target' is binary (0 or 1), you may need to adjust this based on your data
# Convert the target variable to binary if necessary
#merged_data['Reporting Date'] = merged_data['Reporting Date'].astype(int)
merged_data['Reporting Date'] = pd.to_datetime(merged_data['Reporting Date'])
# Split the data into features (X) and target variable (y)
X = merged_data.drop(columns=['EAD', 'Reporting Date', 'Previous EAD', 'Reporting Date'])
y = merged_data['Reporting Date']
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply logistic regression
logistic_model = LogisticRegression()
logistic_model.fit(X_test, y_test)

# Predictions
y_pred = logistic_model.predict(X_train)

# Model evaluation
accuracy = accuracy_score(y_train, y_pred)
conf_matrix = confusion_matrix(y_train, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)

# Plot scatter plot of actual vs. predicted values
plt.scatter(y_train, y_pred, alpha=0.5)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs. Predicted")
plt.show()

from sklearn.ensemble import RandomForestClassifier

# Apply Random Forest classifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Model evaluation
accuracy_rf = accuracy_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

print("Random Forest Model Accuracy:", accuracy_rf)
print("Random Forest Confusion Matrix:\n", conf_matrix_rf)


