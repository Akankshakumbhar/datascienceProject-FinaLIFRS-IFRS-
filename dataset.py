import pandas as pd
from typing import Optional, List, Tuple, Dict, Union

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

join_coll_config=pd.merge(model_config,model_collateral,on='id')
#print(join_coll_config);

join_coll_auth=pd.merge(model_collateral,model_authorrep,on='id')
#print(join_coll_auth);





# step 3#
# stage 1,stage2,stage3


import duckdb
cols=duckdb.query("select PD12,PDLT,EAD,LGD ,(EAD*PD12) as Stage1 ,(EAD*PDLT)as stag2 ,(EAD*LGD) as stage3 from join_coll_auth ").df()
print(cols)



#part1(writing in csv)


final_stage=pd.DataFrame(cols)
output_excel_path=r"C:\Users\Star\Desktop\Stuffs(IMP)\dataScience expotent files\Project (datascience)\Output_data.csv"

final_stage.to_csv(output_excel_path,index=False)



#Part2(writing in csv)
data2 = duckdb.query('SELECT "Reporting Date",EAD,"Previous EAD",(EAD-"Previous EAD") as Change_EAD,(Change_EAD/"Previous EAD")*100 as Percentage FROM join_coll_auth ').df()
print(data2)




final_stage1=pd.DataFrame(data2)
output_excel_path=r"C:\Users\Star\Desktop\Stuffs(IMP)\dataScience expotent files\Project (datascience)\A.csv"

final_stage1.to_csv(output_excel_path,index=False)
