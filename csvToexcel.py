import pandas as pd

# Define input and output file paths
csv_file_path = r'E:\College\DSC_Hack_hybrid\data.csv'
excel_file_path =r'E:\College\DSC_Hack_hybrid\data.xlsx'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path)

# Convert the DataFrame to an Excel file, setting index=False to avoid writing the pandas index to the Excel sheet
df.to_excel(excel_file_path, index=False, header=True)

print(f"Successfully converted {csv_file_path} to {excel_file_path}")