

import pandas as pd
import numpy as np
from pulp import *

# Read the Excel file
excel_file = r"C:\Users\sumair\OneDrive\Desktop\Hyderabad Dispatch Report 22-23.xlsx"

# Create an ExcelFile object
xls = pd.ExcelFile(excel_file)

# Get the sheet names
sheet_names = xls.sheet_names

# Create an empty dictionary to store the dataframes for each sheet
dataframes = {}
dataframes1 = {}
dataframes2 = {}
dataframes3 = {}
dataframes4 = {}
dataframes5 = {}
dataframes6 = {}

# Iterate over each sheet and read the data into a dataframe
for i in range(7):  # Iterate from 0 to 6
    sheet_name = sheet_names[i]
    if sheet_name == "Girmapur":
        dataframes[sheet_name] = pd.read_excel(xls, sheet_name)
    elif sheet_name == "Girmapur2":
        dataframes1[sheet_name] = pd.read_excel(xls, sheet_name)
    elif sheet_name == "RSDSH":
        dataframes2[sheet_name] = pd.read_excel(xls, sheet_name)
    elif sheet_name == "LKDRM2":
        dataframes3[sheet_name] = pd.read_excel(xls, sheet_name)
    elif sheet_name == "LKDRM4":
        dataframes4[sheet_name] = pd.read_excel(xls, sheet_name)
    elif sheet_name == "KSR3":
        dataframes5[sheet_name] = pd.read_excel(xls, sheet_name)
    elif sheet_name == "SLKPY":
        dataframes6[sheet_name] = pd.read_excel(xls, sheet_name)

# Concatenate the dataframes
dfs = [df for dict_name in [dataframes, dataframes1, dataframes2, dataframes3, dataframes4, dataframes5, dataframes6] for df in dict_name.values()]
result = pd.concat(dfs, ignore_index=True)



# Drop unwanted columns from the concatenated dataframe
columns_to_drop = ["DocNum", "DocType", "FORT NITE", "Posting Date", "DCNo", "Customer/Vendor Code",
                   "Customer/Vendor Name", "Item Description", "VehicleNO", "SlpName", "Type",
                   "RoboQty", "Months", "Segment", "Delivery Date"]
result = result.drop(columns=columns_to_drop)

# Drop rows with missing values
result.dropna(inplace=True)
result.isna().sum()

# Extract required columns
data2 = result[["Distribution Rule", "Transporter Name", "Freight Rate"]].drop_duplicates()
data3 = result[["Distribution Rule", "Transporter Name", "Quantity", "U_KM","Amount"]].copy()
data3.dropna(inplace=True)  # Drop rows with missing values in data3


# Initialize Total Cost column
data3["Total Cost"] = 0

# Iterate over each row in data3
for index, row in data3.iterrows():
    condition = ((data2['Distribution Rule'] == row['Distribution Rule']) & (data2['Transporter Name'] == row['Transporter Name'])) & (data2['Freight Rate'] < 5.0)
    if condition.any():
        # Condition met, calculate Total Cost
        data3.at[index, "Total Cost"] = row['Quantity'] * row['U_KM'] * data2.loc[condition, 'Freight Rate'].values[0]
    else:
        condition = ((data2['Distribution Rule'] == row['Distribution Rule']) & (data2['Transporter Name'] == row['Transporter Name'])) & (data2['Freight Rate'] >= 5.0)
        if condition.any():
            data3.at[index, "Total Cost"] = data2.loc[condition, 'Freight Rate'].values[0]

# Calculate the Total cost
total_cost = data3["Total Cost"].sum()
#print("Total Cost:", total_cost)




# Create a list of unique warehouses and parties
warehouses9 = result['Distribution Rule'].unique()
parties9 = result['Transporter Name'].unique()

# Create the supply and demand dictionaries
supply9 = {}
demand9 = {}

# Calculate the supply and demand based on the dataset
for i, row9 in result.iterrows():
    source9 = row9['Distribution Rule']
    destination9 = row9['Transporter Name']
    supply9[source9, destination9] = supply9.get((source9, destination9), 0) + row9['Quantity']
    demand9[source9, destination9] = demand9.get((source9, destination9), 0) + row9['Quantity']

 

# Create the LP problem
prob9 = LpProblem("Minimize_Transportation_Cost", LpMinimize)

# Create decision variables
routes9 = LpVariable.dicts("Route", (warehouses9, parties9), lowBound=0, cat='Continuous')

# Set the objective function
costs9 = result.set_index(['Distribution Rule', 'Transporter Name'])['Freight Rate'].to_dict()
prob9 += lpSum([costs9[(i, j)] * routes9[i][j] for i in warehouses9 for j in parties9 if (i, j) in costs9])

# Add supply constraints -> are added to LP -> total outgoing supply of each warehouse matches the supply values
for i in warehouses9:
    for j in parties9:
        prob9 += lpSum([routes9[i][j]]) == supply9.get((i, j), 0)

# Add demand constraints
for j in parties9:
    for i in warehouses9:
        prob9 += lpSum([routes9[i][j]]) == demand9.get((i, j), 0)

# Solve the LP problem
prob9.solve()

# Print the optimal solution -> stored np array <- each element represents opt goods transported from warehouse to party
solution9 = np.zeros((len(warehouses9), len(parties9)))
for i, warehouse9 in enumerate(warehouses9):
    for j, party9 in enumerate(parties9):
        solution9[i][j] = routes9[warehouse9][party9].varValue

# Create a DataFrame to store the solution matrix
solution_df9 = pd.DataFrame(solution9, index=warehouses9, columns=parties9)

# Reshape the cost matrix into np array for matching the dimensions of the solution array
reshaped_costs9 = np.zeros((len(warehouses9), len(parties9)))
for i, warehouse9 in enumerate(warehouses9):
    for j, party9 in enumerate(parties9):
        if (warehouse9, party9) in costs9:
            reshaped_costs9[i][j] = costs9[(warehouse9, party9)]
            
# Create a DataFrame to store the reshaped_costs matrix
reshaped_costs_df9 = pd.DataFrame(reshaped_costs9, index=warehouses9, columns=parties9)

print("")

# print the Total cost
print("Total Cost:           ", total_cost)

# Calculate the total minimized cost
total_minimized_cost9 = np.sum(solution9 * reshaped_costs9)
print("Total Minimized Cost: ", total_minimized_cost9)

# Calculate the overall cost
overall_cost = total_cost - total_minimized_cost9
print("Overall Cost:         ", overall_cost)

# Calculate the percentage
percentage = (total_minimized_cost9 / total_cost) * 100
print("Percentage:           ", percentage)


