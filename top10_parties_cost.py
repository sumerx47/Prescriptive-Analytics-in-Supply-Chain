import pandas as pd
import numpy as np
from pulp import *

df = pd.read_excel(r'C:\Users\sumair\OneDrive\Desktop\1 Year Data.xlsx')

# Filter out the rows with warehouses 'RSGIR' , 'LKDRM3' and 'GIR II Trading'
data = df[~((df['Warehouse'] == 'LKDRM4') | (df['Warehouse'] == 'LKDRM3') | (df['Warehouse'] == 'LKDRM2') | (df['Warehouse'] == 'SLKPY') | (df['Warehouse'] == 'RSDSH') | (df['Warehouse'] == 'KSR3'))]


# Group data by party and calculate the count of occurrences for each party
party_counts = data['Party Name'].value_counts().reset_index()

# Rename the columns for clarity
party_counts.columns = ['Party Name', 'Count']

# Sort parties based on the count in descending order
sorted_parties = party_counts.sort_values(by='Count', ascending=False)

# Get the top 10 party names
top_10_parties = sorted_parties['Party Name'].head(10)
      



# Create a list of unique warehouses and parties
warehouses = data['Warehouse'].unique()
parties = data['Party Name'].unique()

# Create the supply and demand dictionaries
supply = {}
demand = {}

# Calculate the supply and demand based on the dataset
for k, row in data.iterrows():
    source = row['Warehouse']
    destination = row['Party Name']  # Use parties_tuple instead of parties
    supply[source, destination] = supply.get((source, destination), 0) + row['Net Weight']
    demand[source, destination] = demand.get((source, destination), 0) + row['Net Weight']

    
data['Cost per MT per KM'] = 0
for i in data.index:
    if (data['Warehouse'][i] == 'GIR' or data['Warehouse'][i] == 'GIR II' or data['Warehouse'][i] == 'GIR II Trading') and data['Distance'][i] < 20:
        data.at[i, 'Cost per MT per KM'] = 100
    elif (data['Warehouse'][i] == 'GIR' or data['Warehouse'][i] == 'GIR II' or data['Warehouse'][i] == 'GIR II Trading') and data['Distance'][i] >= 20:
        data.at[i, 'Cost per MT per KM'] = 2.9
    elif (data['Warehouse'][i] == 'LKDRM4' or data['Warehouse'][i] == 'LKDRM3' or data['Warehouse'][i] == 'LKDRM2' or data['Warehouse'][i] == 'SLKPY') and data['Distance'][i] < 20:
        data.at[i, 'Cost per MT per KM'] = 100
    elif (data['Warehouse'][i] == 'LKDRM4' or data['Warehouse'][i] == 'LKDRM3' or data['Warehouse'][i] == 'LKDRM2' or data['Warehouse'][i] == 'SLKPY') and data['Distance'][i] >= 20:
        data.at[i, 'Cost per MT per KM'] = 2.4
    elif (data['Warehouse'][i] == 'RSDSH' or data['Warehouse'][i] == 'RSGIR') and data['Distance'][i] < 40:
        data.at[i, 'Cost per MT per KM'] = 100
    elif (data['Warehouse'][i] == 'RSDSH' or data['Warehouse'][i] == 'RSGIR') and data['Distance'][i] >= 40:
        data.at[i, 'Cost per MT per KM'] = 2.5
    elif (data['Warehouse'][i] == 'KSR3') and data['Distance'][i] < 20:
        data.at[i, 'Cost per MT per KM'] = 100
    elif (data['Warehouse'][i] == 'KSR3') and data['Distance'][i] >= 20:
        data.at[i, 'Cost per MT per KM'] = 2.9

#data['Total Cost'] =data.at[i, 'Cost per MT per KM']* data['Distance'] * data['Net Weight']
data["Total Cost"] = 0
for i in data.index:
    data.at[i, "Total Cost"] = data.at[i, 'Cost per MT per KM'] * data.at[i, 'Distance'] * data.at[i, 'Net Weight']

# Calculate the Total cost
print("Total Cost:", data["Total Cost"].sum())



# Calculate the total cost for the top 10 parties with warehouses
total_cost_top_10 = data[data['Party Name'].isin(top_10_parties)]['Cost per MT per KM'] * data['Distance'] * data['Net Weight']
total_cost = total_cost_top_10.sum()

# Print the total cost
print("Total Cost for Top 10 Parties with Warehouses:", total_cost)





# Create the LP problem
prob = LpProblem("Transportation Problem", LpMinimize)

# Create decision variables
routes = LpVariable.dicts("Route", (warehouses, parties), lowBound=0, cat='Integer')

# Define the objective function
#prob += is used to add constraints and objective function to the LP problem. augmented assignment operator that adds the right-hand side expression to the left-hand side variable.
#lpSum calculates the sum of the costs of all routes multiplied by their respective decision variables
costs = data.set_index(['Warehouse', 'Party Name'])['Cost per MT per KM'].to_dict()
prob += lpSum([costs[(i, j)] * routes[i][j] for i in warehouses for j in parties if (i, j) in costs])

# Add supply constraints -> are added to LP -> total outgoing supply of each warehouse matches the supply values
for i in warehouses:
    for j in parties:
        prob += lpSum([routes[i][j]]) == supply.get((i, j), 0)

# Add demand constraints
for j in parties:
    for i in warehouses:
        prob += lpSum([routes[i][j]]) == demand.get((i, j), 0)

# Solve the LP problem
prob.solve()

# Print the optimal solution -> stored np array <- each element represents opt goods transportedfrom warehouse to party
solution = np.zeros((len(warehouses), len(parties)))
for i, warehouse in enumerate(warehouses):
    for j, party in enumerate(parties):
        solution[i][j] = routes[warehouse][party].varValue

# Create a DataFrame to store the solution matrix
solution_df = pd.DataFrame(solution, index=warehouses, columns=parties)

# Reshape the cost matrix into np array for matching the dimensions of solution array
reshaped_costs = np.zeros((len(warehouses), len(parties)))
for i, warehouse in enumerate(warehouses):
    for j, party in enumerate(parties):
        if (warehouse, party) in costs:
            reshaped_costs[i][j] = costs[(warehouse, party)]

reshaped_costs_df = pd.DataFrame(reshaped_costs, index=warehouses, columns=parties)

# Calculate the total minimized cost
total_minimized_cost = np.sum(solution * reshaped_costs)
print("Total Minimized Cost:", total_minimized_cost)


#Total Cost - Total Minimized Cost 
print("overall Cost:",(total_cost - total_minimized_cost))


# Calculating the percentage 
print("percentage:",( total_minimized_cost/ data["Total Cost"].sum())*100)




