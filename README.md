## Ambulance Location Optimization

**Description**

This project focuses on optimizing ambulance locations in Aguascalientes, Mexico, using accident data from 2023. The objective is to maximize the number of accidents covered by a limited number of ambulances, considering a coverage radius of 2 km.

The project uses the **GRASP (Greedy Randomized Adaptive Search Procedure)** metaheuristic combined with local search to find effective ambulance locations.

## Objectives

- Optimize ambulance locations based on historical accident data.

- Maximize the number of accidents covered within a 2 km radius.

- Compare coverage results using 3, 5, 7, and 10 ambulances.

- Apply GRASP and local search to solve the Maximum Coverage Problem (MCP).

## Input Data

The accident data comes from INEGI's 2023 traffic accident dataset. The original Excel data was processed to generate accidentes_Ags_2023.csv, which contains the accident records used by the program.

The dataset includes latitude and longitude coordinates, which are used to calculate distances between accidents and candidate ambulance locations.

## Methodology

The optimization process consists of the following steps:

1. Candidate generation: Creates a grid of potential ambulance locations based on the geographic coordinates of the accident dataset.
2. Coverage calculation: Uses the Haversine formula to calculate geographic distances and determine which accidents fall within a 2 km radius of each candidate location.
3. GRASP construction: Iteratively selects ambulance locations based on the number of additional accidents they can cover. A Restricted Candidate List (RCL) introduces randomization into the selection process.
4. Local search: Attempts to improve each solution by replacing an ambulance location with another candidate that increases total accident coverage.
5. Solution evaluation: Repeats the optimization process 40 times for each ambulance configuration and retains the solution with the highest coverage.

## Howw to run
1. Install Python 3 and the required libraries: python -m pip install pandas numpy
2. Place accidentes_Ags_2023.csv in the same directory as the Python script.
3. Run the script: python name_script.py
4. The program evaluates each ambulance configuration and prints the number and percentage of accidents covered.

## Output
The program generates a CSV file for each ambulance configuration containing the selected ambulance coordinates:
- solucion_mcp_3ambulancias_2km.csv
- solucion_mcp_5ambulancias_2km.csv
- solucion_mcp_7ambulancias_2km.csv
- solucion_mcp_10ambulancias_2km.csv

These files contain the latitude and longitude of the selected ambulance locations and can be used to visualize the results on a map.

## Limitations

The results depend on the quality of the accident data, the candidate grid, and the 2 km coverage radius. GRASP with local search is a heuristic approach and does not guarantee a globally optimal solution.

The estimated coverage is based on geographic distance rather than actual road networks, traffic conditions, or emergency response times.
