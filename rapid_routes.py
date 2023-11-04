from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import pandas as pd
import numpy as np

# Create Dataframe of Matrix Distance
def create_excel(data):
    n_col = len(data['distance_matrix'][0])
    n_row = len(data['distance_matrix'])
    list_row = ['row' + str(i) for i in range(n_row)]
    list_col = ['col' + str(i) for i in range(n_row)]

    matrix = np.array(data['distance_matrix'])
    df = pd.DataFrame(data=matrix, index=list_row, columns=list_col)
    df.to_excel('df_distance_matrix.xlsx')


# Get user input for distance matrix
n_destinations = int(input("Enter the number of destinations: "))
print("Enter the distance matrix (separate values by spaces):")
distance_matrix = []
for i in range(n_destinations):
    row = list(map(int, input().split()))
    distance_matrix.append(row)
data = {'distance_matrix': distance_matrix}

# Get user input for demand and vehicle capacity
demands = list(map(int, input("Enter the demand for each destination (separate values by spaces): ").split()))
vehicle_capacities = list(map(int, input("Enter the vehicle capacity for each vehicle (separate values by spaces): ").split()))
num_vehicles = int(input("Enter the number of vehicles: "))
depot = int(input("Enter the index of the depot: "))
data.update({'demands': demands, 'vehicle_capacities': vehicle_capacities, 'num_vehicles': num_vehicles, 'depot': depot})

# Create Dataframe of Matrix Distance
create_excel(data)

# Transform to Numpy Array
distance_matrix = np.array(data['distance_matrix'])

# Create dictionnary with data
data = {}
data['distance_matrix'] = distance_matrix

# Orders quantity (Boxes)
data['demands'] = demands
# Vehicles Capacities (Boxes)
data['vehicle_capacities'] = vehicle_capacities
# Fleet informations
# Number of vehicles
data['num_vehicles'] = num_vehicles
# Location of the depot
data['depot'] = depot


def distance_callback(from_index, to_index):
    """Returns the distance between the two nodes."""
    # Convert from routing variable Index to distance matrix NodeIndex.
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return data['distance_matrix'][from_node][to_node]

def demand_callback(from_index):
    """Returns the demand of the node."""
    # Convert from routing variable Index to demands NodeIndex.
    from_node = manager.IndexToNode(from_index)
    return data['demands'][from_node]

# Create the routing index manager.
manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                       data['num_vehicles'], data['depot'])

# Create Routing Model
routing = pywrapcp.RoutingModel(manager)

# Create and register a transit callback.
transit_callback_index = routing.RegisterTransitCallback(distance_callback)

# Define cost of each arc.
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

# Add Capacity constraint.
demand_callback_index = routing.RegisterUnaryTransitCallback(
    demand_callback)
routing.AddDimensionWithVehicleCapacity(demand_callback_index,
    0,  # null capacity slack
    data['vehicle_capacities'],  # vehicle maximum capacities
    True,  # start cumul to zero
    'Capacity')

# Setting first solution heuristic
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
search_parameters.local_search_metaheuristic = (
    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
search_parameters.time_limit.FromSeconds(1)

# Solve the problem.
solution = routing.SolveWithParameters(search_parameters)

if solution:
    total_distance = 0
    total_load = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for driver {}:\n'.format(vehicle_id)
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            plan_output += ' {0} Parcels({1}) -> '.format(node_index, route_load)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += ' {0} Parcels({1})\n'.format(manager.IndexToNode(index),
                                                 route_load)
        plan_output += 'Distance of the route: {} (m)\n'.format(route_distance)
        plan_output += 'Parcels Delivered: {} (parcels)\n'.format(route_load)
        print(plan_output)
        total_distance += route_distance
        total_load += route_load
    print('Total distance of all routes: {:,} (m)'.format(total_distance))
    print('Parcels Delivered: {:,}/{:,}'.format(total_load, sum(data['demands'])))
else:
    print('No Solution')