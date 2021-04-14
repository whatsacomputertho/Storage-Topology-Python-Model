from StorageTopology import Node, Edge, Object, StorageTopology
import sympy as sp
import csv

topology = StorageTopology(1.0, 1.0, 13, 3, StorageTopology.read_weights("weights.txt"))

alloc = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1],
        [1, 1, 0]
    ]

i = 0
allocations = topology.filter_feasible_replicative_allocations(topology.generate_all_replicative_allocations())
for allocation in allocations:
    topology.allocate_coded_objects(allocation)
    f = open("aws-effectiveness-replication-1.csv", "x")
    line = "{alloc_id},{allocation},{wc_lat},{avg_lat}".format(alloc_id = i, allocation = topology.get_allocation(), wc_lat = topology.calculate_worst_case_coded_latency(), avg_lat = topology.calculate_average_coded_latency())
    f.write(line + "\n")
    f.close()
    print(line)
    i += 1
    
print("done")
