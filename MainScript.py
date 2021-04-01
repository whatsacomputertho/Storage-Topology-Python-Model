from StorageTopology import Node, Edge, Object, StorageTopology
import sympy as sp
import csv

topology = StorageTopology(1.0, 1.0, 5, 3, StorageTopology.read_weights("weights.txt"))
#for allocation in topology.filter_feasible_replicative_allocations(topology.generate_all_replicative_allocations()):
#    topology.allocate_objects_replicatively(allocation)
#    alloc = ""
#    for node in topology.nodes:
#        alloc += node.objects[0].name
#    print("Allocation: {a} Worst Case Latency: {wc_lat}, Average Latency: {avg_lat}".format(a=alloc, wc_lat=topology.calculate_worst_case_replicative_latency(), avg_lat=topology.calculate_average_replicative_latency()))

alloc = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1],
        [1, 1, 0]
    ]

lines = []
lines.append("ID,N1 Contents,N2 Contents,N3 Contents,N4 Contents,N5 Contents,Worst Case Latency,Average Latency\n")
i = 0
for permutation in StorageTopology.permute(alloc):
    topology.allocate_coded_objects(permutation)
    topology.generate_all_coded_storage_coefficient_matrices()

#    for obj in topology.objects:
#        print("Retreival set for object: {obj_name}".format(obj_name=obj.name))
#        for node_set in topology.generate_coded_storage_retrieval_set_by_object(obj):
#            node_set_string = ""
#            for node in node_set:
#                node_index = "{node_ind}".format(node_ind=topology.nodes.index(node))
#                node_set_string += node_index
#            print(node_set_string)

    lines.append("{alloc_id},{allocation},{wc_lat},{avg_lat}\n".format(alloc_id = i, allocation = topology.get_allocation(), wc_lat = topology.calculate_worst_case_coded_latency(), avg_lat = topology.calculate_average_coded_latency()))
    i += 1

f = open("effectiveness-2.csv", "x")
for line in lines:
    f.write(line)
f.close()
print("done")
