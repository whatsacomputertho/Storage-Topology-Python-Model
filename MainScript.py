from StorageTopology import Node, Edge, Object, StorageTopology
import sympy as sp

topology = StorageTopology(1.0, 1.0, 5, 3, StorageTopology.read_weights("weights.txt"))
allocation = [
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 1],
        [1, 1, 0],
        [0, 0, 1]
    ]
topology.allocate_coded_objects(allocation)

matrices = []
for k in range(1, len(topology.nodes) + 1):
    for matrix in topology.generate_coded_storage_coefficient_matrices(k):
        matrices.append(matrix)

for obj in topology.objects:
    print("Retreival set for object: {obj_name}".format(obj_name=obj.name))
    for node_set in topology.generate_coded_storage_retrieval_set_by_object(matrices, obj):
        node_set_string = ""
        for node in node_set:
            node_index = "{node_ind}".format(node_ind=topology.nodes.index(node))
            node_set_string += node_index
        print(node_set_string)
