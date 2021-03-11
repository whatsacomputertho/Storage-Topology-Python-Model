import numpy as np
import sympy as sp
from itertools import combinations

class Node:
    def __init__(self, capacity):
        self.objects = []
        self.coefficients = []
        self.capacity = capacity

    def get_capacity(self):
        return self.capacity

    def set_capacity(self, capacity):
        self.capacity = capacity

    def get_objects(self):
        return self.objects

    def set_objects(self, objects):
        self.objects = objects

    def get_coefficients(self):
        return self.coefficients

    def set_coefficients(self, coefficients):
        self.coefficients = coefficients

class Edge:
    def __init__(self, weight, nodes):
        self.weight = weight
        self.nodes = nodes

    def get_weight(self):
        return self.weight

    def set_weight(self, weight):
        self.weight = weight

    def get_nodes(self):
        return self.nodes

    def set_nodes(self, nodes):
        self.nodes = nodes

class Object:
    def __init__(self, name, size):
        self.name = name
        self.size = size

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    def get_size(self):
        return self.size

    def set_size(self, size):
        self.size = size

class StorageTopology:
    def __init__(self, capacity, size, num_nodes, num_objects, weights):
        self.nodes = []
        self.objects = []
        self.edges = []
        for i in range(num_nodes):
            self.nodes.append(Node(capacity))
        for i in range(num_objects):
            o_name = "X{index:d}"
            o_name = o_name.format(index = i)
            self.objects.append(Object(o_name, size))
        k = 0
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                e = [self.nodes[i], self.nodes[j]]
                self.edges.append(Edge(weights[k], e))
                k += 1

    def read_weights(filepath):
        weights=[]
        file = open(filepath)
        for line in file.readlines():
            weights.append(float(line))
        return weights
        
    read_weights = staticmethod(read_weights)

    def generate_all_replicative_allocation_patterns(self):
        patterns = []
        for i in range(len(self.objects)**len(self.nodes)):
            dec_pattern = i
            pattern = []
            j = len(self.nodes) - 1
            while j >= 0:
                base_n_place_value = 0
                power = len(self.objects)**j
                while dec_pattern >= power:
                    dec_pattern -= power
                    base_n_place_value += 1
                pattern.append(base_n_place_value)
                j -= 1
            patterns.append(pattern)
        return patterns

    def generate_all_replicative_allocations(self):
        allocations = []
        for pattern in self.generate_all_replicative_allocation_patterns():
            allocation = []
            for i in range(len(pattern)):
                allocation.append(self.objects[pattern[i]])
            allocations.append(allocation)
        return allocations

    def filter_feasible_replicative_allocations(self, all_allocations):
        for x in self.objects:
            i = len(all_allocations) - 1
            while i >= 0:
                has_x = False
                for y in all_allocations[i]:
                    if x is y:
                        has_x = True
                if has_x == False:
                    all_allocations.pop(i)
                i -= 1
        feasible_allocations = all_allocations
        return feasible_allocations

    def clear_objects_stored(self):
        for node in self.nodes:
            node.objects.clear()
            node.coefficients.clear()

    def allocate_objects_replicatively(self, allocation):
        self.clear_objects_stored()
        for i in range(len(allocation)):
            objects = []
            objects.append(allocation[i])
            self.nodes[i].objects = objects

    def allocate_coded_objects(self, allocation):
        self.clear_objects_stored()
        for node in self.nodes:
            node.objects = self.objects
        for i in range(len(allocation)):
            self.nodes[i].coefficients = allocation[i]

    def check_coded_storage_feasibility(self):
        coefficient_matrix = []
        for node in self.nodes:
            coefficient_matrix.append(node.coefficients)
        if np.linalg.matrix_rank(coefficient_matrix) == len(self.objects):
            return True
        else:
            return False

    def generate_coded_storage_coefficient_matrices(self, k):
        k_combinations = combinations(self.nodes, k)
        matrices = []
        for combination in k_combinations:
            matrix = []
            for node in combination:
                matrix.append(node.coefficients)
            matrices.append(matrix)
        return matrices

    def check_matrix_solvable(self, matrix, index):
        for i in range(matrix.shape[0]):
            indices = []
            for j in range(len(matrix.row(i))):
                if matrix.row(i)[j] != 0:
                    indices.append(j)
            if len(indices) == 1 and indices[0] == index:
                return  True
        return False

    def get_node_set_from_coefficient_matrix(self, matrices):
        sets = []
        for i in range(len(matrices)):
            nodes = []
            for j in range(len(matrices[i])):
                for node in self.nodes:
                    if node.coefficients is matrices[i][j]:
                        nodes.append(node)
            sets.append(nodes)
        return sets

    def generate_coded_storage_retrieval_set_by_object(self, matrices, obj):
        solvable_matrices = []
        obj_index = self.objects.index(obj)
        for i in range(len(matrices)):
            if self.check_matrix_solvable(sp.Matrix(matrices[i]).rref()[0], obj_index):
                not_superlist = True
                for j in range(len(solvable_matrices)):
                    if(all(row in matrices[i] for row in solvable_matrices[j])):
                        not_superlist = False
                if not_superlist:
                    solvable_matrices.append(matrices[i])
        return self.get_node_set_from_coefficient_matrix(solvable_matrices)

    def calculate_replicative_retrieval_latencies_by_node_and_object(self, node, obj):
        latencies = []
        for n in self.nodes:
            if n.objects[0] is obj:
                for edge in self.edges:
                    if (((edge.nodes[0] is node) and (edge.nodes[1] is n)) or ((edge.nodes[0] is n) and (edge.nodes[1] is node))):
                        latencies.append(edge.weight)
                    elif node is n:
                        latencies.append(0.0)
        return latencies

    def calculate_minimum_latency_from_retrieval_list(self, latencies):
        min_latency = latencies[0]
        for latency in latencies:
            if latency < min_latency:
                min_latency = latency
        return min_latency

    def calculate_minimum_replicative_latencies_by_node(self, node):
        min_latencies = []
        for obj in self.objects:
            min_latencies.append(self.calculate_minimum_latency_from_retrieval_list(self.calculate_replicative_retrieval_latencies_by_node_and_object(node, obj)))
        return min_latencies

    def calculate_average_replicative_latency_by_object(self, obj):
        latency_sum = 0.0
        for node in self.nodes:
            latency_sum += self.calculate_minimum_latency_from_retrieval_list(self.calculate_replicative_retrieval_latencies_by_node_and_object(node, obj))
        return (latency_sum / len(self.nodes))

    def calculate_worst_case_replicative_latency_by_node(self, node):
        min_latencies = self.calculate_minimum_replicative_latencies_by_node(node)
        worst_case_latency = min_latencies[0]
        for f in min_latencies:
            if f > worst_case_latency:
                worst_case_latency = f
        return worst_case_latency

    def calculate_minimum_replicative_latencies(self):
        min_latencies = []
        for node in self.nodes:
            min_latencies.append(self.calculate_minimum_replicative_latencies_by_node(node))
        return min_latencies

    def calculate_average_replicative_latencies(self):
        average_latencies = []
        for obj in self.objects:
            average_latencies.append(self.calculate_average_replicative_latency_by_object(obj))
        return average_latencies

    def calculate_worst_case_replicative_latency(self):
        worst_case_latencies = []
        for node in self.nodes:
            worst_case_latencies.append(self.calculate_worst_case_replicative_latency_by_node(node))
        worst_case_latency = worst_case_latencies[0]
        for f in worst_case_latencies:
            if f > worst_case_latency:
                worst_case_latency = f
        return worst_case_latency

    def calculate_average_replicative_latency(self):
        latency_sum = 0.0
        average_replicative_latencies = self.calculate_average_replicative_latencies()
        for f in average_replicative_latencies:
            latency_sum += f
        return (latency_sum / len(average_replicative_latencies))
