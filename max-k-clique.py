import random
import json
import time
import math


class Graph:
    def __init__(self, num_of_vertices) -> None:
        self.graph = [([False] * num_of_vertices)
                      for _ in range(num_of_vertices)]

    def add_edge(self, u, v) -> None:
        if u == v:
            return
        self.graph[u][v] = self.graph[v][u] = True

    def contain_edge(self, u, v) -> bool:
        return self.graph[u][v]

    def num_of_vertices(self) -> int:
        return len(self.graph)

    @staticmethod
    def load_from_json(path):
        data = json.load(open(path))
        graph = Graph(data["vertices"] + 1)
        edges = data["edges"]

        for edge in edges:
            graph.add_edge(edge[0], edge[1])

        return graph


class Population:
    def __init__(self, graph, population_size, mutation_rate) -> None:
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.graph = graph
        self.population = []
        self.matching_pool = []
        self.generation = 1
        self.cutpoint = graph.num_of_vertices() // 2

        for _ in range(self.population_size):
            self.population.append(Chromosome(
                num=graph.num_of_vertices(), random_generate=True, graph=graph))

    def get_max_clique(self):
        max_fitness = 0
        max_genes = None
        for chromo in self.population:
            fitness = sum(chromo.genes)
            if fitness > max_fitness:
                max_fitness = fitness
                max_genes = chromo.genes

        clique = [v for v in range(len(max_genes)) if max_genes[v] == 1]
        return clique

    def natural_selection(self):
        self.matching_pool = []
        for chromosome in self.population:
            self.matching_pool.append(chromosome)
            for _ in range(chromosome.fitness):
                self.matching_pool.append(chromosome)

    def generate(self):
        new_population = [] + self.population
        for _ in range(len(self.population)):
            a = random.randint(0, len(self.matching_pool) - 1)
            b = random.randint(0, len(self.matching_pool) - 1)

            parent_a = self.matching_pool[a]
            parent_b = self.matching_pool[b]

            offsprings = [parent_a.crossover(
                parent_b, self.cutpoint), parent_b.crossover(parent_a, self.cutpoint)]

            for offspring in offsprings:
                offspring.mutate(self.mutation_rate)
                offspring.extract_clique(self.graph)
                offspring.random_expand_clique(self.graph)
                offspring.calc_fitness()
                new_population.append(offspring)

        new_population.sort(key=lambda chromo: chromo.fitness, reverse=True)
        self.population = new_population[:self.population_size]
        self.generation += 1

    def update_param(self):
        if self.generation % 20 == 0:
            self.cutpoint -= 1
            self.mutation_rate -= 0.01
            self.cutpoint = max(2, self.cutpoint)
            self.mutation_rate = max(0.05, self.mutation_rate)

    def calc_fitness(self):
        for chromosome in self.population:
            chromosome.calc_fitness()


class Chromosome:
    def __init__(self, num, random_generate=False, graph=None) -> None:
        self.genes = [0] * num
        self.fitness = 0

        if random_generate:
            self.random_generate(graph)

    def random_generate(self, graph):
        n_vertices = graph.num_of_vertices()
        u = random.randint(0, n_vertices - 1)
        self.expand_clique_on_vertex(u, graph)

    def random_expand_clique(self, graph):
        vertices = [v for v in range(
            graph.num_of_vertices()) if self.genes[v] == 1]
        if len(vertices) == 0:
            vertices.append(random.randint(0, graph.num_of_vertices() - 1))
        self.expand_clique_on_vertex(
            vertices[random.randint(0, len(vertices) - 1)], graph)

    def expand_clique_on_vertex(self, u, graph):
        self.genes[u] = 1

        adjacent_vertices = []
        for v in range(graph.num_of_vertices()):
            if self.genes[v] == 1:
                continue

            if graph.contain_edge(u, v):
                adjacent_vertices.append(v)

        for _ in range(len(adjacent_vertices)):
            v = random.randint(0, len(adjacent_vertices) - 1)

            while adjacent_vertices[v] == -1:
                v += 1
                v %= len(adjacent_vertices)

            selected_vertex = adjacent_vertices[v]
            adjacent_vertices[v] = -1

            can_form_clique = True
            for u in range(graph.num_of_vertices()):
                if self.genes[u] == 0:
                    continue

                if not graph.contain_edge(u, selected_vertex):
                    can_form_clique = False
                    break

            if can_form_clique:
                self.genes[selected_vertex] = 1

    def calc_fitness(self):
        self.fitness = sum(self.genes)

    def crossover(self, partner, cutpoint):
        child = Chromosome(len(self.genes))
        for i in range(len(self.genes)):
            if i < cutpoint:
                child.genes[i] = self.genes[i]
            else:
                child.genes[i] = partner.genes[i]

        return child

    def mutate(self, mutationRate):
        for i in range(len(self.genes)):
            if random.random() < mutationRate:
                self.genes[i] ^= 1

    def extract_clique(self, graph):
        vertices = []
        for i in range(len(self.genes)):
            if self.genes[i] == 1:
                vertices.append((i, sum(graph.graph[i])))

        vertices.sort(key=lambda tup: tup[1], reverse=True)

        while not self.isClique(graph):
            self.genes[vertices.pop()[0]] = 0

    def isClique(self, graph):
        vertices = []
        for i in range(len(self.genes)):
            if self.genes[i] == 1:
                vertices.append(i)

        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                if not graph.contain_edge(vertices[i], vertices[j]):
                    return False
        return True


def main() -> None:
    graph = Graph.load_from_json("171_vertices_11_clique.json")
    stagnancy = 100
    population_size = 50
    mutation_rate = 0.5

    population = Population(
        graph=graph,
        population_size=population_size,
        mutation_rate=mutation_rate)

    max_fitness = 0

    start_time = time.time()

    for i in range(stagnancy):
        population.calc_fitness()
        population_fitness = len(population.get_max_clique())
        if population_fitness > max_fitness:
            max_fitness = population_fitness
            i = 0

        population.natural_selection()
        population.generate()
        population.update_param()

    population.calc_fitness()
    end_time = time.time()

    max_clique = population.get_max_clique()
    print("Max clique:", max_clique)
    print("Clique size:", len(max_clique))
    print("Total number of generation:", population.generation)
    print("Time elapsed:", math.ceil(end_time - start_time), "s")


if __name__ == "__main__":
    main()
