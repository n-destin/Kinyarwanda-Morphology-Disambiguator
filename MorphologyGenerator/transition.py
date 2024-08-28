'''
@Author: Destin Niyomufasha
@Learning Intelligence and Signal Processing Lab
@Kinyarwanda NLP

This transition graph is built to help in constructing possible inflection form a root word. 
@It improves the morphogramephic graph proposed in the disambiguation of Kinyarwanda morphology analyzers paper 
'''
# import torch
import math

class Morpheme():
    def __init__(self, morpheme, type, ending = False, count = 1):
        self.morpheme = morpheme
        self.ending = ending
        self.count = count
        self.type = type # if it is an affix or suffix
        self.probability = 0

    def __hash__(self) -> int:
        return hash(self.morpheme)
    
    def __str__(self):
        return  "morpheme:" + self.morpheme +  "\n  probability " + str(self.probability )+ "\n  count: " + str(self.count) + "\n  ending morpheme? : " + str(self.ending)
    
    def __eq__(self, comparing):
        return comparing.morpheme == self.morpheme


class GraphNode():
    def __init__(self, index):
        self.index_holder = index
        self.morphemes = set()

    def __hash__(self) -> int:
        return hash(self.index_holder)


class TransitionGraph():
    def __init__(self, inputFile):
        self.graph = []
        self.starting_nodes = []
        self.build_transition_graph(inputFile)

    def softmax(self, nums):
        den = 0
        for number in nums:
            den += math.exp(number)
        return [math.exp(count) / den for count in nums]



    def build_transition_graph(self, inputFile):
        if inputFile == None:
            raise KeyError("Please provide a valid input file")
        with open(inputFile, "r") as segmentatioons:
            for line in segmentatioons.readlines():
                self.process_segmentation(line)
        # normalize nodes
        for node in self.graph:
            self.normalize_node(node)
    
    def normalize_node(self, node):
        if node == None:
            return
        counts = [morpheme.count for morpheme in list(node.morphemes)]
        probabilities = self.softmax(counts)
        new_list = list(node.morphemes)
        # print(probabilities, type(probabilities))
        for index in range(len(probabilities)):
            new_list[index].probability = probabilities[index] 
        node.morphemes = set(new_list)


    def process_segmentation(self, segementation):
        root_processed = False
        segments = segementation[1:len(segementation)].split("+")
        for index in range(len(segments)):
            current = GraphNode(index)
            appending_morpheme = segments[index] if segments[index].isupper() else 'root'
            real_morpheme = appending_morpheme[0:len(appending_morpheme) - 1] if appending_morpheme[len(appending_morpheme) - 1] == "\n" else appending_morpheme
            appending = Morpheme(real_morpheme, index == len(segments) - 1, appending_morpheme != real_morpheme)
            if not root_processed:
                appending.type = "prefix"
            else:
                appending.type = "suffix"
            if index < len(self.graph): # this means that if the index has already been visited before. meaning we have some morphemes there
                if appending in list(self.graph[index].morphemes):
                    new_morphemes = []
                    # print("checking morphemes", index)
                    # for morpheme in list(self.graph[index].morphemes):
                    #     print(morpheme.count, morpheme.morpheme)
                    for changing in list(self.graph[index].morphemes):
                        if changing == appending:
                            changing.count += 1
                        new_morphemes.append(changing)
                    self.graph[index].morphemes = set(new_morphemes)

                else:
                    self.graph[index].morphemes.add(appending) # this adds the appending
            else:
                current.morphemes.add(appending)
                self.graph.append(current)
            if appending.morpheme == "root":
                root_processed = True

    def get_inflection_helper(self, root, current, index, morphemes, root_processed):
        if index >= len(self.graph):
            if root_processed:
                morphemes.append(current)
            return 
        for morpheme in self.graph[index].morphemes:
            new_current = ""
            if morpheme.morpheme == "root" and not root_processed:
                new_current = current + "+" + root
                root_processed = True
            else:
                if not root_processed and morpheme.type == "prefix":
                    new_current = current + "+" + morpheme.morpheme
                elif root_processed and morpheme.type == "suffix":
                    new_current = current + "+" + morpheme.morpheme
            # print(new_current, morpheme.ending, morpheme.morpheme)
            if morpheme.ending and root_processed:
                # print("morpheme ending", morpheme, new_current)
                morphemes.append(new_current)
            else:
                # print("notn ending", morpheme, new_current)
                self.get_inflection_helper(root, new_current, index + 1, morphemes, root_processed)


    def get_inflections(self, root):
        inflections = []
        self.get_inflection_helper(root, "", 0, inflections, False)
        return inflections
                

    def print_transition_graph(self):
        for item in self.graph:
            print("\nmorphemes index : " + str(item.index_holder)  + "\n")
            for morpheme in item.morphemes:
                print(morpheme)


graph = TransitionGraph("testinggraph.txt")
graph.print_transition_graph()
returned = graph.get_inflections("reng")
# print(returned, len(returned))
