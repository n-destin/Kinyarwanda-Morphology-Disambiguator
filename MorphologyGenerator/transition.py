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
        self.morpheme_mapping = {}
        self.morpheme_count = -1
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
                if line[0] == "+":
                    self.process_segmentation(line)
        for node in self.graph:
            self.normalize_node(node)
    
    def normalize_node(self, node):
        if node == None:
            return
        counts = [morpheme.count for morpheme in list(node.morphemes)]
        counts_ = [count * 500 / max(counts) for count in counts]
        counts = counts_
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
            if real_morpheme not in self.morpheme_mapping.keys():
                self.morpheme_mapping[real_morpheme] = self.morpheme_count + 1
                self.morpheme_count += 1
            
            if not root_processed and appending.morpheme != "root":
                appending.type = "prefix"
            elif root_processed and appending.morpheme != "root":
                appending.type = "suffix"
            if index < len(self.graph):
                if appending in list(self.graph[index].morphemes):
                    new_morphemes = []
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
        if len(current.split("+")) >= 8:
            return
        if len(morphemes) > 5000:
            return
        if index >= len(self.graph):
            if root_processed:
                morphemes.append(current)
            return
        
        past = root_processed
        range = sorted(self.graph[index].morphemes, key=lambda morpheme: morpheme.probability)[:5] if index > 2 else self.graph[index].morphemes
        for morpheme in range:
            new_current = ""
            if morpheme.morpheme == "root" and not root_processed:
                new_current = current + "+" + root
                root_processed = True
            else:
                if not root_processed and morpheme.type == "prefix":
                    new_current = current + "+" + morpheme.morpheme if morpheme.morpheme not in current.split("+") else ""
                elif root_processed and morpheme.type == "suffix":
                    new_current = current + "+" + morpheme.morpheme if morpheme.morpheme not in current.split("+") else ""
            if morpheme.ending and root_processed and len(new_current) > 0:
                morphemes.append(new_current)
            else:
                next = new_current if len(new_current) > 0 else current
                self.get_inflection_helper(root, next, index + 1, morphemes, root_processed)
                # self.get_inflection_helper(root, current, index + 1, morphemes, past)


    def get_inflections(self, root):
        inflections = []
        self.get_inflection_helper(root, "", 0, inflections, False)
        return set(inflections)
                

    def print_transition_graph(self):
        for item in self.graph[1:]:
            print("\nmorphemes index : " + str(item.index_holder)  + "\n")
            for morpheme in item.morphemes:
                print(morpheme.probability, morpheme.morpheme)

graph = TransitionGraph("testinggraph.txt")