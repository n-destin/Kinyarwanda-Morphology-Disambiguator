from helper import stopset_kin, vowels, consonants

def process_word(word):
    to_return = ""
    for char in word:
        if char not in vowels and char not in consonants:
            if to_return in stopset_kin:
                return word[len(to_return) + 1 : len(word)]
            return process_word(word[len(to_return) + 1 : len(word)])
        else:
            to_return += char
    return to_return

    
def prepare(file_name, output_file):
    words = set()
    with open(file_name, "r") as file:
        lines = file.readlines()
        for line in lines[1:len(lines)]:
            splitted = line.split(',')[-1].split(" ")
            for word in splitted[0:len(splitted)-1]:
                processed_word = process_word(word)
                if len(processed_word) > 0:
                    processed_word = processed_word[0:len(processed_word) - 1] if processed_word[-1] == "." else processed_word
                words.add(processed_word)
    with open(output_file, "w") as output:
        for word in words:
            output.write(word + "\n")


with open("../MorphologyGenerator/word_to_root.txt", "r") as file:
    lines =  file.readlines()
    with open("corpus.txt", "w") as out:
        for line in lines:
            word, root = line.split(",")
            out.write(word + "\n")


prepare("train.csv", "corputs.txt")