
def read_words(files, output_file):

    output = open(output_file, "w")
    for input_file in files:
        with open(input_file, "r") as file:
            for line in file.readlines():
                output.write(line.split(" ")[0].split("\t")[0] + "\n")

read_words(["../kin-morph-fst/token-decomp/words1.txt", "../kin-morph-fst/token-decomp/words2.txt"], "output.txt")
