# The tagger.py starter code for CSC384 A4.
# Currently reads in the names of the training files, test file and output file,
# and calls the tagger (which you need to implement)
import os
import sys
import numpy as np

# 91 POS tags, values storing the index in matrix
TAGS = {
    "AJ0": 0,
    "AJC": 1,
    "AJS": 2,
    "AT0": 3,
    "AV0": 4,
    "AVP": 5,
    "AVQ": 6,
    "CJC": 7,
    "CJS": 8,
    "CJT": 9,
    "CRD": 10,
    "DPS": 11,
    "DT0": 12,
    "DTQ": 13,
    "EX0": 14,
    "ITJ": 15,
    "NN0": 16,
    "NN1": 17,
    "NN2": 18,
    "NP0": 19,
    "ORD": 20,
    "PNI": 21,
    "PNP": 22,
    "PNQ": 23,
    "PNX": 24,
    "POS": 25,
    "PRF": 26,
    "PRP": 27,
    "PUL": 28,
    "PUN": 29,
    "PUQ": 30,
    "PUR": 31,
    "TO0": 32,
    "UNC": 33,
    "VBB": 34,
    "VBD": 35,
    "VBG": 36,
    "VBI": 37,
    "VBN": 38,
    "VBZ": 39,
    "VDB": 40,
    "VDD": 41,
    "VDG": 42,
    "VDI": 43,
    "VDN": 44,
    "VDZ": 45,
    "VHB": 46,
    "VHD": 47,
    "VHG": 48,
    "VHI": 49,
    "VHN": 50,
    "VHZ": 51,
    "VM0": 52,
    "VVB": 53,
    "VVD": 54,
    "VVG": 55,
    "VVI": 56,
    "VVN": 57,
    "VVZ": 58,
    "XX0": 59,
    "ZZ0": 60,
    "AJ0-AV0": 61,
    "AJ0-NN1": 62,
    "AJ0-VVD": 63,
    "AJ0-VVG": 64,
    "AJ0-VVN": 65,
    "AV0-AJ0": 66,
    "AVP-PRP": 67,
    "AVQ-CJS": 68,
    "CJS-AVQ": 69,
    "CJS-PRP": 70,
    "CJT-DT0": 71,
    "CRD-PNI": 72,
    "DT0-CJT": 73,
    "NN1-AJ0": 74,
    "NN1-NP0": 75,
    "NN1-VVB": 76,
    "NN1-VVG": 77,
    "NN2-VVZ": 78,
    "NP0-NN1": 79,
    "PNI-CRD": 80,
    "PRP-AVP": 81,
    "PRP-CJS": 82,
    "VVB-NN1": 83,
    "VVD-AJ0": 84,
    "VVD-VVN": 85,
    "VVG-AJ0": 86,
    "VVG-NN1": 87,
    "VVN-AJ0": 88,
    "VVN-VVD": 89,
    "VVZ-NN2": 90,
}

# n observed words, values storing the index in matrix
OBSERVATION = {}


# return emission probability p(word|tag) matrix
def get_emission(emission, word):
    # handle unseen word, return small probabilities
    if word not in OBSERVATION:
        return np.ones((91,)) * 0.00001
    return emission[:, OBSERVATION[word]]


# read and return initial, transition, emission probabilities
def read_probability(training_list):
    # read number of words in total first
    for train_file in training_list:
        train = open(train_file, "r")
        for line in train.readlines():
            word = line.split()[0]
            if word not in OBSERVATION:
                OBSERVATION[word] = len(OBSERVATION)

    k = len(TAGS)
    n = len(OBSERVATION)
    # # initial state probability
    # initial = np.zeros(k)
    # # initial transition matrix
    # transition = np.zeros((k, k))
    # # emission matrix
    # emission = np.zeros((k, n))

    # initial state probability
    initial = np.ones(k) * 0.00001
    # initial transition matrix
    transition = np.ones((k, k)) * 0.00001
    # emission matrix
    emission = np.ones((k, n)) * 0.00001

    # load and form transition and emission table
    for train_file in training_list:
        train = open(train_file, "r")
        # previous tag
        line = train.readline()
        pre_tag = line.split()[2]
        # update initial count
        initial[TAGS[pre_tag]] += 1
        for line in train.readlines():
            word, tags = line.split()[0], line.split()[2]
            transition[TAGS[pre_tag], TAGS[tags]] += 1
            emission[TAGS[tags], OBSERVATION[word]] += 1
            pre_tag = tags
        train.close()
    # convert count into probabilities
    initial = initial / np.sum(initial)
    transition = transition / np.sum(transition, axis=1, keepdims=True)
    emission = emission / np.sum(emission, axis=1, keepdims=True)
    initial = initial.reshape(91)
    return initial, transition, emission


def tag(training_list, test_file, output_file):
    # Tag the words from the untagged input file and write them into the output file.
    # Doesn't do much else beyond that yet.
    print("Tagging the file.")

    # YOUR IMPLEMENTATION GOES HERE

    initial, transition, emission = read_probability(training_list)

    # run model on test file
    test = open(test_file, "r")
    # observation sequence
    sequence = []
    # read the sequence of observations first
    for line in test.readlines():
        word = line.strip()
        sequence.append(word)

    k = len(TAGS)
    n = len(sequence)
    # run Viterbi algorithm
    # prob_trellis = np.zeros((k, n))
    # path_trellis = np.zeros((k, n))
    prob_trellis = np.ones((k, n)) * 0.00001
    path_trellis = np.ones((k, n)) * 0.00001

    # initial state probability
    x1 = initial * get_emission(emission, sequence[0])
    prob_trellis[:, 0] = x1 / np.sum(x1)
    path_trellis[:, 0] = np.array(range(k))

    for o in range(1, n):
        x = prob_trellis[:, o - 1].reshape(91, 1) * transition * get_emission(
            emission, sequence[o])
        max_x = np.max(x, axis=0)
        prob_trellis[:, o] = max_x / np.sum(max_x)
        path_trellis = path_trellis[np.argmax(x, axis=0)]
        path_trellis[:, o] = np.array(range(k))
    test.close()
    # find the most likely path
    index_lst = path_trellis[np.argmax(prob_trellis[:, -1])]
    writing_result(output_file, index_lst, sequence)


def writing_result(output_file, index_lst, sequence):
    out_file = open(output_file, "w")
    print("Writing to {}".format(output_file))
    keys = list(TAGS.keys())
    for i in range(len(sequence)):
        word = sequence[i]
        tags = keys[int(index_lst[i])]
        out_file.write("{} : {}\n".format(word, tags))
    out_file.close()


if __name__ == '__main__':
    # Run the tagger function.
    print("Starting the tagging process.")

    # Tagger expects the input call: "python3 tagger.py -d <training files> -t <test file> -o <output file>"
    parameters = sys.argv
    training_list = parameters[
                    parameters.index("-d") + 1:parameters.index("-t")]
    test_file = parameters[parameters.index("-t") + 1]
    output_file = parameters[parameters.index("-o") + 1]
    # print("Training files: " + str(training_list))
    # print("Test file: " + test_file)
    # print("Ouptut file: " + output_file)

    # Start the training and tagging operation.
    tag(training_list, test_file, output_file)
