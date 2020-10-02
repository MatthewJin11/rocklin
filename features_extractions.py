import os


# function to retrieve file extension
def get_file_ext(filename):
    return filename[filename.index(".") + 1:]


# current working directory
files = os.listdir(os.getcwd())

# list of all the features
all_features = []

# temporary storage variables
current_dictionary = {}
current_features = []

# output csv file
output_file = open("result.csv", "w")
output_file.write("pdb_name, ")

for file in files:
    if "." in file:
        if get_file_ext(file) == "pdb":

            # open for read and write
            input_text = open(file, "r+")
            
            is_features = False

            # put the input into an array
            input_array = input_text.readlines()

            for line in input_array:

                # check for #END marker
                if line[0:4] == "#END":
                    is_features = True
                    continue

                if is_features:

                    if " " in line:
                        if line[:line.index(" ")] not in all_features:
                            all_features.append(line[:line.index(" ")])

# alphabetical sorting
all_features.sort()

for feature in all_features:
    if feature == "total_score":
        output_file.write(feature)
    else:
        output_file.write(feature + ", ")

output_file.write("\n")

for file in files:
    if "." in file:
        if get_file_ext(file) == "pdb":

            # open for read and write
            input_text = open(file, "r+")

            is_features = False
            current_features = []

            # put the input into an array
            input_array = input_text.readlines()

            for line in input_array:
                # check for #END marker
                if line[0:4] == "#END":
                    is_features = True
                    continue

                # put feature values into dictionary
                if is_features:

                    if line == "\n":
                        continue

                    current_dictionary[line[:line.index(" ")]] = line[line.index(" ") + 1:len(line) - 1]
                    current_features.append(line[:line.index(" ")])

            output_file.write(file[:file.index(".")] + ", ")

            for feature in all_features:
                if feature not in current_features:
                    output_file.write("0, ")
                elif feature == "total_score":
                    output_file.write(current_dictionary[feature])
                else:
                    output_file.write(current_dictionary[feature] + ", ")

            output_file.write("\n")
