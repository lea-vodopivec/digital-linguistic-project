input_path = "ReGrouped_Corpus_of_idioms.txt"
output_path = "ReGrouped_Corpus_of_idioms_numbered.txt"

with open(input_path, "r", encoding="utf-8") as infile, \
     open(output_path, "w", encoding="utf-8") as outfile:
    counter = 1
    for line in infile:
        stripped = line.rstrip("\n")
        if stripped == "":
            outfile.write("\n")
            continue
        outfile.write(f"{counter}. {stripped}\n")
        counter += 1
