import os

# Local path of the analysis project
input_path = raw_input()
os.chdir(input_path)

# General structure of every project
if os.path.exists("./input_files") == False:
    os.mkdir("./input_files")
if os.path.exists("./output_files") == False:
    os.mkdir("./output_files")
if os.path.exists("./modules") == False:
    os.mkdir("./modules")
if os.path.exists("./scripts") == False:
    os.mkdir("./scripts")
if os.path.exists("./logs") == False:
    os.mkdir("./logs")

# General structure of input files
if os.path.exists("./input_files/raw_data") == False:
    os.mkdir("./input_files/raw_data")
if os.path.exists("./input_files/processed_data") == False:
    os.mkdir("./input_files/processed_data")
if os.path.exists("./input_files/data_dictionary") == False:
    os.mkdir("./input_files/data_dictionary")