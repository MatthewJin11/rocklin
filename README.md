# Rocklin Lab 
## Local Structure Prediction with Deep Learning
Given a sequence of amino acids, local_structure_prediction.py predicts its ABEGO sequence with a deep learning model in TensorFlow and Keras. 
### Implementation
A list of PDB IDs is extracted from a CSV database. From these IDs, the corresponding PDB files are downloaded from the RCSB Protein Data Bank. Functions in Biopython are then used to parse PDB files to obtain primary sequence and ABEGO sequence. 

## Features Extraction 
Pipeline to extract features from PDB files and store these features into a CSV file. 

