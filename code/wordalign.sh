#!/bin/bash

# Compile the Java files in the 'code/nlp/mt' directory
javac code/nlp/mt/*.java

# Run the WordAligner class with the full package name and specified memory
# The classpath includes 'code' since the script is now in 'code' instead of 'mt'
java -Xmx2G -cp code code.nlp.mt.WordAligner "$1" "$2" "$3" "$4"
