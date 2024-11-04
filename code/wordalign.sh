#!/bin/bash

# Compile the Java files located in 'nlp/mt' relative to the 'code' directory
javac nlp/mt/*.java

# Run the WordAligner class with the current directory as classpath
# The classpath includes '.' (current directory) since the script is in 'code'
java -Xmx2G -cp . code.nlp.mt.WordAligner "$1" "$2" "$3" "$4"
