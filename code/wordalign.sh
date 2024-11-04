#!/bin/bash

# Change to the code directory (where this script is located)
cd "$(dirname "$0")"

# Compile the Java files in the 'nlp/mt' directory, specifying 'code' as the root for packages
javac -d . nlp/mt/*.java

# Run the WordAligner class with the full package name and classpath as '.'
java -Xmx2G -cp . code.nlp.mt.WordAligner "$1" "$2" "$3" "$4"
