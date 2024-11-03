#! /bin/bash

# This is just an example for Java.  You'll need to modify it for your package structure
# and names.  See the last assignment for more examples.
javac nlp/mt/*.java
java -Xmx2G -cp . nlp.mt.WordAligner $1 $2 $3 $4
