package nlp.mt;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * IBM Model 1 Word Aligner with NULL Alignment
 * 
 * This program implements the IBM Model 1 algorithm to learn word alignments between
 * English and Foreign (e.g., Spanish) sentences. It includes the handling of NULL alignment,
 * allowing foreign words to align to a special NULL token when they do not have a direct
 * English counterpart.
 * 
 * Usage:
 * java WordAligner <english_sentences> <foreign_sentences> <iterations> <probability_threshold>
 * 
 * Example:
 * java WordAligner ../data/es-en.100k.en ../data/es-en.100k.es 10 0.3
 */
public class WordAligner {

    public static void main(String[] args) {
        // Ensure exactly four command-line arguments are provided
        if (args.length != 4) {
            System.err.println("Usage: java WordAligner <english_sentences> <foreign_sentences> <iterations> <probability_threshold>");
            System.exit(1); // Exit if incorrect usage
        }

        // Parse command-line arguments
        String englishFile = args[0];           // Path to English sentences file
        String foreignFile = args[1];           // Path to Foreign sentences file
        int iterations = Integer.parseInt(args[2]);       // Number of EM iterations
        double probThreshold = Double.parseDouble(args[3]); // Probability threshold

        // Read and preprocess sentence pairs, including NULL alignment
        List<SentencePair> sentencePairs = readSentencesWithNull(englishFile, foreignFile);

        // Initialize p(f|e) uniformly for co-occurring pairs, including NULL
        Map<String, Map<String, Double>> pFe = initializePFWithNull(sentencePairs);

        // Perform EM training for the specified number of iterations
        pFe = performEM(sentencePairs, pFe, iterations);

        // Print the learned probability table with threshold and sorting
        printProbabilityTable(pFe, probThreshold);
    }

    /**
     * Represents a pair of English and Foreign sentences as sets of words.
     * This ensures that repetitions are ignored, aligning with the assignment's requirements.
     */
    static class SentencePair {
        Set<String> englishWords; // Set of unique English words
        Set<String> foreignWords; // Set of unique Foreign words

        SentencePair(Set<String> eng, Set<String> forw) {
            this.englishWords = eng;
            this.foreignWords = forw;
        }
    }

    /**
     * Reads parallel English and Foreign sentences from the provided files.
     * Each English sentence is treated as a set of words with a special "NULL" token prepended.
     * This allows foreign words to align to "NULL" if they do not have a direct English counterpart.
     *
     * @param engFile Path to English sentences file
     * @param forwFile Path to Foreign sentences file
     * @return List of SentencePair objects containing sets of words
     */
    private static List<SentencePair> readSentencesWithNull(String engFile, String forwFile) {
        List<SentencePair> pairs = new ArrayList<>(); // Initialize list to hold sentence pairs
        try (
            BufferedReader engReader = new BufferedReader(new FileReader(engFile)); // Reader for English sentences
            BufferedReader forwReader = new BufferedReader(new FileReader(forwFile))  // Reader for Foreign sentences
        ) {
            String engLine;   // Variable to hold each English sentence
            String forwLine;  // Variable to hold each Foreign sentence
            // Read both files line by line simultaneously
            while ((engLine = engReader.readLine()) != null && (forwLine = forwReader.readLine()) != null) {
                // Split English sentence into words based on whitespace
                Set<String> engWords = new HashSet<>(Arrays.asList(engLine.trim().split("\\s+")));
                engWords.add("NULL"); // Add "NULL" token to handle unaligned foreign words

                // Split Foreign sentence into words based on whitespace
                Set<String> forwWords = new HashSet<>(Arrays.asList(forwLine.trim().split("\\s+")));

                // Create a new SentencePair object and add it to the list
                pairs.add(new SentencePair(engWords, forwWords));
            }
        } catch (IOException e) { // Handle any I/O exceptions
            System.err.println("Error reading input files: " + e.getMessage());
            System.exit(1); // Exit if there's an error reading files
        }
        return pairs; // Return the list of sentence pairs
    }

    /**
     * Initializes the translation probabilities p(f|e) uniformly for all co-occurring word pairs,
     * including the "NULL" token.
     *
     * @param sentencePairs List of SentencePair objects
     * @return Nested Map representing p(f|e)
     */
    private static Map<String, Map<String, Double>> initializePFWithNull(List<SentencePair> sentencePairs) {
        // Map to hold English vocabulary and their possible Foreign word alignments
        Map<String, Set<String>> eToF = new HashMap<>();

        // Extract vocabulary and co-occurring word pairs from all sentence pairs
        for (SentencePair pair : sentencePairs) {
            for (String e : pair.englishWords) { // Iterate over English words in the sentence
                eToF.computeIfAbsent(e, _ -> new HashSet<>()).addAll(pair.foreignWords);



 // Map e to all foreign words in the sentence
            }
        }

        // Initialize p(f|e) uniformly
        Map<String, Map<String, Double>> pFe = new HashMap<>(); // Nested map to hold p(f|e) probabilities
        for (String e : eToF.keySet()) { // Iterate over each English word
            Set<String> fSet = eToF.get(e); // Get the set of Foreign words co-occurring with e
            int fCount = fSet.size(); // Number of Foreign words co-occurring with e
            if (fCount == 0) continue; // Skip if there are no Foreign words for e

            double uniformProb = 1.0 / fCount; // Calculate uniform probability for each f given e

            Map<String, Double> fProbMap = new HashMap<>(); // Map to hold p(f|e) for the current e
            for (String f : fSet) { // Iterate over each Foreign word co-occurring with e
                fProbMap.put(f, uniformProb); // Assign uniform probability to each f
            }
            pFe.put(e, fProbMap); // Map e to its corresponding f probabilities
        }

        return pFe; // Return the initialized p(f|e) probabilities
    }

    /**
     * Performs the EM algorithm to train the IBM Model 1 word aligner.
     *
     * @param sentencePairs List of SentencePair objects
     * @param pFe Initial p(f|e) probabilities
     * @param iterations Number of EM iterations to perform
     * @return Updated p(f|e) probabilities after training
     */
    private static Map<String, Map<String, Double>> performEM(List<SentencePair> sentencePairs,
                                                             Map<String, Map<String, Double>> pFe,
                                                             int iterations) {
        for (int it = 1; it <= iterations; it++) { // Loop over the number of iterations
            // Initialize count(e,f) and count(e) as nested maps with default value 0.0
            Map<String, Map<String, Double>> countEF = new HashMap<>(); // Nested map to hold counts for each (e,f) pair
            Map<String, Double> countE = new HashMap<>(); // Map to hold total counts for each English word e

            // E-step: Calculate expected counts based on current p(f|e)
            for (SentencePair pair : sentencePairs) { // Iterate over each sentence pair
                Set<String> engWords = pair.englishWords;   // Set of English words in the sentence
                Set<String> forwWords = pair.foreignWords; // Set of Foreign words in the sentence

                for (String f : forwWords) { // Iterate over each Foreign word in the sentence
                    double totalProb = 0.0; // Variable to hold the sum of p(f|e) over all e in the sentence

                    // Calculate the normalization factor for f
                    for (String e : engWords) { // Iterate over each English word in the sentence
                        double prob = pFe.containsKey(e) ? pFe.get(e).getOrDefault(f, 0.0) : 0.0; // Get p(f|e)
                        totalProb += prob; // Accumulate the total probability
                    }

                    if (totalProb == 0.0) continue; // Skip if the total probability is zero to avoid division by zero

                    // Distribute the probability mass to each e in the English sentence
                    for (String e : engWords) { // Iterate over each English word in the sentence
                        if (!pFe.containsKey(e)) continue; // Skip if e has no associated Foreign words
                        double prob = pFe.get(e).getOrDefault(f, 0.0) / totalProb; // Calculate the fractional probability

                        // Update count(e,f)
                        countEF.computeIfAbsent(e, _ -> new HashMap<>())

                        .put(f, countEF.get(e).getOrDefault(f, 0.0) + prob);
                 
                 
                 
                        // Update count(e)
                        countE.put(e, countE.getOrDefault(e, 0.0) + prob); // Increment the total count for e by prob
                    }
                }
            }

            // M-step: Update p(f|e) based on the accumulated counts
            for (String e : pFe.keySet()) { // Iterate over each English word e
                Map<String, Double> fProbMap = pFe.get(e); // Get the map of Foreign words and their probabilities for e
                for (String f : fProbMap.keySet()) { // Iterate over each Foreign word f associated with e
                    // Update p(f|e) = count(e,f) / count(e)
                    if (countE.containsKey(e) && countE.get(e) > 0.0) { // Ensure count(e) is greater than zero
                        double newProb = countEF.getOrDefault(e, Collections.emptyMap()).getOrDefault(f, 0.0) / countE.get(e); // Calculate the new probability
                        pFe.get(e).put(f, newProb); // Assign the updated probability to p(f|e)
                    }
                }
            }

            // Optional: Print iteration completion status for monitoring
            System.out.println("Iteration " + it + " completed.");
        }

        return pFe; // Return the updated p(f|e) probabilities after all iterations
    }

    /**
     * Prints the p(f|e) probability table to standard output.
     * Only includes entries with probability >= threshold and sorts them alphabetically.
     *
     * @param pFe Trained p(f|e) probabilities
     * @param threshold Probability threshold for filtering
     */
    private static void printProbabilityTable(Map<String, Map<String, Double>> pFe, double threshold) {
        // Create a list to hold the output entries
        List<ProbabilityEntry> outputEntries = new ArrayList<>();

        // Iterate through each English word and its Foreign word probabilities
        for (String e : pFe.keySet()) {
            for (String f : pFe.get(e).keySet()) {
                double prob = pFe.get(e).get(f); // Get p(f|e)
                if (prob >= threshold) { // Check if p(f|e) meets the threshold
                    outputEntries.add(new ProbabilityEntry(e, f, prob)); // Add to output list if it does
                }
            }
        }

        // Sort the entries alphabetically by English word, then Foreign word
        Collections.sort(outputEntries);

        // Print the sorted and filtered probability table
        for (ProbabilityEntry entry : outputEntries) {
            System.out.println(entry.englishWord + " " + entry.foreignWord + " " + entry.probability);
        }
    }

    /**
     * Represents a probability entry for sorting and output.
     * Implements Comparable to allow sorting based on English and Foreign words.
     */
    static class ProbabilityEntry implements Comparable<ProbabilityEntry> {
        String englishWord; // English word e
        String foreignWord; // Foreign word f
        double probability; // p(f|e)

        ProbabilityEntry(String e, String f, double p) {
            this.englishWord = e;
            this.foreignWord = f;
            this.probability = p;
        }

        // Define sorting order: first by English word, then by Foreign word
        @Override
        public int compareTo(ProbabilityEntry other) {
            int cmp = this.englishWord.compareTo(other.englishWord); // Compare English words
            if (cmp != 0) return cmp; // If not equal, return the comparison result
            return this.foreignWord.compareTo(other.foreignWord); // Else, compare Foreign words
        }
    }
}
