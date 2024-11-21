package code.nlp.mt;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * IBM Model 2 Word Aligner with Position-Based Alignment Probabilities, Memory Efficiency, and Alignment Calculation
 * 
 * Usage:
 * java code.nlp.mt.WordAligner <english_sentences> <foreign_sentences> <iterations> <probability_threshold> [<min_frequency>] [--align]
 */
public class WordAligner {

    private static final String NULL_TOKEN = "NULL";
    private static boolean calculateAlignments = false; // Flag for alignment calculation

    public static void main(String[] args) {
        if (args.length < 4) {
            System.err.println("Usage: java code.nlp.mt.WordAligner <english_sentences> <foreign_sentences> <iterations> <probability_threshold> [<min_frequency>] [--align]");
            System.exit(1);
        }

        String englishFile = args[0];
        String foreignFile = args[1];
        int iterations = Integer.parseInt(args[2]);
        double probThreshold = Double.parseDouble(args[3]);
        int minFrequency = args.length > 4 ? Integer.parseInt(args[4]) : 1; // Default to include all words
        if (args.length > 5 && args[5].equals("--align")) {
            calculateAlignments = true;
        }

        // Step 1: Read and preprocess sentences
        List<SentencePair> sentencePairs = readSentencesWithNull(englishFile, foreignFile);

        // Step 2: Create word mappings with frequency filtering
        Map<String, Integer> wordToId = createWordMapping(sentencePairs, minFrequency);
        Map<Integer, String> idToWord = reverseMapping(wordToId); // Reverse mapping for easy lookup

        // Step 3: Convert sentences to integer pairs for efficiency
        List<IntSentencePair> intSentencePairs = convertToIntegerPairs(sentencePairs, wordToId);

        // Step 4: Initialize p(f|e) with integer mappings
        Map<Integer, Map<Integer, Double>> pFe = initializePFWithNull(intSentencePairs);

        // Step 5: Initialize a(j|i, l_e, l_f) alignment probabilities
        AlignmentProbabilities aJifef = initializeA(intSentencePairs);

        // Step 6: Perform EM Algorithm for IBM Model 2
        pFe = performEMModel2(intSentencePairs, pFe, aJifef, iterations);

        // Step 7: Output results based on the flag
        if (!calculateAlignments) {
            printProbabilityTable(pFe, idToWord, probThreshold);
        } else {
            calculateAlignments(intSentencePairs, pFe, aJifef, probThreshold, idToWord);
        }
    }

    
    private static List<SentencePair> readSentencesWithNull(String engFile, String forwFile) {
        List<SentencePair> pairs = new ArrayList<>();
        try (BufferedReader engReader = new BufferedReader(new FileReader(engFile));
             BufferedReader forwReader = new BufferedReader(new FileReader(forwFile))) {

            String engLine;
            String forwLine;
            while ((engLine = engReader.readLine()) != null && (forwLine = forwReader.readLine()) != null) {
                List<String> engWords = new ArrayList<>(Arrays.asList(engLine.trim().split("\\s+")));
                engWords.add(0, NULL_TOKEN); // Add NULL token at the beginning for alignment

                List<String> forwWords = new ArrayList<>(Arrays.asList(forwLine.trim().split("\\s+")));
                pairs.add(new SentencePair(engWords, forwWords));
            }
        } catch (IOException e) {
            System.err.println("Error reading input files: " + e.getMessage());
            System.exit(1);
        }
        return pairs;
    }

  
    private static Map<String, Integer> createWordMapping(List<SentencePair> sentencePairs, int minFrequency) {
        Map<String, Integer> wordToId = new HashMap<>();
        Map<String, Integer> wordFrequency = new HashMap<>();

        for (SentencePair pair : sentencePairs) {
            for (String word : pair.englishWords) {
                wordFrequency.put(word, wordFrequency.getOrDefault(word, 0) + 1);
            }
            for (String word : pair.foreignWords) {
                wordFrequency.put(word, wordFrequency.getOrDefault(word, 0) + 1);
            }
        }

        int id = 0;
        for (String word : wordFrequency.keySet()) {
            if (wordFrequency.get(word) >= minFrequency) {
                wordToId.put(word, id++);
            }
        }
        return wordToId;
    }

    /**
    * Reverse the word-to-integer mapping for easy lookup when printing.
     */
    private static Map<Integer, String> reverseMapping(Map<String, Integer> wordToId) {
        Map<Integer, String> idToWord = new HashMap<>();
        for (Map.Entry<String, Integer> entry : wordToId.entrySet()) {
            idToWord.put(entry.getValue(), entry.getKey());
        }
        return idToWord;
    }

 
    private static List<IntSentencePair> convertToIntegerPairs(List<SentencePair> sentencePairs, Map<String, Integer> wordToId) {
        List<IntSentencePair> intPairs = new ArrayList<>();

        for (SentencePair pair : sentencePairs) {
            List<Integer> engIds = new ArrayList<>();
            List<Integer> forIds = new ArrayList<>();

            for (String e : pair.englishWords) {
                Integer id = wordToId.get(e);
                if (id != null) engIds.add(id);
            }
            for (String f : pair.foreignWords) {
                Integer id = wordToId.get(f);
                if (id != null) forIds.add(id);
            }

            if (!engIds.isEmpty() && !forIds.isEmpty()) {
                intPairs.add(new IntSentencePair(engIds, forIds));
            }
        }
        return intPairs;
    }

    /**
     * Initialize p(f|e) with integer mappings.
     */
    private static Map<Integer, Map<Integer, Double>> initializePFWithNull(List<IntSentencePair> sentencePairs) {
        Map<Integer, Set<Integer>> eToF = new HashMap<>();

        for (IntSentencePair pair : sentencePairs) {
            for (Integer e : pair.englishWords) {
                eToF.computeIfAbsent(e, key -> new HashSet<>()).addAll(pair.foreignWords);
            }
        }

        Map<Integer, Map<Integer, Double>> pFe = new HashMap<>();
        for (Integer e : eToF.keySet()) {
            Set<Integer> fSet = eToF.get(e);
            double uniformProb = 1.0 / fSet.size();

            Map<Integer, Double> fProbMap = new HashMap<>();
            for (Integer f : fSet) {
                fProbMap.put(f, uniformProb);
            }
            pFe.put(e, fProbMap);
        }
        return pFe;
    }

    /**
     *  Initialize a(j|i, l_e, l_f) uniformly.
     */
    private static AlignmentProbabilities initializeA(List<IntSentencePair> sentencePairs) {
        AlignmentProbabilities aJifef = new AlignmentProbabilities();

        for (IntSentencePair pair : sentencePairs) {
            int l_e = pair.englishWords.size();
            int l_f = pair.foreignWords.size();

            for (int j = 0; j < l_f; j++) { // Target word positions
                for (int i = 0; i < l_e; i++) { // Source word positions
                    aJifef.incrementCount(i, j, l_e, l_f, 1.0); // Initialize to uniform
                }
            }
        }

        // Normalize a(j|i, l_e, l_f)
        aJifef.normalize();
        return aJifef;
    }

    /**
     *  Perform EM algorithm for IBM Model 2.
     */
    private static Map<Integer, Map<Integer, Double>> performEMModel2(List<IntSentencePair> sentencePairs,
                                                                       Map<Integer, Map<Integer, Double>> pFe,
                                                                       AlignmentProbabilities aJifef,
                                                                       int iterations) {
        for (int it = 1; it <= iterations; it++) {
            // Initialize count dictionaries
            Map<Integer, Map<Integer, Double>> countEF = new HashMap<>();
            AlignmentProbabilities countA = new AlignmentProbabilities();

            for (IntSentencePair pair : sentencePairs) {
                int l_e = pair.englishWords.size();
                int l_f = pair.foreignWords.size();

                for (int j = 0; j < l_f; j++) { // Target word positions
                    Integer f = pair.foreignWords.get(j);
                    double totalProb = 0.0;

                    // Calculate total probability for normalization
                    for (int i = 0; i < l_e; i++) { // Source word positions
                        Integer e = pair.englishWords.get(i);
                        double p_f_e = pFe.getOrDefault(e, Collections.emptyMap()).getOrDefault(f, 0.0);
                        double a_j_i_l_e_l_f = aJifef.getProbability(i, j, l_e, l_f);
                        totalProb += p_f_e * a_j_i_l_e_l_f;
                    }

                    if (totalProb == 0.0) continue;

                    // Collect expected counts
                    for (int i = 0; i < l_e; i++) {
                        Integer e = pair.englishWords.get(i);
                        double p_f_e = pFe.getOrDefault(e, Collections.emptyMap()).getOrDefault(f, 0.0);
                        double a_j_i_l_e_l_f = aJifef.getProbability(i, j, l_e, l_f);
                        double delta = (p_f_e * a_j_i_l_e_l_f) / totalProb;

                        // Update countEF
                        countEF.computeIfAbsent(e, key -> new HashMap<>()).put(f,
                                countEF.getOrDefault(e, Collections.emptyMap()).getOrDefault(f, 0.0) + delta);

                        // Update countA
                        countA.incrementCount(i, j, l_e, l_f, delta);
                    }
                }
            }

            // Update p(f|e)
            for (Integer e : countEF.keySet()) {
                double total = countEF.get(e).values().stream().mapToDouble(Double::doubleValue).sum();
                for (Integer f : countEF.get(e).keySet()) {
                    double newProb = countEF.get(e).get(f) / total;
                    pFe.get(e).put(f, newProb);
                }
            }

            // Update a(j|i, l_e, l_f)
            aJifef = countA;
            aJifef.normalize();

            System.out.println("Iteration " + it + " completed.");
        }
        return pFe;
    }

    /**
     * Print the p(f|e) probability table.
     */
    private static void printProbabilityTable(Map<Integer, Map<Integer, Double>> pFe, Map<Integer, String> idToWord, double threshold) {
        List<ProbabilityEntry> outputEntries = new ArrayList<>();
        for (Integer e : pFe.keySet()) {
            for (Integer f : pFe.get(e).keySet()) {
                double prob = pFe.get(e).get(f);
                if (prob >= threshold) {
                    outputEntries.add(new ProbabilityEntry(idToWord.get(e), idToWord.get(f), prob));
                }
            }
        }
        Collections.sort(outputEntries);
        for (ProbabilityEntry entry : outputEntries) {
            System.out.printf("%s %s %.15f%n", entry.englishWord, entry.foreignWord, entry.probability);
        }
    }

    /**
     *  Calculate and print alignments for sentence pairs using IBM Model 2 probabilities.
     */
    private static void calculateAlignments(List<IntSentencePair> sentencePairs,
                                           Map<Integer, Map<Integer, Double>> pFe,
                                           AlignmentProbabilities aJifef,
                                           double threshold,
                                           Map<Integer, String> idToWord) {
        for (IntSentencePair pair : sentencePairs) {
            int l_e = pair.englishWords.size();
            int l_f = pair.foreignWords.size();

            System.out.println("Sentence pair alignment:");
            for (int j = 0; j < l_f; j++) { // Target word positions
                Integer f = pair.foreignWords.get(j);
                Integer bestE = -1;
                double bestProb = threshold;

                for (int i = 0; i < l_e; i++) { // Source word positions
                    Integer e = pair.englishWords.get(i);
                    double p_f_e = pFe.getOrDefault(e, Collections.emptyMap()).getOrDefault(f, 0.0);
                    double a_j_i_l_e_l_f = aJifef.getProbability(i, j, l_e, l_f);
                    double prob = p_f_e * a_j_i_l_e_l_f;

                    if (prob > bestProb) {
                        bestProb = prob;
                        bestE = e;
                    }
                }

                if (bestE != -1) {
                    System.out.println(idToWord.get(bestE) + " -> " + idToWord.get(f) + " (p=" + bestProb + ")");
                }
            }
        }
    }

    // Inner classes for sentence pairs and probability entries
    static class SentencePair {
        List<String> englishWords;
        List<String> foreignWords;

        SentencePair(List<String> eng, List<String> forw) {
            this.englishWords = eng;
            this.foreignWords = forw;
        }
    }

    static class IntSentencePair {
        List<Integer> englishWords;
        List<Integer> foreignWords;

        IntSentencePair(List<Integer> eng, List<Integer> forw) {
            this.englishWords = eng;
            this.foreignWords = forw;
        }
    }

    static class ProbabilityEntry implements Comparable<ProbabilityEntry> {
        String englishWord;
        String foreignWord;
        double probability;

        ProbabilityEntry(String e, String f, double p) {
            this.englishWord = e;
            this.foreignWord = f;
            this.probability = p;
        }

        @Override
        public int compareTo(ProbabilityEntry other) {
            int cmp = this.englishWord.compareTo(other.englishWord);
            return cmp != 0 ? cmp : this.foreignWord.compareTo(other.foreignWord);
        }
    }

    /**
     * Class to handle alignment probabilities a(j|i, l_e, l_f).
     */
    static class AlignmentProbabilities {
        // Key: i_j_l_e_l_f combined as a string (i_j_l_e_l_f)
        private Map<String, Double> aJifef;

        public AlignmentProbabilities() {
            aJifef = new HashMap<>();
        }

        /**
         * Increment count for a specific alignment.
         */
        public void incrementCount(int i, int j, int l_e, int l_f, double value) {
            String key = generateKey(i, j, l_e, l_f);
            aJifef.put(key, aJifef.getOrDefault(key, 0.0) + value);
        }

        /**
         * Get probability for a specific alignment.
         */
        public double getProbability(int i, int j, int l_e, int l_f) {
            String key = generateKey(i, j, l_e, l_f);
            return aJifef.getOrDefault(key, 0.0);
        }

        /**
         * Normalize the alignment probabilities.
         */
        public void normalize() {
            // Group keys by (i, l_e, l_f) to normalize a(j|i, l_e, l_f)
            Map<String, Double> sumMap = new HashMap<>();
            for (String key : aJifef.keySet()) {
                String[] parts = key.split("_");
                String groupKey = parts[0] + "_" + parts[2] + "_" + parts[3];
                sumMap.put(groupKey, sumMap.getOrDefault(groupKey, 0.0) + aJifef.get(key));
            }

            // Divide each a(j|i, l_e, l_f) by the sum for its group
            for (String key : aJifef.keySet()) {
                String[] parts = key.split("_");
                String groupKey = parts[0] + "_" + parts[2] + "_" + parts[3];
                double sum = sumMap.get(groupKey);
                if (sum > 0) {
                    aJifef.put(key, aJifef.get(key) / sum);
                }
            }
        }

        /**
         * Generate a unique key based on alignment parameters.
         */
        private String generateKey(int i, int j, int l_e, int l_f) {
            return i + "_" + j + "_" + l_e + "_" + l_f;
        }
    }
}
