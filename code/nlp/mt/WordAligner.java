package code.nlp.mt;


import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * IBM Model 1 Word Aligner with NULL Alignment, Memory Efficiency, and Alignment Calculation
 * 
 * Usage:
 * java WordAligner <english_sentences> <foreign_sentences> <iterations> <probability_threshold> [<min_frequency>] [--align]
 */
public class WordAligner {

    private static final String NULL_TOKEN = "NULL";
    private static boolean calculateAlignments = false; // Flag for alignment calculation

    public static void main(String[] args) {
        if (args.length < 4) {
            System.err.println("Usage: java WordAligner <english_sentences> <foreign_sentences> <iterations> <probability_threshold> [<min_frequency>] [--align]");
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

        List<SentencePair> sentencePairs = readSentencesWithNull(englishFile, foreignFile);
        Map<String, Integer> wordToId = createWordMapping(sentencePairs, minFrequency);
        Map<Integer, String> idToWord = reverseMapping(wordToId); // Reverse mapping for easy lookup

        List<IntSentencePair> intSentencePairs = convertToIntegerPairs(sentencePairs, wordToId);

        // Initialize p(f|e) with integer mappings
        Map<Integer, Map<Integer, Double>> pFe = initializePFWithNull(intSentencePairs);
        pFe = performEM(intSentencePairs, pFe, iterations);

        // Print the probability table if alignment mode is off
        if (!calculateAlignments) {
            printProbabilityTable(pFe, idToWord, probThreshold);
        } else {
            calculateAlignments(intSentencePairs, pFe, probThreshold, idToWord);
        }
    }

    /**
     * Step 1: Create a word-to-integer mapping with frequency-based filtering.
     */
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

    /**
     * Convert sentences to integer pairs using the word mapping.
     */
    private static List<IntSentencePair> convertToIntegerPairs(List<SentencePair> sentencePairs, Map<String, Integer> wordToId) {
        List<IntSentencePair> intPairs = new ArrayList<>();

        for (SentencePair pair : sentencePairs) {
            Set<Integer> engIds = new HashSet<>();
            Set<Integer> forIds = new HashSet<>();

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
     * Read sentences and add NULL token to English words for alignment.
     */
    private static List<SentencePair> readSentencesWithNull(String engFile, String forwFile) {
        List<SentencePair> pairs = new ArrayList<>();
        try (BufferedReader engReader = new BufferedReader(new FileReader(engFile));
             BufferedReader forwReader = new BufferedReader(new FileReader(forwFile))) {

            String engLine;
            String forwLine;
            while ((engLine = engReader.readLine()) != null && (forwLine = forwReader.readLine()) != null) {
                Set<String> engWords = new HashSet<>(Arrays.asList(engLine.trim().split("\\s+")));
                engWords.add(NULL_TOKEN); // Add NULL token for unaligned words

                Set<String> forwWords = new HashSet<>(Arrays.asList(forwLine.trim().split("\\s+")));
                pairs.add(new SentencePair(engWords, forwWords));
            }
        } catch (IOException e) {
            System.err.println("Error reading input files: " + e.getMessage());
            System.exit(1);
        }
        return pairs;
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
     * Perform EM algorithm on integer-based sentences.
     */
    private static Map<Integer, Map<Integer, Double>> performEM(List<IntSentencePair> sentencePairs, Map<Integer, Map<Integer, Double>> pFe, int iterations) {
        for (int it = 1; it <= iterations; it++) {
            Map<Integer, Map<Integer, Double>> countEF = new HashMap<>();
            Map<Integer, Double> countE = new HashMap<>();

            for (IntSentencePair pair : sentencePairs) {
                for (Integer f : pair.foreignWords) {
                    double totalProb = 0.0;

                    for (Integer e : pair.englishWords) {
                        totalProb += pFe.getOrDefault(e, Collections.emptyMap()).getOrDefault(f, 0.0);
                    }

                    if (totalProb == 0.0) continue;

                    for (Integer e : pair.englishWords) {
                        double prob = pFe.getOrDefault(e, Collections.emptyMap()).getOrDefault(f, 0.0) / totalProb;
                        countEF.computeIfAbsent(e, k -> new HashMap<>()).put(f, countEF.getOrDefault(e, new HashMap<>()).getOrDefault(f, 0.0) + prob);
                        countE.put(e, countE.getOrDefault(e, 0.0) + prob);
                    }
                }
            }

            for (Integer e : pFe.keySet()) {
                Map<Integer, Double> fProbMap = pFe.get(e);
                for (Integer f : fProbMap.keySet()) {
                    if (countE.get(e) > 0) {
                        double newProb = countEF.getOrDefault(e, Collections.emptyMap()).getOrDefault(f, 0.0) / countE.get(e);
                        fProbMap.put(f, newProb);
                    }
                }
            }

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
     * Calculate and print alignments for sentence pairs.
     */
    private static void calculateAlignments(List<IntSentencePair> sentencePairs, Map<Integer, Map<Integer, Double>> pFe, double threshold, Map<Integer, String> idToWord) {
        for (IntSentencePair pair : sentencePairs) {
            System.out.println("Sentence pair alignment:");
            for (Integer f : pair.foreignWords) {
                Integer bestE = -1;
                double bestProb = threshold;
                for (Integer e : pair.englishWords) {
                    double prob = pFe.getOrDefault(e, Collections.emptyMap()).getOrDefault(f, 0.0);
                    if (prob > bestProb) {
                        bestProb = prob;
                        bestE = e;
                    }
                }
                System.out.println(idToWord.get(bestE) + " -> " + idToWord.get(f) + " (p=" + bestProb + ")");
            }
        }
    }

    // Inner classes for sentence pairs and probability entries
    static class SentencePair {
        Set<String> englishWords;
        Set<String> foreignWords;

        SentencePair(Set<String> eng, Set<String> forw) {
            this.englishWords = eng;
            this.foreignWords = forw;
        }
    }

    static class IntSentencePair {
        Set<Integer> englishWords;
        Set<Integer> foreignWords;

        IntSentencePair(Set<Integer> eng, Set<Integer> forw) {
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
}
