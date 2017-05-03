import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Set;
import java.util.TreeMap;
import java.util.stream.IntStream;

import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.SentenceUtils;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.stats.SimpleGoodTuring;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;

public class Main {

	// Global constants
	static String train_dir = new File("").getAbsolutePath().concat("\\train_set\\");
	static String test_dir = new File("").getAbsolutePath().concat("\\test_set\\");
	static String wekaPath = new File("").getAbsolutePath().concat("\\output_values\\");

	static String spanish_abb = "SP";
	static String known_abb = "known";
	static String extention = ".txt";
	static String unknown_abb = "unknown";

	// Character level variables
	static ArrayList<String> characterSet = new ArrayList<>();
	static HashMap<String, Integer> characterIndex = new HashMap<>();

	// POS variables
	static MaxentTagger tagger;
	static ArrayList<String> posTags = new ArrayList<>();
	static HashMap<String, Integer> posTagIndex = new HashMap<>();

	// Train ground truth file
	static String[] groundTruth = new String[100];

	// Helper
	private static Helper helper = new Helper();

	public static void main(String[] args) throws IOException {

		// Loading Spanish character set and initializing character level
		// variables
		loadCharacterSet();

		// Loading Spanish POS tag set and initializing pos tag variables
		loadPOSTagSet();

		// Load train truth file
		loadTruth(test_dir);

		// Loading the "General Models" constructed with all training documents
		// for all authors for later comparison against the individual model for
		// each problem
		// int[][] gCharacterMatrix =
		// helper.buildMatrixFromFile("generalCharacterCountMatrix");
		// int[][] gPosMatrix =
		// helper.buildMatrixFromFile("generalPOSTagCountMatrix");

		// double[][] gctm = laplaceAddOneSmoothing(gCharacterMatrix,
		// characterSet);
		// double[][] gptm = laplaceAddOneSmoothing(gPosMatrix, posTags);

		double[] x_values = new double[100]; // To store the character
												// probability values for each
												// training instance
		double[] y_values = new double[100]; // To store the POS probability
												// values for each training
												// instance

		// For each training problem
		for (int i = 1; i <= 100; i++) {

			// Author's Character Level Markov Chain
			int[][] characterMatrix = initializeCharacterMatrix();

			// Author's POS TAG Markov Chain
			int[][] posMatrix = initializePOSMatrix();

			// For each known text
			for (int j = 1; j <= 4; j++) {
				String knownText = loadKnownText(i, j, test_dir);
				ArrayList<String> sentences = splitIntoSentences(knownText);

				updateCharacterMatrix(characterMatrix, knownText);
				updatePOSMatrix(posMatrix, sentences);

			}

			// Obtaining the transition probability matrix based on count matrix
			// without smoothing
			//double[][] ctm = noSmoothing(characterMatrix, characterSet);
			//double[][] ptm = noSmoothing(posMatrix, posTags);

			// Obtaining the transition probability matrix based on count matrix
			// and (add-one) Laplace smoothing
			//double[][] ctm = laplaceAddOneSmoothing(characterMatrix, characterSet);
			//double[][] ptm = laplaceAddOneSmoothing(posMatrix, posTags);

			// TODO error with some files wher k < 5 .
			// Obtaining the transition probability matrix based on count matrix
			// and Good Turing smoothing
			 double[][] ctm = goodTuringSmoothing(characterMatrix,characterSet);
			 double[][] ptm = goodTuringSmoothing(posMatrix, posTags);

			// Loading the unknown text. First as a char sequence and then as a
			// POS tag sequence
			String unknownText = loadUnknownText(i, test_dir);
			ArrayList<String> sentences = splitIntoSentences(unknownText);
			ArrayList<String> unknownTags = loadUnknownAsTags(sentences);

			// Now, inference
			double p1 = characterInference(unknownText, 1, ctm);
			double p2 = posTagInference(unknownTags, 1, ptm);

			// double p3 = characterInference(unknownText, 1, gctm);
			// double p4 = posTagInference(unknownTags, 1, gptm);

			x_values[i - 1] = p1;
			y_values[i - 1] = p2;

			//System.out.println(Math.round(p1) + " " + Math.round(p2) + " " + groundTruth[i - 1]);
			
		}

		helper.exportToWekaFormat(x_values, y_values, groundTruth, "test_model_GT_smoothing_100.arff");

	}

	private static void loadCharacterSet() throws IOException {

		// Reading the character set directly for a file
		String path = new File("").getAbsolutePath().concat("/resources/spanish_character_set_reduced");
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
		int i = 0;
		while (br.ready()) {// Reading until the EOF
			String character = br.readLine();
			characterSet.add(character);
			characterIndex.put(character, i);
			i++;
		}
		br.close();

	}

	private static void loadPOSTagSet() throws IOException {

		// Initializing the tagger
		tagger = new MaxentTagger("taggers/spanish-distsim.tagger");

		// Reading the POS tag set directly for a file
		String path = new File("").getAbsolutePath().concat("/resources/pos_tag_set_reduced");
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
		int i = 0;
		while (br.ready()) {// Reading until the EOF
			String tag = br.readLine();
			posTags.add(tag);
			posTagIndex.put(tag, i);
			i++;
		}
		br.close();
	}

	private static void loadTruth(String type) throws IOException {
		// Reading the truth information directly for a file
		String path = type + "truth.txt";
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
		int i = 0;
		while (br.ready()) {// Reading until the EOF
			String[] truth = br.readLine().split(" ");
			groundTruth[i] = truth[1];
			i++;
		}
		br.close();

	}

	private static int[][] initializeCharacterMatrix() {
		int[][] countMatrix = new int[characterSet.size()][characterSet.size()];
		return countMatrix;
	}

	private static int[][] initializePOSMatrix() {
		int[][] countMatrix = new int[posTags.size()][posTags.size()];
		return countMatrix;
	}

	private static String loadKnownText(int problemNumber, int knownNumber, String type) throws IOException {

		String fullDir = type + spanish_abb + helper.buildConsecutiveString(problemNumber) + "\\" + known_abb + "0" + ""
				+ knownNumber + extention;
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(new File(fullDir)), "UTF-8"));
		String resp = br.readLine().concat(". "); // Reading the title and
													// adding a dot at the end.
		while (br.ready()) {// Reading until the EOF
			String line = br.readLine();
			resp += line + " ";
		}
		br.close();
		return resp.trim();

	}

	private static ArrayList<String> splitIntoSentences(String paragraph) {
		Reader reader = new StringReader(paragraph);
		DocumentPreprocessor dp = new DocumentPreprocessor(reader);
		ArrayList<String> sentenceList = new ArrayList<String>();

		for (List<HasWord> sentence : dp) {
			// SentenceUtils not Sentence
			String sentenceString = SentenceUtils.listToString(sentence);
			sentenceList.add(sentenceString);
		}
		return sentenceList;
	}

	private static void updateCharacterMatrix(int[][] characterMatrix, String paragraph) {

		for (int i = 0; i < paragraph.length() - 1; i++) {

			String currentChar = "" + paragraph.charAt(i);
			String nextChar = "" + paragraph.charAt(i + 1);
			if (characterSet.contains(currentChar) && characterSet.contains(nextChar)) {
				characterMatrix[characterIndex.get(currentChar)][characterIndex.get(nextChar)] += 1;
			}

		}

	}

	private static void updatePOSMatrix(int[][] posMatrix, ArrayList<String> sentences) {

		for (String sent : sentences) {
			String tagString = tagger.tagString(sent);
			String[] split = tagString.split(" ");
			for (int i = 0; i < split.length - 1; i++) {
				String currentTag = split[i].substring(split[i].lastIndexOf("_") + 1, split[i].length());
				String nextTag = split[i + 1].substring(split[i + 1].lastIndexOf("_") + 1, split[i + 1].length());
				if (posTags.contains(currentTag) && posTags.contains(nextTag)) {
					posMatrix[posTagIndex.get(currentTag)][posTagIndex.get(nextTag)] += 1;
				}
			}
		}

	}

	private static String loadUnknownText(int problemNumber, String type) throws IOException {

		String fullDir = type + spanish_abb + helper.buildConsecutiveString(problemNumber) + "\\" + unknown_abb
				+ extention;
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(new File(fullDir)), "UTF-8"));
		String resp = br.readLine().concat(". "); // Reading the title and
													// adding a dot at the end.
		while (br.ready()) {// Reading until the EOF
			String line = br.readLine();
			resp += line + " ";
		}
		br.close();
		return resp.trim();
	}

	private static ArrayList<String> loadUnknownAsTags(ArrayList<String> sentences) {

		ArrayList<String> resp = new ArrayList<>();

		for (String sent : sentences) {
			String tagString = tagger.tagString(sent);
			String[] split = tagString.split(" ");
			for (int i = 0; i < split.length; i++) {
				String currentTag = split[i].substring(split[i].lastIndexOf("_") + 1, split[i].length());
				resp.add(currentTag);
			}
		}

		return resp;
	}

	private static double[][] noSmoothing(int[][] matrix, ArrayList<String> itemSet) {

		double[][] resp = new double[matrix.length][matrix.length];

		for (int i = 0; i < resp.length; i++) {

			int n = IntStream.of(matrix[i]).sum();

			for (int j = 0; j < resp.length; j++) {

				int m = matrix[i][j];
				// Estimating probabilities using MLE
				resp[i][j] = (double) m / n;
			}
		}
		return resp;
	}

	private static double[][] laplaceAddOneSmoothing(int[][] matrix, ArrayList<String> itemSet) {

		double[][] resp = new double[matrix.length][matrix.length];

		for (int i = 0; i < resp.length; i++) {

			int n = IntStream.of(matrix[i]).sum() + itemSet.size();

			for (int j = 0; j < resp.length; j++) {

				int m = matrix[i][j];
				m++;
				// Estimating probabilities using MLE
				resp[i][j] = (double) m / n;
			}
		}
		return resp;
	}

	private static double[][] goodTuringSmoothing(int[][] matrix, ArrayList<String> characterSet) {

		double[][] resp = new double[matrix.length][matrix.length];

		for (int i = 0; i < matrix.length; i++) {

			int[] distribution = matrix[i];
			TreeMap<Integer, Integer> countsAndCountOfCounts = findCounts(distribution);

			Set<Integer> keySet = countsAndCountOfCounts.keySet();

			if(keySet.size()>=5){ //This constrain is imposed by the implementation of GT discount
				int[] r = new int[keySet.size()];
				int[] n = new int[keySet.size()];

				int index = 0;
				for (int count : keySet) {
					r[index] = count;
					n[index] = countsAndCountOfCounts.get(count);
					index++;
				}

				HashMap<Integer, Integer> countIndex = findIndex(r);

				SimpleGoodTuring sgt = new SimpleGoodTuring(r, n);
				double[] pForSeens = sgt.getProbabilities();
				double pForUnseens = sgt.getProbabilityForUnseen();

				for (int j = 0; j < distribution.length; j++) {
					int value = distribution[j];
					if (value == 0) { // Unseen
						resp[i][j] = pForUnseens;
					} else {
						resp[i][j] = pForSeens[countIndex.get(value)];
					}
				}
			}
			else{ 
				
				/*int n = IntStream.of(distribution).sum();
				for (int j = 0; j < distribution.length; j++) { //We just use MLE
					int m = distribution[j];
					resp[i][j] = (double) m / n;
				}
				*/
				for (int j = 0; j < distribution.length; j++) { //We just ignore this item
					resp[i][j] = 0d;
				}
			}
			

		}

		return resp;
	}

	private static HashMap<Integer, Integer> findIndex(int[] r) {
		HashMap<Integer, Integer> resp = new HashMap<>();
		for (int i = 0; i < r.length; i++) {
			resp.put(r[i], i);
		}
		return resp;
	}

	private static TreeMap<Integer, Integer> findCounts(int[] distribution) {

		TreeMap<Integer, Integer> map = new TreeMap<>();

		for (int i = 0; i < distribution.length; i++) {
			int count = distribution[i];

			if (count != 0) {
				if (map.containsKey(count)) {
					map.put(count, map.get(count) + 1);
				} else {
					map.put(count, 1);
				}
			}
		}

		return map;
	}

	private static double characterInference(String unknownText, double tam, double[][] transitionMatrix) {

		long limit = Math.round((unknownText.length() - 1) * tam); // We explore
																	// for
		// differents (tams) to
		// make inference

		double probability = 0;

		for (int i = 0; i < limit; i++) {

			String currentChar = "" + unknownText.charAt(i);
			String nextChar = "" + unknownText.charAt(i + 1);

			if (characterSet.contains(currentChar) && characterSet.contains(nextChar)) {
				int currentIndex = characterIndex.get(currentChar);
				int nextIndex = characterIndex.get(nextChar);

				double p = transitionMatrix[currentIndex][nextIndex];

				if (p > 0) {
					probability += Math.log(p);
				}

			}

		}

		return probability;
	}

	private static double posTagInference(ArrayList<String> unknownTags, double tam, double[][] transitionMatrix) {

		long limit = Math.round((unknownTags.size() - 1) * tam); // We explore
																	// for
																	// differents
		// (tams) to make inference

		double probability = 0;

		for (int i = 0; i < limit; i++) {

			String currentTag = "" + unknownTags.get(i);
			String nextTag = "" + unknownTags.get(i + 1);

			if (posTags.contains(currentTag) && posTags.contains(nextTag)) {
				int currentIndex = posTagIndex.get(currentTag);
				int nextIndex = posTagIndex.get(nextTag);

				double p = transitionMatrix[currentIndex][nextIndex];

				if (p > 0) {
					probability += Math.log(p);
				}
			}

		}

		return probability;
	}

}
