import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;

public class Helper {

	
	public static int[][] buildMatrixFromFile(String fileName) throws IOException {

		ArrayList<ArrayList<Integer>> temp = new ArrayList<>();

		String path = new File("").getAbsolutePath().concat("/output_values/" + fileName);
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(path)));

		while (br.ready()) {
			String[] line = br.readLine().split(" ");
			ArrayList<Integer> values = new ArrayList<>();
			for (int i = 0; i < line.length; i++) {
				int v = Integer.parseInt(line[i]);
				values.add(v);
			}
			temp.add(values);
		}
		br.close();

		int[][] resp = new int[temp.size()][temp.size()];
		for (int i = 0; i < temp.size(); i++) {
			ArrayList<Integer> arrayList = temp.get(i);
			for (int j = 0; j < arrayList.size(); j++) {
				int value = arrayList.get(j);
				resp[i][j] = value;
			}
		}
		return resp;
	}

	public static void exportMatrixToFile(int[][] matrix, String name) throws IOException {
		String path = new File("").getAbsolutePath().concat("/output_values/").concat(name);
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(new File(path))));

		for (int i = 0; i < matrix.length; i++) {
			String line = "";
			for (int j = 0; j < matrix.length; j++) {
				line += buildConsecutiveString(matrix[i][j]) + " ";
			}

			bw.write(line + "\n");
		}
		bw.close();
	}
	
	public static String buildConsecutiveString(int problemNumber) {
		String resp = "" + problemNumber;
		while (resp.length() < 3) {
			resp = "0" + resp;
		}
		return resp;
	}

	public void exportToWekaFormat(double[] x_values, double[] y_values, String[] trainTruth, String fileName) throws IOException {
		String path = new File("").getAbsolutePath().concat("/output_values/").concat(fileName);
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(new File(path))));
		
		bw.write("@RELATION "+fileName+"\n");
		bw.write("\n");
		
		bw.write("@ATTRIBUTE x_value NUMERIC\n");
		bw.write("@ATTRIBUTE y_value NUMERIC\n");
		bw.write("@ATTRIBUTE clase { N, Y }\n");
		
		bw.write("\n");
		bw.write("@DATA\n");
		
		for (int i = 0; i < trainTruth.length; i++) {
			bw.write(x_values[i] + " , " +y_values[i] + ", " + trainTruth[i]+"\n");
		}
		bw.close();
	}
}
