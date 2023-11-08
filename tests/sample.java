// File used for integration test in ci

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

class task {
    public static void main(String[] args) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));

        String[] line = reader.readLine().split(" ");

        long a = Long.parseLong(line[0]);
        long b = Long.parseLong(line[1]);

        System.out.println(a + b);

        reader.close();
    }
}

