package com.github.sinedsem.dl;

import java.io.*;
import java.util.Date;
import java.util.List;

public class DownloadSites {

    public static void main(String[] args) throws IOException {

        File[] urlFiles = new File("urls").listFiles();
        for (File file : urlFiles) {

            String category = file.getName().substring(0, file.getName().length() - 4);

//            if ("advertising".equals(category)) {
//                continue;
//            }

            File dir = new File("html/" + category);
            if (!dir.exists()) {
                dir.mkdirs();
            }
            int already = dir.listFiles().length;

            System.out.println("==========================");
            System.out.println("Category: " + category);
            System.out.println("==========================");

            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file)));

            int i = 0;
            String url;
            while ((url = reader.readLine()) != null) {
                System.out.println((new Date()) + ": " + i);
                if (i >= 2001) {
                    break;
                }
                if (i < already) {
                    continue;
                }
                String fileName = dir.getAbsoluteFile() + File.separator + url.substring(7).replaceAll("/", " ").replaceAll(" ", "") + ".txt";
                if (new File(fileName).exists()) {
                    continue;
                }
                try {
                    List<String> htmlContent = Util.loadUrl(url);
                    if (htmlContent.size() > 1) {
                        PrintWriter writer = new PrintWriter(fileName);
                        for (String s : htmlContent) {
                            writer.println(s);
                        }
                        writer.close();
                    }
                } catch (Throwable e) {
                    e.printStackTrace();
                    continue;
                }
                i++;
            }
        }
    }


}
