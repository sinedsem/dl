package com.github.sinedsem.dl;

import org.apache.commons.io.FileUtils;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;

import java.io.*;
import java.util.HashSet;
import java.util.Set;

public class MakeTextFromHtml {

    static Set<String> stopwords = new HashSet<>(500);

    public static void main(String[] args) throws IOException {

        InputStream stopwordsStream = MakeTextFromHtml.class.getClassLoader().getResourceAsStream("stopwords.txt");

        BufferedReader stopwordsReader = new BufferedReader(new InputStreamReader(stopwordsStream));

        String stopword;
        while ((stopword = stopwordsReader.readLine()) != null) {
            stopwords.add(stopword);
        }
        stopwordsReader.close();


        File textDir = new File("python/data/text");
        FileUtils.cleanDirectory(textDir);
        textDir.mkdirs();

        File[] categories = new File("html").listFiles();
        for (File catFile : categories) {

            String category = catFile.getName();
            new File("python/data/text/" + category).mkdir();

            File[] htmlFiles = new File("html/" + category).listFiles();
            int count = htmlFiles.length;
            int i = 0;
            System.out.println("Category " + category);
            for (File htmlFile : htmlFiles) {
                try {
                    makeText(category, htmlFile.getName());
                } catch (Exception e) {
                    e.printStackTrace();
                }
                System.out.println(i + "/" + count);
                i++;
            }
        }
    }

    public static void makeText(String category, String filename) throws IOException {
        String htmlFile = "html/" + category + "/" + filename;
        String textFile = "python/data/text/" + category + "/" + filename;
        BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(htmlFile)));

        StringBuilder html = new StringBuilder();

        String line;
        while ((line = reader.readLine()) != null) {
            html.append(line.replaceAll("><", "> <"));
        }

        Document doc = Jsoup.parse(html.toString());

//        String text = doc.body().text(); // "An example link"
        /*String text = doc.getElementsByTag("meta").stream()
            .filter(e -> "description".equals(e.attr("name")) || "keywords".equals(e.attr("name")))
            .map(element -> element.attr("content")).collect(Collectors.joining(" "));
        text = doc.getElementsByTag("title").text() + " " + text;*/

        String text = doc.getElementsByTag("title").text();

        text = text.toLowerCase().replaceAll("[^a-z'\\-]", " ");

        String[] words = text.split(" ");


        StringBuilder sb = new StringBuilder();

        int i = 0;
        for (String word : words) {
            if (word.isEmpty()) {
                continue;
            }
            if (word.startsWith("-")) {
                continue;
            }
            if (stopwords.contains(word)) {
                continue;
            }
            sb.append(word);
            sb.append(" ");
            if (i++ == 100) {
                break;
            }
        }
        if (i > 2) {
            PrintWriter writer = new PrintWriter(textFile);
            writer.print(sb.toString());
            writer.close();
        }
    }


}
