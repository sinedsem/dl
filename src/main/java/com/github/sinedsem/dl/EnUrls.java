package com.github.sinedsem.dl;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class EnUrls {

    private static final Pattern urlPattern = Pattern.compile(">([^<]+)</a>$");

    public static void main(String[] args) throws IOException {

        File urlsDir = new File("urls");
        if (!urlsDir.exists()) {
            urlsDir.mkdir();
        }
        int i = 0;
        String[] categories = {"Adult", "Arts", "Business", "Computers", "Games", "Health", "Home", "Kids and Teens", "News", "Recreation", "Reference", "Regional", "Science", "Shopping", "Society", "Sports", "World"};
        for (String category : categories) {
            processCategory(category);
//            System.out.println("proceed " + (++i) + " of " + categories.length);
        }

    }

    private static void processCategory(String category) throws IOException {
        Matcher urlMatcher = urlPattern.matcher("");


        Set<String> foundUrls = new HashSet<>(500);

        int i = 0;
        while (true) {
            List<String> lines = Util.loadUrl("http://www.alexa.com/topsites/category;" + i + "/Top/" + category);

            for (String line : lines) {


                if (!line.contains("<div class=\"desc-container\">")) {
                    continue;
                }

                urlMatcher.reset(line);
                if (urlMatcher.find()) {
                    String url = urlMatcher.group(1).toLowerCase();
                    foundUrls.add(url);
                }

            }
            if (i > 21) {
                break;
            }
            i++;
        }

        if (foundUrls.size() < 30) {
            return;
        }

        System.out.println("found " + foundUrls.size() + " urls for category " + category);

        PrintWriter writer = new PrintWriter("urls/" + category.toLowerCase() + ".txt");
        for (String url : foundUrls) {
            String urlToWrite = url;
            if (!urlToWrite.startsWith("https://")) {
                urlToWrite = "http://" + urlToWrite;
            }
            writer.println(urlToWrite);
        }
        writer.close();

    }


}
