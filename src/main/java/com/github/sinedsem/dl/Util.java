package com.github.sinedsem.dl;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLConnection;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Util {
    public static List<String> loadUrl(String url) {
        URL urlObj;
        try {
            urlObj = new URL(url);
        } catch (MalformedURLException e) {
            return Collections.emptyList();
        }
        BufferedReader reader;
        String line;
        List<String> result = new ArrayList<>(30);

        try {
            URLConnection con = urlObj.openConnection();
            con.setConnectTimeout(120000);
            con.setReadTimeout(120000);
            try (InputStream is = con.getInputStream()) {
                reader = new BufferedReader(new InputStreamReader(is));
                while ((line = reader.readLine()) != null) {
                    result.add(line);
                }
            }
        } catch (IOException ignored) {
        }
        return result;
    }
}
