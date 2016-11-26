package com.github.sinedsem.dl;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;

import java.io.File;
import java.io.IOException;

public class Test {
    public static void main(String[] args) throws IOException {
        ParagraphVectors paragraphVectors = WordVectorSerializer.readParagraphVectors(new File("dov2vec4.model"));
        System.out.println(paragraphVectors.getLookupTable());
    }
}
