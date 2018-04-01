                            Stemmer for Czech

Copyright © 2010 Luís Gomes <luismsgomes@gmail.com>.

UTF-8 is assumed everywhere.

To compile the Java program use this command:
    javac -encoding utf8 CzechStemmer.java

Original Java version by Ljiljana Dolamic, University of Neuchatel.
Downloaded from http://members.unine.ch/jacques.savoy/clef/index.html.

Java code was fixed and reformatted by Luís Gomes.  Added a main procedure
 that applies stemming as a UNIX filter (reads stdin, writes stdout).

Ported to Python by Luís Gomes.

How to use the Java program:
    LC_ALL=UTF-8 java CzechStemmer light < example.txt > light.txt
    LC_ALL=UTF-8 java CzechStemmer aggressive < example.txt > aggressive.txt

How to use the Python program:
    LC_ALL=UTF-8 ./czech_stemmer.py light < example.txt > light.txt
    LC_ALL=UTF-8 ./czech_stemmer.py aggressive < example.txt > aggressive.txt

