/**
 *  Czech stemmer.
 *
 * Compile with this command:
 *      javac -encoding utf8 CzechStemmer.java
 *
 * Original code by Ljiljana Dolamic, University of Neuchatel.
 * Downloaded from http://members.unine.ch/jacques.savoy/clef/index.html.
 * Fixed and reformatted by Luís Gomes <luismsgomes@gmail.com>.
 *
 * Removes case endings from nouns and adjectives;
 * Removes possesive adj. endings from names;
 * Removes diminutive, augmentative, comparative sufixes and derivational
 *  sufixes from nouns (only in aggressive stemming mode);
 * Takes care of palatalisation.
 */
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.IOException;
import java.util.regex.Pattern;

public class CzechStemmer {
    private StringBuffer sb = new StringBuffer();
    private Boolean aggressive;
    private Pattern wordpat = Pattern.compile("\\p{L}+");

    public CzechStemmer(Boolean aggressive) {
        this.aggressive = aggressive;
    }

    public String stem(String input) {
        if (!wordpat.matcher(input).matches()) {
            return input;
        }
        sb.delete(0, sb.length()); //reset string buffer
        sb.insert(0, input.toLowerCase());
        removeCase(sb);
        removePossessives(sb);
        if (aggressive) {
            removeComparative(sb);
            removeDiminutive(sb);
            removeAugmentative(sb);
            removeDerivational(sb);
        }
        if (input.equals(input.toUpperCase()))
            return sb.toString().toUpperCase();
        if (input.substring(0, 1).equals(input.substring(0, 1).toUpperCase()))
            return sb.toString().substring(0, 1).toUpperCase()
                 + sb.toString().substring(1);
        return sb.toString();
    }
    private void removeDerivational(StringBuffer buffer) {
        int len = buffer.length();
        if ((len > 8) && buffer.substring(len-6 ,len).equals("obinec")) {
            buffer.delete(len-6 , len);
            return;
        }
        if (len > 7) {
            if (buffer.substring(len-5 ,len).equals("ionář")) {
                buffer.delete(len-4 , len);
                palatalise(buffer);
                return;
            }
            if (buffer.substring(len-5 ,len).equals("ovisk") ||
                    buffer.substring(len-5 ,len).equals("ovstv") ||
                    buffer.substring(len-5 ,len).equals("ovišt") ||
                    buffer.substring(len-5 ,len).equals("ovník")) {
                buffer.delete(len-5 , len);
                return;
            }
        }
        if (len > 6) {
            if (buffer.substring(len-4 ,len).equals("ásek") ||
                    buffer.substring(len-4 ,len).equals("loun") ||
                    buffer.substring(len-4 ,len).equals("nost") ||
                    buffer.substring(len-4 ,len).equals("teln") ||
                    buffer.substring(len-4 ,len).equals("ovec") ||
                    buffer.substring(len-5 ,len).equals("ovík") ||
                    buffer.substring(len-4 ,len).equals("ovtv") ||
                    buffer.substring(len-4 ,len).equals("ovin") ||
                    buffer.substring(len-4 ,len).equals("štin")) {
                buffer.delete(len-4 , len);
                return;
            }
            if (buffer.substring(len-4 ,len).equals("enic") ||
                    buffer.substring(len-4 ,len).equals("inec") ||
                    buffer.substring(len-4 ,len).equals("itel")) {
                buffer.delete(len-3 , len);
                palatalise(buffer);
                return;
            }
        }
        if (len > 5) {
            if (buffer.substring(len-3 ,len).equals("árn")) {
                buffer.delete(len-3 , len);
                return;
            }
            if (buffer.substring(len-3 ,len).equals("ěnk")) {
                buffer.delete(len-2 , len);
                palatalise(buffer);
                return;
            }
            if (buffer.substring(len-3 ,len).equals("ián") ||
                    buffer.substring(len-3 ,len).equals("ist") ||
                    buffer.substring(len-3 ,len).equals("isk") ||
                    buffer.substring(len-3 ,len).equals("išt") ||
                    buffer.substring(len-3 ,len).equals("itb") ||
                    buffer.substring(len-3 ,len).equals("írn")) {
                buffer.delete(len-2 , len);
                palatalise(buffer);
                return;
            }
            if (buffer.substring(len-3 ,len).equals("och") ||
                    buffer.substring(len-3 ,len).equals("ost") ||
                    buffer.substring(len-3 ,len).equals("ovn") ||
                    buffer.substring(len-3 ,len).equals("oun") ||
                    buffer.substring(len-3 ,len).equals("out") ||
                    buffer.substring(len-3 ,len).equals("ouš")) {
                buffer.delete(len-3 , len);
                return;
            }
            if (buffer.substring(len-3 ,len).equals("ušk")) {
                buffer.delete(len-3 , len);
                return;
            }
            if (buffer.substring(len-3 ,len).equals("kyn") ||
                    buffer.substring(len-3 ,len).equals("čan") ||
                    buffer.substring(len-3 ,len).equals("kář") ||
                    buffer.substring(len-3 ,len).equals("néř") ||
                    buffer.substring(len-3 ,len).equals("ník") ||
                    buffer.substring(len-3 ,len).equals("ctv") ||
                    buffer.substring(len-3 ,len).equals("stv")) {
                buffer.delete(len-3 , len);
                return;
            }
        }
        if (len > 4) {
            if (buffer.substring(len-2 ,len).equals("áč") ||
                    buffer.substring(len-2 ,len).equals("ač") ||
                    buffer.substring(len-2 ,len).equals("án") ||
                    buffer.substring(len-2 ,len).equals("an") ||
                    buffer.substring(len-2 ,len).equals("ář") ||
                    buffer.substring(len-2 ,len).equals("as")) {
                buffer.delete(len-2 , len);
                return;
            }
            if (buffer.substring(len-2 ,len).equals("ec") ||
                    buffer.substring(len-2 ,len).equals("en") ||
                    buffer.substring(len-2 ,len).equals("ěn") ||
                    buffer.substring(len-2 ,len).equals("éř")) {
                buffer.delete(len-1 , len);
                palatalise(buffer);
                return;
            }
            if (buffer.substring(len-2 ,len).equals("íř") ||
                    buffer.substring(len-2 ,len).equals("ic") ||
                    buffer.substring(len-2 ,len).equals("in") ||
                    buffer.substring(len-2 ,len).equals("ín") ||
                    buffer.substring(len-2 ,len).equals("it") ||
                    buffer.substring(len-2 ,len).equals("iv")) {
                buffer.delete(len-1 , len);
                palatalise(buffer);
                return;
            }
            if (buffer.substring(len-2 ,len).equals("ob") ||
                    buffer.substring(len-2 ,len).equals("ot") ||
                    buffer.substring(len-2 ,len).equals("ov") ||
                    buffer.substring(len-2 ,len).equals("oň")) {
                buffer.delete(len-2 , len);
                return;
            }
            if (buffer.substring(len-2 ,len).equals("ul")) {
                buffer.delete(len-2 , len);
                return;
            }
            if (buffer.substring(len-2 ,len).equals("yn")) {
                buffer.delete(len-2 , len);
                return;
            }
            if (buffer.substring(len-2 ,len).equals("čk") ||
                    buffer.substring(len-2 ,len).equals("čn") ||
                    buffer.substring(len-2 ,len).equals("dl") ||
                    buffer.substring(len-2 ,len).equals("nk") ||
                    buffer.substring(len-2 ,len).equals("tv") ||
                    buffer.substring(len-2 ,len).equals("tk") ||
                    buffer.substring(len-2 ,len).equals("vk")) {
                buffer.delete(len-2 , len);
                return;
            }
        }
        if (len > 3) {
            if (buffer.charAt(buffer.length()-1) ==  'c' ||
                    buffer.charAt(buffer.length()-1) ==  'č' ||
                    buffer.charAt(buffer.length()-1) ==  'k' ||
                    buffer.charAt(buffer.length()-1) ==  'l' ||
                    buffer.charAt(buffer.length()-1) ==  'n' ||
                    buffer.charAt(buffer.length()-1) ==  't') {
                buffer.delete(len-1 , len);
            }
        }

    }

    private void removeAugmentative(StringBuffer buffer) {
        int len = buffer.length();
        if (len > 6 &&  buffer.substring(len-4 ,len).equals("ajzn")) {
            buffer.delete(len-4 , len);
            return;
        }
        if (len > 5 && (buffer.substring(len-3 ,len).equals("izn")
                || buffer.substring(len-3 ,len).equals("isk"))) {
            buffer.delete(len-2 , len);
            palatalise(buffer);
            return;
        }
        if (len > 4 && buffer.substring(len-2 ,len).equals("ák")) {
            buffer.delete(len-2 , len);
            return;
        }
    }

    private void removeDiminutive(StringBuffer buffer) {
        int len = buffer.length();
        if ((len > 7) && buffer.substring(len-5 ,len).equals("oušek")) {
            buffer.delete(len-5 , len);
            return;
        }
        if (len > 6) {
            if (buffer.substring(len-4,len).equals("eček") ||
                    buffer.substring(len-4,len).equals("éček") ||
                    buffer.substring(len-4,len).equals("iček") ||
                    buffer.substring(len-4,len).equals("íček") ||
                    buffer.substring(len-4,len).equals("enek") ||
                    buffer.substring(len-4,len).equals("ének") ||
                    buffer.substring(len-4,len).equals("inek") ||
                    buffer.substring(len-4,len).equals("ínek")) {
                buffer.delete(len-3 , len);
                palatalise(buffer);
                return;
            }
            if (buffer.substring(len-4,len).equals("áček") ||
                    buffer.substring(len-4,len).equals("aček") ||
                    buffer.substring(len-4,len).equals("oček") ||
                    buffer.substring(len-4,len).equals("uček") ||
                    buffer.substring(len-4,len).equals("anek") ||
                    buffer.substring(len-4,len).equals("onek") ||
                    buffer.substring(len-4,len).equals("unek") ||
                    buffer.substring(len-4,len).equals("ánek")) {
                buffer.delete(len-4 , len);
                return;
            }
        }
        if (len > 5) {
            if (buffer.substring(len-3,len).equals("ečk") ||
                    buffer.substring(len-3,len).equals("éčk") ||
                    buffer.substring(len-3,len).equals("ičk") ||
                    buffer.substring(len-3,len).equals("íčk") ||
                    buffer.substring(len-3,len).equals("enk") ||
                    buffer.substring(len-3,len).equals("énk") ||
                    buffer.substring(len-3,len).equals("ink") ||
                    buffer.substring(len-3,len).equals("ínk")) {
                buffer.delete(len-3 , len);
                palatalise(buffer);
                return;
            }
            if (buffer.substring(len-3,len).equals("áčk") ||
                    buffer.substring(len-3,len).equals("ačk") ||
                    buffer.substring(len-3,len).equals("očk") ||
                    buffer.substring(len-3,len).equals("učk") ||
                    buffer.substring(len-3,len).equals("ank") ||
                    buffer.substring(len-3,len).equals("onk") ||
                    buffer.substring(len-3,len).equals("unk")) {
                buffer.delete(len-3 , len);
                return;

            }
            if (buffer.substring(len-3,len).equals("átk") ||
                    buffer.substring(len-3,len).equals("ánk") ||
                    buffer.substring(len-3,len).equals("ušk")) {
                buffer.delete(len-3 , len);
                return;
            }
        }
        if (len > 4) {
            if (buffer.substring(len-2,len).equals("ek") ||
                    buffer.substring(len-2,len).equals("ék") ||
                    buffer.substring(len-2,len).equals("ík") ||
                    buffer.substring(len-2,len).equals("ik")) {
                buffer.delete(len-1 , len);
                palatalise(buffer);
                return;
            }
            if (buffer.substring(len-2,len).equals("ák") ||
                    buffer.substring(len-2,len).equals("ak") ||
                    buffer.substring(len-2,len).equals("ok") ||
                    buffer.substring(len-2,len).equals("uk")) {
                buffer.delete(len-1 , len);
                return;
            }
        }
        if (len > 3 && buffer.substring(len-1 ,len).equals("k")) {
            buffer.delete(len-1, len);
            return;
        }
    }

    private void removeComparative(StringBuffer buffer) {
        int len = buffer.length();
        if (len > 5
                && (buffer.substring(len-3,len).equals("ejš")
                    ||buffer.substring(len-3,len).equals("ějš"))) {
            buffer.delete(len-2 , len);
            palatalise(buffer);
            return;
        }

    }

    private void palatalise(StringBuffer buffer) {
        int len = buffer.length();
        if (buffer.substring(len-2 ,len).equals("ci") ||
                buffer.substring(len-2 ,len).equals("ce") ||
                buffer.substring(len-2 ,len).equals("či") ||
                buffer.substring(len-2 ,len).equals("če")) {
            buffer.replace(len-2 ,len, "k");
            return;
        }
        if (buffer.substring(len-2 ,len).equals("zi") ||
                buffer.substring(len-2 ,len).equals("ze") ||
                buffer.substring(len-2 ,len).equals("ži") ||
                buffer.substring(len-2 ,len).equals("že")) {
            buffer.replace(len-2 ,len, "h");
            return;
        }
        if (buffer.substring(len-3 ,len).equals("čtě") ||
                buffer.substring(len-3 ,len).equals("čti") ||
                buffer.substring(len-3 ,len).equals("čtí")) {
            buffer.replace(len-3 ,len, "ck");
            return;
        }
        if (buffer.substring(len-2 ,len).equals("ště") ||
                buffer.substring(len-2 ,len).equals("šti") ||
                buffer.substring(len-2 ,len).equals("ští")) {
            buffer.replace(len-2 ,len, "sk");
            return;
        }
        buffer.delete(len-1 , len);
        return;
    }

    private void removePossessives(StringBuffer buffer) {
        int len = buffer.length();
        if (len > 5) {
            if (buffer.substring(len-2 ,len).equals("ov")) {
                buffer.delete(len-2 , len);
                return;
            }
            if (buffer.substring(len-2,len).equals("ův")) {
                buffer.delete(len-2 , len);
                return;
            }
            if (buffer.substring(len-2 ,len).equals("in")) {
                buffer.delete(len-1 , len);
                palatalise(buffer);
                return;
            }
        }
    }

    private void removeCase(StringBuffer buffer) {
        int len = buffer.length();
        if (len > 7 && buffer.substring(len-5 ,len).equals("atech")) {
            buffer.delete(len-5 , len);
            return;
        }
        if (len > 6) {
            if (buffer.substring(len-4 ,len).equals("ětem")) {
                buffer.delete(len-3 , len);
                palatalise(buffer);
                return;
            }
            if (buffer.substring(len-4 ,len).equals("atům")) {
                buffer.delete(len-4 , len);
                return;
            }
        }
        if (len > 5) {
            if (buffer.substring(len-3,len).equals("ech") ||
                    buffer.substring(len-3,len).equals("ich") ||
                    buffer.substring(len-3,len).equals("ích")) {
                buffer.delete(len-2 , len);
                palatalise(buffer);
                return;
            }
            if (buffer.substring(len-3,len).equals("ého") ||
                    buffer.substring(len-3,len).equals("ěmi") ||
                    buffer.substring(len-3,len).equals("emi") ||
                    buffer.substring(len-3,len).equals("ému") ||
                    buffer.substring(len-3,len).equals("ete") ||
                    buffer.substring(len-3,len).equals("eti") ||
                    buffer.substring(len-3,len).equals("ěte") ||
                    buffer.substring(len-3,len).equals("ěti") ||
                    buffer.substring(len-3,len).equals("iho") ||
                    buffer.substring(len-3,len).equals("ího") ||
                    buffer.substring(len-3,len).equals("ími") ||
                    buffer.substring(len-3,len).equals("imu")) {

                buffer.delete(len-2 , len);
                palatalise(buffer);
                return;
            }
            if (buffer.substring(len-3,len).equals("ách") ||
                    buffer.substring(len-3,len).equals("ata") ||
                    buffer.substring(len-3,len).equals("aty") ||
                    buffer.substring(len-3,len).equals("ých") ||
                    buffer.substring(len-3,len).equals("ama") ||
                    buffer.substring(len-3,len).equals("ami") ||
                    buffer.substring(len-3,len).equals("ové") ||
                    buffer.substring(len-3,len).equals("ovi") ||
                    buffer.substring(len-3,len).equals("ými")) {
                buffer.delete(len-3 , len);
                return;
            }
        }
        if (len > 4) {
            if (buffer.substring(len-2,len).equals("em")) {
                buffer.delete(len-1 , len);
                palatalise(buffer);
                return;
            }
            if (buffer.substring(len-2,len).equals("es") ||
                    buffer.substring(len-2,len).equals("ém") ||
                    buffer.substring(len-2,len).equals("ím")) {
                buffer.delete(len-2 , len);
                palatalise(buffer);
                return;
            }
            if (buffer.substring(len-2,len).equals("ům")) {
                buffer.delete(len-2 , len);
                return;
            }
            if (buffer.substring(len-2,len).equals("at") ||
                    buffer.substring(len-2,len).equals("ám") ||
                    buffer.substring(len-2,len).equals("os") ||
                    buffer.substring(len-2,len).equals("us") ||
                    buffer.substring(len-2,len).equals("ým") ||
                    buffer.substring(len-2,len).equals("mi") ||
                    buffer.substring(len-2,len).equals("ou")) {
                buffer.delete(len-2 , len);
                return;
            }
        }
        if (len > 3) {
            if (buffer.substring(len-1,len).equals("e") ||
                    buffer.substring(len-1,len).equals("i")) {
                palatalise(buffer);
                return;
            }
            if (buffer.substring(len-1,len).equals("í") ||
                    buffer.substring(len-1,len).equals("ě")) {
                palatalise(buffer);
                return;
            }
            if (buffer.substring(len-1,len).equals("u") ||
                    buffer.substring(len-1,len).equals("y") ||
                    buffer.substring(len-1,len).equals("ů")) {
                buffer.delete(len-1 , len);
                return;
            }
            if (buffer.substring(len-1,len).equals("a") ||
                    buffer.substring(len-1,len).equals("o") ||
                    buffer.substring(len-1,len).equals("á") ||
                    buffer.substring(len-1,len).equals("é") ||
                    buffer.substring(len-1,len).equals("ý")) {
                buffer.delete(len-1 , len);
                return;
            }
        }
    }

    public static void main(String[] args) throws IOException {
        if (args.length !=  1
            || !args[0].equals("light") && !args[0].equals("aggressive")) {
            System.err.println("usage: java CzechStemmer light|aggressive");
            System.exit(2);
        }
        CzechStemmer stemmer = new CzechStemmer(args[0].equals("aggressive"));
        BufferedReader stdin = new BufferedReader(
            new InputStreamReader(System.in));
        String line;
        while (null !=  (line = stdin.readLine())) {
            java.util.Scanner scanner = new java.util.Scanner(line);
            if (scanner.hasNext()) {
                System.out.print(stemmer.stem(scanner.next()));
                while (scanner.hasNext()) {
                    System.out.print(" " + stemmer.stem(scanner.next()));
                }
            }
            System.out.println();
        }
    }
}
