package layers;

import org.junit.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.Arrays;

public class IssueTest {
    @Test
    public void test() {
        SameDiff SD = SameDiff.create();
        SDVariable in = SD.one("test", 5, 8, 3, 4);
        SDVariable out = in.reshape(-1, 4);
        SDVariable out1 = out.reshape(4, 15, -1);
        SDVariable out2 = SD.dot(out1, out1, 2);

        SDVariable out3 = out2.reshape(-1, 4);  // <----  error here

        System.out.println(Arrays.toString(out3.eval().toFloatMatrix()));

    }
}
