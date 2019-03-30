package layers;

import com.google.common.collect.Lists;
import org.junit.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

public class IssueTest {

    @Test
    public void test1() {
        SameDiff SD = SameDiff.create();
        SDVariable a = SD.reshape("a", SD.linspace("at", 1, 15, 15), 3, 5);//.add(1);
        SDVariable b = SD.one("b", 3, 5);//.add(3);

        SDVariable out = a.mul(b).mean(1);

        System.out.println(Arrays.toString(out.eval().shape()));
        out.eval();

        SD.execBackwards(null, Lists.asList("a", new String[]{}));
        System.out.println(out.eval());
        System.out.println(SD.grad("a").eval());
    }

    @Test
    public void test2() {
        SameDiff SD = SameDiff.create();
        SDVariable a = SD.reshape("a", SD.linspace("at", 1, 15, 15), 3, 5);//.add(1);
        SDVariable b = SD.one("b", 3, 5);//.add(3);

        SDVariable out = a.mul(b).mean(0, 1);

        System.out.println(Arrays.toString(out.eval().shape()));
        out.eval();

        SD.execBackwards(null, Lists.asList("a", new String[]{}));
        System.out.println(out.eval());
        System.out.println(SD.grad("a").eval());
    }

    @Test
    public void testW() {
        INDArray a1 = Nd4j.linspace(1, 15, 15).reshape(3, 5);
        INDArray b1 = Nd4j.ones(3, 5);
        LossWasserstein lw = new LossWasserstein();
        INDArray loss = lw.computeScoreArray(b1, a1, new ActivationIdentity(), null);
        INDArray grad = lw.computeGradient(b1, a1, new ActivationIdentity(), null);
        System.out.println(loss);
        System.out.println(grad);
    }
}
