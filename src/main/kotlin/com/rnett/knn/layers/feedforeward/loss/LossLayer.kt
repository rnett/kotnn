package com.rnett.knn.layers.feedforeward.loss

import com.rnett.knn.layers.feedforeward.FeedForwardLayer
import org.deeplearning4j.nn.conf.layers.Convolution3D
import org.deeplearning4j.nn.conf.layers.Layer
import org.deeplearning4j.nn.conf.layers.RnnLossLayer
import org.nd4j.linalg.activations.impl.ActivationIdentity
import org.nd4j.linalg.lossfunctions.ILossFunction
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT
import org.deeplearning4j.nn.conf.layers.Cnn3DLossLayer as DL4JCnn3DLossLayer
import org.deeplearning4j.nn.conf.layers.CnnLossLayer as DL4JCnnLossLayer
import org.deeplearning4j.nn.conf.layers.LossLayer as DL4JLossLayer

abstract class BaseLossLayer<out T : org.deeplearning4j.nn.conf.layers.FeedForwardLayer, S : BaseLossLayer<T, S>>(
    name: String,
    var loss: ILossFunction = LossMCXENT()
) :
    FeedForwardLayer<T, S>(name) {

    var hasBias = true

    override fun apply(builder: Layer.Builder<*>) {
        super.apply(builder)

        builder as org.deeplearning4j.nn.conf.layers.BaseOutputLayer.Builder
        builder.hasBias(hasBias)
        builder.lossFunction(loss)
    }

    override var nOut: Int
        get() = 0
        set(v) {
            throw UnsupportedOperationException("This layer has no parameters, thus nOut == nIn")
        }
}

class LossLayer(
    loss: ILossFunction = LossMCXENT(),
    name: String = "loss",
    layerBuilder: LossLayer.() -> Unit = {}
) : BaseLossLayer<DL4JLossLayer, LossLayer>(name, loss) {
    override fun builder() = DL4JLossLayer.Builder()

    init {
        activation = ActivationIdentity()
        layerBuilder()
    }

    //TODO use? override fun outputShape(inputShape: List<Int>) = inputShape
    //TODO I think outputShape is always zero
}

class RNNLossLayer(
    loss: ILossFunction = LossMCXENT(),
    name: String = "rnnloss",
    layerBuilder: RNNLossLayer.() -> Unit = {}
) : BaseLossLayer<RnnLossLayer, RNNLossLayer>(name, loss) {
    override fun builder() = RnnLossLayer.Builder()

    init {
        activation = ActivationIdentity()
        layerBuilder()
    }

    override fun outputShape(inputShape: List<Int>) = inputShape
}

class CnnLossLayer(
    loss: ILossFunction = LossMCXENT(),
    name: String = "cnnloss",
    layerBuilder: CnnLossLayer.() -> Unit = {}
) : BaseLossLayer<DL4JCnnLossLayer, CnnLossLayer>(name, loss) {
    override fun builder() = DL4JCnnLossLayer.Builder()

    init {
        activation = ActivationIdentity()
        layerBuilder()
    }

    override fun outputShape(inputShape: List<Int>) = inputShape
}

class Cnn3DLossLayer(
    loss: ILossFunction = LossMCXENT(),
    name: String = "cnn3dloss",
    layerBuilder: Cnn3DLossLayer.() -> Unit = {}
) : BaseLossLayer<DL4JCnn3DLossLayer, Cnn3DLossLayer>(name, loss) {

    var dataFormat: Convolution3D.DataFormat = Convolution3D.DataFormat.NCDHW

    override fun builder() = DL4JCnn3DLossLayer.Builder(dataFormat)

    init {
        activation = ActivationIdentity()
        layerBuilder()
    }

    override fun outputShape(inputShape: List<Int>) = inputShape
}
