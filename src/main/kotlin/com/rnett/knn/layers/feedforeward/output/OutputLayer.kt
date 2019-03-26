package com.rnett.knn.layers.feedforeward.output

import com.rnett.knn.layers.feedforeward.FeedForwardLayer
import org.deeplearning4j.nn.conf.layers.Layer
import org.nd4j.linalg.activations.impl.ActivationIdentity
import org.nd4j.linalg.activations.impl.ActivationSoftmax
import org.nd4j.linalg.lossfunctions.ILossFunction
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT
import org.deeplearning4j.nn.conf.layers.BaseOutputLayer as DL4JBaseOutputLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer as DL4JOutputLayer
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer as DL4JRnnOutputLayer
import org.deeplearning4j.nn.conf.ocnn.OCNNOutputLayer as DL4JOCNNOutputLayer
import org.deeplearning4j.nn.conf.layers.CenterLossOutputLayer as DL4JCenterLossOutputLayer

abstract class BaseOutputLayer<out T : DL4JBaseOutputLayer, S : BaseOutputLayer<T, S>>(
    nodes: Int,
    name: String,
    var loss: ILossFunction
) :
    FeedForwardLayer<T, S>(name, nodes) {

    var hasBias = true

    override fun apply(builder: Layer.Builder<*>) {
        super.apply(builder)

        builder as DL4JBaseOutputLayer.Builder
        builder.hasBias(hasBias)
        builder.lossFunction(loss)
    }
}

class OutputLayer(
    nodes: Int,
    loss: ILossFunction = LossMCXENT(),
    name: String = "output",
    layerBuilder: OutputLayer.() -> Unit = {}
) : BaseOutputLayer<DL4JOutputLayer, OutputLayer>(nodes, name, loss) {
    override fun builder() = DL4JOutputLayer.Builder()

    init {
        activation = ActivationSoftmax()
        layerBuilder()
    }
}

class RnnOutputLayer(
    nodes: Int,
    loss: ILossFunction = LossMCXENT(),
    name: String = "output",
    layerBuilder: RnnOutputLayer.() -> Unit = {}
) : BaseOutputLayer<DL4JRnnOutputLayer, RnnOutputLayer>(nodes, name, loss) {
    override fun builder() = DL4JRnnOutputLayer.Builder()

    init {
        activation = ActivationSoftmax()
        layerBuilder()
    }
}

class OCNNOutputLayer(
    nodes: Int,
    var hiddenLayerSize: Int,
    loss: ILossFunction = LossMCXENT(),
    name: String = "ocnnoutput",
    layerBuilder: OCNNOutputLayer.() -> Unit = {}
) : BaseOutputLayer<DL4JOCNNOutputLayer, OCNNOutputLayer>(nodes, name, loss) {
    override fun builder() = DL4JOCNNOutputLayer.Builder()

    var nu: Double = 0.04
    var windowSize: Int = 10_000
    var ocnnActivation = ActivationIdentity()
    var initialRValue: Double = 0.1
    var configureR = true

    override fun apply(builder: Layer.Builder<*>) {
        super.apply(builder)

        builder as DL4JOCNNOutputLayer.Builder

        builder.hiddenLayerSize(hiddenLayerSize)
        builder.nu(nu)
        builder.windowSize(windowSize)
        builder.activation(ocnnActivation)
        builder.initialRValue(initialRValue)
        builder.configureR((configureR))
    }

    init{
        layerBuilder()
    }
}

class CenterLossOutputLayer(
    nodes: Int,
    loss: ILossFunction = LossMCXENT(),
    name: String = "centerlossoutput",
    layerBuilder: CenterLossOutputLayer.() -> Unit = {}
) :
    BaseOutputLayer<DL4JCenterLossOutputLayer, CenterLossOutputLayer>(nodes, name, loss) {
    override fun builder() = DL4JCenterLossOutputLayer.Builder()

    var alpha: Double = 0.05
    var lambda: Double = 2e-4
    var gradientCheck = false

    override fun apply(builder: Layer.Builder<*>) {
        super.apply(builder)

        builder as DL4JCenterLossOutputLayer.Builder

        builder.alpha(alpha)
        builder.lambda(lambda)
        builder.gradientCheck(gradientCheck)
    }

    init{
        layerBuilder()
    }
}
