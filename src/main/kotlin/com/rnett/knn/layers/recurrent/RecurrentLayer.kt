package com.rnett.knn.layers.recurrent

import com.rnett.knn.layers.feedforeward.FeedForwardLayer
import org.deeplearning4j.nn.api.layers.LayerConstraint
import org.deeplearning4j.nn.conf.distribution.Distribution
import org.deeplearning4j.nn.conf.layers.LSTM
import org.deeplearning4j.nn.conf.layers.Layer
import org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn
import org.nd4j.linalg.activations.impl.ActivationSigmoid
import org.nd4j.weightinit.WeightInit
import org.deeplearning4j.nn.conf.layers.BaseRecurrentLayer as DL4JBaseRecurrentLayer

abstract class BaseRecurrentLayer<T : DL4JBaseRecurrentLayer, S : BaseRecurrentLayer<T, S>>(
    nodes: Int,
    name: String = "recurrent"
) :
    FeedForwardLayer<T, S>(name, nodes) {

    override fun outputShape(inputShape: List<Int>) = listOf(this.nodes, inputShape[1])

    var recurrentConstraints = mutableListOf<LayerConstraint>()
    var inputWeightConstraints = mutableListOf<LayerConstraint>()

    var weightInitRecurrent: WeightInit? = null
    private var _distRecurrent: Distribution? = null

    var distRecurrent: Distribution?
        get() = _distRecurrent
        set(v) {
            _distRecurrent = v
            weightInitRecurrent = WeightInit.DISTRIBUTION
        }

    override fun apply(builder: Layer.Builder<*>) {
        super.apply(builder)

        builder as DL4JBaseRecurrentLayer.Builder

        builder.constrainRecurrent(*recurrentConstraints.toTypedArray())
        builder.constrainInputWeights(*inputWeightConstraints.toTypedArray())
        builder.weightInitRecurrent(weightInit)

        if (_distRecurrent != null)
            builder.dist(distRecurrent)
    }
}

class SimpleRnnLayer(
    nodes: Int, name: String = "simplernn",
    layerBuilder: SimpleRnnLayer.() -> Unit = {}
) :
    BaseRecurrentLayer<SimpleRnn, SimpleRnnLayer>(nodes, name) {
    override fun builder() = SimpleRnn.Builder()

    init {
        layerBuilder()
    }
}

class LSTMLayer(
    nodes: Int, name: String = "lstm",
    layerBuilder: LSTMLayer.() -> Unit = {}
) : BaseRecurrentLayer<LSTM, LSTMLayer>(nodes, name) {
    override fun builder() = LSTM.Builder()

    var forgetGateBiasInit: Double = -1.0
    var gateActivation = ActivationSigmoid()

    override fun apply(builder: Layer.Builder<*>) {
        super.apply(builder)

        builder as LSTM.Builder
        builder.forgetGateBiasInit(forgetGateBiasInit)
        builder.gateActivationFunction(gateActivation)
    }

    init{
        layerBuilder()
    }

}