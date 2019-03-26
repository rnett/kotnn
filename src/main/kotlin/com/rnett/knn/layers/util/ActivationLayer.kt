package com.rnett.knn.layers.util

import com.rnett.knn.layers.Layer
import org.nd4j.linalg.activations.IActivation
import org.deeplearning4j.nn.conf.layers.ActivationLayer as DL4JActivationLayer

class ActivationLayer(
    var activation: IActivation, name: String = "activation",
    layerBuilder: ActivationLayer.() -> Unit = {}
) :
    Layer<DL4JActivationLayer, ActivationLayer>(name) {
    override fun builder() = DL4JActivationLayer.Builder()
    override fun outputShape(inputShape: List<Int>) = inputShape

    override fun apply(builder: org.deeplearning4j.nn.conf.layers.Layer.Builder<*>) {
        super.apply(builder)

        builder as DL4JActivationLayer.Builder
        builder.activation(activation)
    }

    init{
        layerBuilder()
    }
}