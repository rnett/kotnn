package com.rnett.knn.layers.feedforeward

import org.deeplearning4j.nn.conf.layers.Layer
import org.deeplearning4j.nn.conf.layers.DenseLayer as DL4JDenseLayer

class DenseLayer(
    nodes: Int,
    name: String = "dense",
    layerBuilder: DenseLayer.() -> Unit = {}
) : FeedForwardLayer<DL4JDenseLayer, DenseLayer>(name, nodes) {
    var hasBias = true

    override fun builder() = DL4JDenseLayer.Builder()


    override fun apply(builder: Layer.Builder<*>) {
        super.apply(builder)

        builder as DL4JDenseLayer.Builder
        builder.hasBias(hasBias)
    }

    init {
        layerBuilder()
    }
}