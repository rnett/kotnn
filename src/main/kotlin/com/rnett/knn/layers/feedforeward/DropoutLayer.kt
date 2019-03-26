package com.rnett.knn.layers.feedforeward

import org.deeplearning4j.nn.conf.dropout.IDropout
import org.deeplearning4j.nn.conf.layers.Layer
import org.deeplearning4j.nn.conf.layers.DropoutLayer as DL4JDropoutLayer

class DropoutLayer(
    name: String = "dropout",
    layerBuilder: DropoutLayer.() -> Unit = {}
) : FeedForwardLayer<DL4JDropoutLayer, DropoutLayer>(name) {
    override fun builder() = DL4JDropoutLayer.Builder()

    override fun outputShape(inputShape: List<Int>) = inputShape

    constructor(dropout: Double, name: String = "dropout", layerBuilder: DropoutLayer.() -> Unit = {}) : this(
        name,
        layerBuilder
    ) {
        dropoutProbability = dropout
    }

    constructor(dropout: IDropout, name: String = "dropout", layerBuilder: DropoutLayer.() -> Unit = {}) : this(
        name,
        layerBuilder
    ) {
        this.dropout = dropout
    }

    init {
        layerBuilder()
    }
}