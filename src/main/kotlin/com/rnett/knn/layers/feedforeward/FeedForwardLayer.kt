package com.rnett.knn.layers.feedforeward

import com.rnett.knn.enforceLength
import org.deeplearning4j.nn.conf.layers.Layer
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer as DL4JFeedForwardLayer

abstract class FeedForwardLayer<out T : DL4JFeedForwardLayer, S : FeedForwardLayer<T, S>>(name: String) : BaseLayer<T, S>(name){


    @Deprecated("Will be automatically set in almost all setups")
    var nIn: Int = 0
    open var nOut: Int = 0

    constructor(name: String, nodes: Int) : this(name){
        this.nodes = nodes
    }

    var nodes
        get() = nOut
        set(v){ nOut = v }

    override fun apply(builder: Layer.Builder<*>) {
        super.apply(builder)

        builder as DL4JFeedForwardLayer.Builder

        //builder.nIn(nIn)
        try {
            builder.nOut(nOut)
        } catch (e: UnsupportedOperationException) {

        }
    }

    override fun outputShape(inputShape: List<Int>): List<Int> {
        inputShape.enforceLength(1, "Input of normal FeedForward Layers must have one dimension ")

        return listOf(nodes)
    }

}