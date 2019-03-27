package com.rnett.knn.layers.recurrent

import com.rnett.knn.layers.Layer
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep
import org.deeplearning4j.nn.conf.layers.Layer as DL4JLayer


class LastTimeStepLayer(
    var layer: Layer<*, *>, name: String = "lasttimestep",
    layerBuilder: LastTimeStepLayer.() -> Unit = {}
) : Layer<LastTimeStep, LastTimeStepLayer>(name) {

    override fun outputShape(inputShape: List<Int>) = listOf(inputShape[0])

    override fun builder() = Builder(layer.buildLayer())

    class Builder(val layer: DL4JLayer) : DL4JLayer.Builder<Builder>() {
        override fun <E : DL4JLayer> build(): E = LastTimeStep(layer) as E
    }
}