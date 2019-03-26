package com.rnett.knn.layers.util

import com.rnett.knn.layers.Layer
import org.deeplearning4j.nn.conf.layers.misc.ElementWiseMultiplicationLayer as DL4JElementWiseMultiplicationLayer

class ElementWiseMultiplicationLayer(
    name: String = "elementwisemultiplication",
    layerBuilder: ElementWiseMultiplicationLayer.() -> Unit = {}
) :
    Layer<DL4JElementWiseMultiplicationLayer, ElementWiseMultiplicationLayer>(name) {
    override fun outputShape(inputShape: List<Int>) = inputShape

    override fun builder() = DL4JElementWiseMultiplicationLayer.Builder()

    init {
        layerBuilder
    }
}