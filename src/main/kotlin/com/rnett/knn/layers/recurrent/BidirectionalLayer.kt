package com.rnett.knn.layers.recurrent

import com.rnett.knn.layers.Layer
import org.deeplearning4j.nn.conf.layers.BaseRecurrentLayer as DL4JBaseRecurrentLayer
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional

class BidirectionalLayer(
    var layer: BaseRecurrentLayer<DL4JBaseRecurrentLayer, *>,
    var mode: Bidirectional.Mode = Bidirectional.Mode.CONCAT,
    name: String = "bidirectional_${layer.name}",
    layerBuilder: BidirectionalLayer.() -> Unit = {}
) :
    Layer<Bidirectional, BidirectionalLayer>(name) {

    override fun builder() = Bidirectional.Builder(mode, layer.build())

    override fun outputShape(inputShape: List<Int>): List<Int> {
        val underlying = layer.outputShape(inputShape)

        return if (mode == Bidirectional.Mode.CONCAT)
            listOf(2 * underlying[0]) + underlying.drop(1)
        else
            underlying
    }

    init{
        layerBuilder()
    }

}