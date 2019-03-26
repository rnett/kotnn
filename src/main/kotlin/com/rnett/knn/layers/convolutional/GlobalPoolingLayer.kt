package com.rnett.knn.layers.convolutional

import com.rnett.knn.layers.Layer
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer as DL4JGlobalPoolingLayer
import org.deeplearning4j.nn.conf.layers.PoolingType

class GlobalPoolingLayer(var poolingType: PoolingType,
                         var poolingDimensions: List<Int>,
                         var pnorm: Int = 2,
                         var collapseDimensions: Boolean = true,
                         name: String = "globalpooling",
                         layerBuilder: GlobalPoolingLayer.() -> Unit = {})
    : Layer<DL4JGlobalPoolingLayer, GlobalPoolingLayer>(name){

    override fun outputShape(inputShape: List<Int>): List<Int> {
        if(inputShape.size == 1)
            throw IllegalArgumentException("Can't apply global pooling to 1D input")

        return if(collapseDimensions)
            listOf(inputShape[0])
        else
            listOf(inputShape[0]) + List(inputShape.size - 1){ 1 }
    }

    override fun builder() = DL4JGlobalPoolingLayer.Builder()

    override fun apply(builder: org.deeplearning4j.nn.conf.layers.Layer.Builder<*>) {
        super.apply(builder)
        builder as DL4JGlobalPoolingLayer.Builder

        builder.poolingType(poolingType)
        builder.poolingDimensions(*poolingDimensions.toIntArray())
        builder.pnorm(pnorm)
        builder.collapseDimensions(collapseDimensions)
    }

    init{
        layerBuilder()
    }
}