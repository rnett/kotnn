package com.rnett.knn.layers.convolutional

import com.rnett.knn.*
import com.rnett.knn.layers.Layer
import org.deeplearning4j.nn.conf.layers.Convolution3D
import org.deeplearning4j.nn.conf.layers.BaseUpsamplingLayer as DL4JBaseUpsamplingLayer
import org.deeplearning4j.nn.conf.layers.Upsampling1D as DL4JUpsampling1DLayer
import org.deeplearning4j.nn.conf.layers.Upsampling2D as DL4JUpsampling2DLayer
import org.deeplearning4j.nn.conf.layers.Upsampling3D as DL4JUpsampling3DLayer

abstract class BaseUpsamplingLayer<T : DL4JBaseUpsamplingLayer, S : BaseUpsamplingLayer<T, S, D>, D : NParams>(
    val dimensions: Int,
    name: String = "upsampling"
) :
    Layer<T, S>(name) {

    abstract var size: D
}

class Upsampling1DLayer(
    size: Int,
    name: String = "upsampling1d",
    layerBuilder: Upsampling1DLayer.() -> Unit
) : BaseUpsamplingLayer<DL4JUpsampling1DLayer, Upsampling1DLayer, Param1>(1, name) {

    override var size by nonNegativeParams(size.p1)

    override fun outputShape(inputShape: List<Int>) = listOf(inputShape[0], inputShape[1] * size[0])

    override fun builder() = DL4JUpsampling1DLayer.Builder()

    override fun apply(builder: org.deeplearning4j.nn.conf.layers.Layer.Builder<*>) {
        super.apply(builder)
        builder as DL4JUpsampling1DLayer.Builder

        builder.size(size[0])
    }

    init {
        layerBuilder()
    }
}

class Upsampling2DLayer(
    size: Param2,
    name: String = "upsampling2d",
    layerBuilder: Upsampling2DLayer.() -> Unit
) : BaseUpsamplingLayer<DL4JUpsampling2DLayer, Upsampling2DLayer, Param2>(2, name) {

    constructor(
        size: Int,
        name: String = "upsampling2d",
        layerBuilder: Upsampling2DLayer.() -> Unit
    ) : this(size.p2, name, layerBuilder)

    constructor(
        sizeH: Int, sizeW: Int,
        name: String = "upsampling2d",
        layerBuilder: Upsampling2DLayer.() -> Unit
    ) : this(P(sizeH, sizeW), name, layerBuilder)

    override var size by nonNegativeParams(size)

    override fun outputShape(inputShape: List<Int>) = listOf(
        inputShape[0],
        inputShape[1] * size[0],
        inputShape[2] * size[1]
    )

    override fun builder() = DL4JUpsampling2DLayer.Builder()

    override fun apply(builder: org.deeplearning4j.nn.conf.layers.Layer.Builder<*>) {
        super.apply(builder)
        builder as DL4JUpsampling2DLayer.Builder

        builder.size(size.toIntArray())
    }

    init {
        layerBuilder()
    }
}

class Upsampling3DLayer(
    size: Param3,
    name: String = "upsampling3d",
    layerBuilder: Upsampling3DLayer.() -> Unit
) : BaseUpsamplingLayer<DL4JUpsampling3DLayer, Upsampling3DLayer, Param3>(3, name) {

    constructor(
        size: Int,
        name: String = "upsampling2d",
        layerBuilder: Upsampling3DLayer.() -> Unit
    ) : this(size.p3, name, layerBuilder)

    constructor(
        sizeD: Int, sizeH: Int, sizeW: Int,
        name: String = "upsampling2d",
        layerBuilder: Upsampling3DLayer.() -> Unit
    ) : this(P(sizeD, sizeH, sizeW), name, layerBuilder)

    override var size by nonNegativeParams(size)

    var dataFormat: Convolution3D.DataFormat = Convolution3D.DataFormat.NCDHW

    override fun outputShape(inputShape: List<Int>) =
        if (dataFormat == Convolution3D.DataFormat.NCDHW)
            listOf(
                inputShape[0],
                inputShape[1] * size[0],
                inputShape[2] * size[1],
                inputShape[3] * size[2]
            )
        else
            listOf(
                inputShape[0] * size[0],
                inputShape[1] * size[1],
                inputShape[2] * size[2],
                inputShape[3]
            )

    override fun builder() = DL4JUpsampling3DLayer.Builder()

    override fun apply(builder: org.deeplearning4j.nn.conf.layers.Layer.Builder<*>) {
        super.apply(builder)
        builder as DL4JUpsampling3DLayer.Builder

        builder.size(size.toIntArray())
    }

    init {
        layerBuilder()
    }
}