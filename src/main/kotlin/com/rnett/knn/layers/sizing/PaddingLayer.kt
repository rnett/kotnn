package com.rnett.knn.layers.sizing

import com.rnett.knn.*
import com.rnett.knn.layers.Layer
import com.rnett.knn.layers.samediff.Index
import com.rnett.knn.layers.samediff.SameDiffDef
import com.rnett.knn.layers.samediff.SameDiffLayer
import org.deeplearning4j.nn.conf.layers.NoParamLayer
import org.deeplearning4j.nn.conf.layers.ZeroPadding1DLayer as DL4JPadding1DLayer
import org.deeplearning4j.nn.conf.layers.ZeroPaddingLayer as DL4JPadding2DLayer
import org.deeplearning4j.nn.conf.layers.ZeroPadding3DLayer as DL4JPadding3DLayer
import org.nd4j.autodiff.samediff.SDVariable

abstract class BasePaddingLayer<T : NoParamLayer, S : BasePaddingLayer<T, S, D>, D : NParams>(
    val dimensions: Int,
    name: String
) : Layer<T, S>(name) {
    abstract var padding: D

}

//TODO test
class PaddingLayer(
    var padding: List<Param2>,
    name: String = "padding",
    layerBuilder: PaddingLayer.() -> Unit = {}
) : SameDiffLayer(name){

    val dimensions = padding.size

    //TODO verify
    override fun SameDiffDef.defineLayer(input: SDVariable): SDVariable {
        var padded = input
        (listOf(0.p2) + padding).forEachIndexed{ dim, pad ->

            val firstPadding = padded.shape.toList().map { it.toInt() }.toMutableList()
            firstPadding[dim] = pad[0]
            padded = SD.concat(dim, SD.zero("zeroFirst$dim", *firstPadding.map { it.toLong() }.toLongArray()), padded)

            val secondPadding = padded.shape.toList().map { it.toInt() }.toMutableList()
            firstPadding[dim] = pad[1]
            padded = SD.concat(dim, padded, SD.zero("zeroSecond$dim", *secondPadding.map { it.toLong() }.toLongArray()))
        }
        return padded
    }

    override fun outputShape(inputShape: List<Int>): List<Int> {
        if(inputShape.size != dimensions)
            throw IllegalArgumentException("Dimensions of input (${inputShape.size}) are not equal to dimensions of padding ($dimensions)")

        return inputShape.zip(padding).map { it.first + it.second.first + it.second.second }
    }

    init{
        layerBuilder()
    }
}

class Padding1DLayer(
    padding: Param2,
    name: String = "padding1d",
    layerBuilder: Padding1DLayer.() -> Unit = {}
) : BasePaddingLayer<DL4JPadding1DLayer, Padding1DLayer, Param2>(1, name) {

    constructor(
        cropTop: Int, cropBottom: Int,
        name: String = "padding1d",
        layerBuilder: Padding1DLayer.() -> Unit = {}
    ) : this(P(cropTop, cropBottom), name, layerBuilder)

    constructor(
        crop: Int,
        name: String = "padding1d",
        layerBuilder: Padding1DLayer.() -> Unit = {}
    ) : this(crop, crop, name, layerBuilder)

    override var padding by nonNegativeParams(padding)

    override fun builder() = DL4JPadding1DLayer.Builder(padding[0], padding[1])

    override fun outputShape(inputShape: List<Int>): List<Int> {
        if (inputShape.size != 2)
            throw IllegalArgumentException("Can only accept input of 2 dimension")

        return listOf(inputShape[0], inputShape[1] + padding[0] + padding[1])
    }

    init {
        layerBuilder()
    }
}

class Padding2DLayer(
    padding: Param4,
    name: String = "padding2d",
    layerBuilder: Padding2DLayer.() -> Unit = {}
) : BasePaddingLayer<DL4JPadding2DLayer, Padding2DLayer, Param4>(2, name) {

    constructor(
        cropTop: Int, cropBottom: Int,
        cropLeft: Int, cropRight: Int,
        name: String = "padding2d",
        layerBuilder: Padding2DLayer.() -> Unit = {}
    ) : this(P(cropTop, cropBottom, cropLeft, cropRight), name, layerBuilder)

    constructor(
        cropTopBottom: Param2, cropLeftRight: Param2,
        name: String = "padding2d",
        layerBuilder: Padding2DLayer.() -> Unit = {}
    ) : this(P(cropTopBottom[0], cropTopBottom[1], cropLeftRight[0], cropLeftRight[1]), name, layerBuilder)

    constructor(
        cropTopBottom: Int, cropLeftRight: Int,
        name: String = "padding2d",
        layerBuilder: Padding2DLayer.() -> Unit = {}
    ) : this(P(cropTopBottom, cropLeftRight), name, layerBuilder)

    constructor(
        cropDims: Param2,
        name: String = "padding2d",
        layerBuilder: Padding2DLayer.() -> Unit = {}
    ) : this(P(cropDims[0], cropDims[0], cropDims[1], cropDims[1]), name, layerBuilder)

    constructor(
        cropAll: Int,
        name: String = "padding2d",
        layerBuilder: Padding2DLayer.() -> Unit = {}
    ) : this(P(cropAll, cropAll, cropAll, cropAll), name, layerBuilder)

    override var padding by nonNegativeParams(padding)

    override fun builder() = DL4JPadding2DLayer.Builder(padding.toIntArray())

    override fun outputShape(inputShape: List<Int>): List<Int> {
        if (inputShape.size != 3)
            throw IllegalArgumentException("Can only accept input of 3 dimension")

        return listOf(
            inputShape[0],
            inputShape[1] + padding[0] + padding[1],
            inputShape[2] + padding[2] + padding[3]
        )
    }

    init {
        layerBuilder()
    }
}

class Padding3DLayer(
    padding: Param6,
    name: String = "padding3d",
    layerBuilder: Padding3DLayer.() -> Unit = {}
) : BasePaddingLayer<DL4JPadding3DLayer, Padding3DLayer, Param6>(3, name) {

    constructor(
        cropTop: Int, cropBottom: Int,
        cropLeft: Int, cropRight: Int,
        cropTowards: Int, cropAway: Int,
        name: String = "padding3d",
        layerBuilder: Padding3DLayer.() -> Unit = {}
    ) : this(P(cropTowards, cropAway, cropTop, cropBottom, cropLeft, cropRight), name, layerBuilder)

    constructor(
        cropTopBottom: Param2, cropLeftRight: Param2, cropTowardsAway: Param2,
        name: String = "padding3d",
        layerBuilder: Padding3DLayer.() -> Unit = {}
    ) : this(
        P(
            cropTowardsAway[0],
            cropTowardsAway[1],
            cropTopBottom[0],
            cropTopBottom[1],
            cropLeftRight[0],
            cropLeftRight[1]
        ), name, layerBuilder
    )

    constructor(
        cropTopBottom: Int, cropLeftRight: Int, cropTowardsAway: Int,
        name: String = "padding3d",
        layerBuilder: Padding3DLayer.() -> Unit = {}
    ) : this(
        P(
            cropTowardsAway,
            cropTowardsAway,
            cropTopBottom,
            cropTopBottom,
            cropLeftRight,
            cropLeftRight
        ), name, layerBuilder
    )

    constructor(
        cropDims: Param3,
        name: String = "padding3d",
        layerBuilder: Padding3DLayer.() -> Unit = {}
    ) : this(P(cropDims[0], cropDims[0], cropDims[1], cropDims[1], cropDims[2], cropDims[2]), name, layerBuilder)

    constructor(
        cropAll: Int,
        name: String = "padding3d",
        layerBuilder: Padding3DLayer.() -> Unit = {}
    ) : this(P(cropAll, cropAll, cropAll, cropAll, cropAll, cropAll), name, layerBuilder)

    override var padding by nonNegativeParams(padding)

    override fun builder() = DL4JPadding3DLayer.Builder(padding.toIntArray())

    override fun outputShape(inputShape: List<Int>): List<Int> {
        if (inputShape.size != 4)
            throw IllegalArgumentException("Can only accept input of 4 dimension")

        return listOf(
            inputShape[0],
            inputShape[1] + padding[0] + padding[1],
            inputShape[2] + padding[2] + padding[3]
        )
    }

    init {
        layerBuilder()
    }
}