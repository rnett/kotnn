package com.rnett.knn.layers.sizing

import com.rnett.knn.*
import com.rnett.knn.layers.Layer
import com.rnett.knn.layers.samediff.Index
import com.rnett.knn.layers.samediff.SameDiffDef
import com.rnett.knn.layers.samediff.SameDiffLayer
import org.deeplearning4j.nn.conf.layers.NoParamLayer
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping1D
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping2D
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping3D
import org.deeplearning4j.nn.modelimport.keras.preprocessors.ReshapePreprocessor
import org.nd4j.autodiff.samediff.SDVariable

abstract class BaseCroppingLayer<T : NoParamLayer, S : BaseCroppingLayer<T, S, D>, D : NParams>(
    val dimensions: Int,
    name: String
) : Layer<T, S>(name) {
    abstract var cropping: D

}

//TODO test
class CroppingLayer(
    var cropping: List<Param2>,
    name: String = "cropping",
    layerBuilder: CroppingLayer.() -> Unit = {}
) : SameDiffLayer(name){

    constructor(vararg cropping: Param2,
                name: String = "cropping",
                layerBuilder: CroppingLayer.() -> Unit = {}) : this(cropping.toList(), name, layerBuilder)

    val dimensions = cropping.size

    override fun SameDiffDef.defineLayer(input: SDVariable): SDVariable {
        val crops = listOf(Index(0, null)) + cropping.map {
            Index(it.first, if(it.second == 0) null else -it.second)
        }
        return input[crops]
    }

    override fun outputShape(inputShape: List<Int>): List<Int> {
        if(inputShape.size != dimensions)
            throw IllegalArgumentException("Dimensions of input (${inputShape.size}) are not equal to dimensions of cropping ($dimensions)")

        return inputShape.zip(cropping).map { it.first - it.second.first - it.second.second }
    }

    init{
        layerBuilder()
    }
}

class Cropping1DLayer(
    cropping: Param2,
    name: String = "cropping1d",
    layerBuilder: Cropping1DLayer.() -> Unit = {}
) : BaseCroppingLayer<Cropping1D, Cropping1DLayer, Param2>(1, name) {

    constructor(
        cropTop: Int, cropBottom: Int,
        name: String = "cropping1d",
        layerBuilder: Cropping1DLayer.() -> Unit = {}
    ) : this(P(cropTop, cropBottom), name, layerBuilder)

    constructor(
        crop: Int,
        name: String = "cropping1d",
        layerBuilder: Cropping1DLayer.() -> Unit = {}
    ) : this(crop, crop, name, layerBuilder)

    override var cropping by nonNegativeParams(cropping)

    override fun builder() = Cropping1D.Builder(cropping.toIntArray())

    override fun outputShape(inputShape: List<Int>): List<Int> {
        if (inputShape.size != 2)
            throw IllegalArgumentException("Can only accept input of 2 dimension")

        return listOf(inputShape[0], inputShape[1] - cropping[0] - cropping[1])
    }

    init {
        layerBuilder()
    }
}

class Cropping2DLayer(
    cropping: Param4,
    name: String = "cropping2d",
    layerBuilder: Cropping2DLayer.() -> Unit = {}
) : BaseCroppingLayer<Cropping2D, Cropping2DLayer, Param4>(2, name) {

    constructor(
        cropTop: Int, cropBottom: Int,
        cropLeft: Int, cropRight: Int,
        name: String = "cropping2d",
        layerBuilder: Cropping2DLayer.() -> Unit = {}
    ) : this(P(cropTop, cropBottom, cropLeft, cropRight), name, layerBuilder)

    constructor(
        cropTopBottom: Param2, cropLeftRight: Param2,
        name: String = "cropping2d",
        layerBuilder: Cropping2DLayer.() -> Unit = {}
    ) : this(P(cropTopBottom[0], cropTopBottom[1], cropLeftRight[0], cropLeftRight[1]), name, layerBuilder)

    constructor(
        cropTopBottom: Int, cropLeftRight: Int,
        name: String = "cropping2d",
        layerBuilder: Cropping2DLayer.() -> Unit = {}
    ) : this(P(cropTopBottom, cropLeftRight), name, layerBuilder)

    constructor(
        cropDims: Param2,
        name: String = "cropping2d",
        layerBuilder: Cropping2DLayer.() -> Unit = {}
    ) : this(P(cropDims[0], cropDims[0], cropDims[1], cropDims[1]), name, layerBuilder)

    constructor(
        cropAll: Int,
        name: String = "cropping2d",
        layerBuilder: Cropping2DLayer.() -> Unit = {}
    ) : this(P(cropAll, cropAll, cropAll, cropAll), name, layerBuilder)

    override var cropping by nonNegativeParams(cropping)

    override fun builder() = Cropping2D.Builder(cropping.toIntArray())

    override fun outputShape(inputShape: List<Int>): List<Int> {
        if (inputShape.size != 3)
            throw IllegalArgumentException("Can only accept input of 3 dimension")

        return listOf(
            inputShape[0],
            inputShape[1] - cropping[0] - cropping[1],
            inputShape[2] - cropping[2] - cropping[3]
        )
    }

    init {
        layerBuilder()
    }
}

class Cropping3DLayer(
    cropping: Param6,
    name: String = "cropping3d",
    layerBuilder: Cropping3DLayer.() -> Unit = {}
) : BaseCroppingLayer<Cropping3D, Cropping3DLayer, Param6>(3, name) {

    constructor(
        cropTop: Int, cropBottom: Int,
        cropLeft: Int, cropRight: Int,
        cropTowards: Int, cropAway: Int,
        name: String = "cropping3d",
        layerBuilder: Cropping3DLayer.() -> Unit = {}
    ) : this(P(cropTowards, cropAway, cropTop, cropBottom, cropLeft, cropRight), name, layerBuilder)

    constructor(
        cropTopBottom: Param2, cropLeftRight: Param2, cropTowardsAway: Param2,
        name: String = "cropping3d",
        layerBuilder: Cropping3DLayer.() -> Unit = {}
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
        name: String = "cropping3d",
        layerBuilder: Cropping3DLayer.() -> Unit = {}
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
        name: String = "cropping3d",
        layerBuilder: Cropping3DLayer.() -> Unit = {}
    ) : this(P(cropDims[0], cropDims[0], cropDims[1], cropDims[1], cropDims[2], cropDims[2]), name, layerBuilder)

    constructor(
        cropAll: Int,
        name: String = "cropping3d",
        layerBuilder: Cropping3DLayer.() -> Unit = {}
    ) : this(P(cropAll, cropAll, cropAll, cropAll, cropAll, cropAll), name, layerBuilder)

    override var cropping by nonNegativeParams(cropping)

    override fun builder() = Cropping3D.Builder(cropping.toIntArray())

    override fun outputShape(inputShape: List<Int>): List<Int> {
        if (inputShape.size != 4)
            throw IllegalArgumentException("Can only accept input of 4 dimension")

        return listOf(
            inputShape[0],
            inputShape[1] - cropping[0] - cropping[1],
            inputShape[2] - cropping[2] - cropping[3]
        )
    }

    init {
        layerBuilder()
    }
}