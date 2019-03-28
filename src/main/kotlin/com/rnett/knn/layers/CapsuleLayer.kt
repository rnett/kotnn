package com.rnett.knn.layers

import com.rnett.knn.Param2
import com.rnett.knn.layers.convolutional.Convolution2DLayer
import com.rnett.knn.layers.samediff.SameDiffDef
import com.rnett.knn.layers.samediff.SameDiffLambdaDef
import com.rnett.knn.layers.samediff.SameDiffLambdaLayer
import com.rnett.knn.layers.samediff.SameDiffLayer
import com.rnett.knn.layers.util.product
import com.rnett.knn.p2
import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.util.ArrayUtil

//fun SameDiff.batchDot(
//    a: SDVariable,
//    aShape: List<Int>,
//    b: SDVariable,
//    bShape: List<Int>,
//    aDim: Int,
//    bDim: Int
//): SDVariable {
//
//    val SD = this
//
//    var A = a
//    var B = b
//
//    try {
//        if (aShape != a.shape?.map { it.toInt() } ?: a.eval().shape().map { it.toInt() }) {
//            println("Bad A shape")
//        }
//
//        if (bShape != b.shape?.map { it.toInt() } ?: b.eval().shape().map { it.toInt() }) {
//            println("Bad B shape")
//        }
//    } catch (e: Exception){
//
//    }
//
//    val aBatch = aShape[0]
//    val bBatch = bShape[0]
//
//    if (aBatch != bBatch)
//        throw IllegalArgumentException("Can only do batchDot on tensors with the same batch size")
//
//    var aDimension = if (aDim < 0) aShape.size + aDim else aDim
//    val bDimension = if (bDim < 0) bShape.size + bDim else bDim
//
//    val sizeA = aShape[aDimension]
//    val sizeB = bShape[bDimension]
//
//    val otherADims = aShape.filterIndexed { index, l -> index != aDimension }.drop(1).toIntArray()
//    val otherBDims = bShape.filterIndexed { index, l -> index != bDimension }.drop(1).toIntArray()
//
//    val aRank = if (aShape.size == 2) {
//        A = SD.expandDims(A, 1)
//        aDimension++
//        aShape.size + 1
//    } else aShape.size
//
//    val bRank = if (bShape.size == 2) {
//        B = SD.expandDims(B, 2)
//        bShape.size + 1
//    } else bShape.size
//
//
//    if (aDimension != aRank - 1)
//        A = A.permute(*(0 until aRank).minus(aDimension).toIntArray(), aDimension)
//
//    if (bDimension != 1)
//        B = B.permute(0, bDimension, *(1 until bRank).minus(bDimension).toIntArray())
//
//    val aOthers = aShape.reduce { acc, l -> acc * l } / (aBatch * sizeA)
//    val bOthers = bShape.reduce { acc, l -> acc * l } / (bBatch * sizeB)
//
//    if (aRank > 2)
//        A = A.reshape(-1, aOthers, sizeA)
//    if (bRank > 2)
//        B = B.reshape(-1, sizeB, bOthers)
//
//    var result = mmul(A, B)
//
//    result = result.reshape(-1, *otherADims, *otherBDims)
//
//    if (aShape.size == 2)
//        result = SD.squeeze(result, 1)
//    else if (bShape.size == 2)
//        result = SD.squeeze(result, -1)
//
//    return result
//}


//TODO optimize.  it is really close to x / scale
fun SameDiffLambdaDef.squash(x: SDVariable): SDVariable {
    val squaredNorm = SD.squaredNorm(x)
    val scale = SD.math.sqrt(squaredNorm + 1e-7)
    return x * scale / (squaredNorm + 1)
}

class CapsuleLayer(
    val inputNumCaps: Int,
    val inputCapDims: Int,
    val capsules: Int,
    val capsuleDimensions: Int,
    val routings: Int,
    name: String = "capsules", layerBuilder: CapsuleLayer.() -> Unit = {}
) : SameDiffLayer(name) {

    class ConfigHolder(
        val capsules: Int,
        val capsuleDimensions: Int,
        val routings: Int,
        val name: String = "capsules", val layerBuilder: CapsuleLayer.() -> Unit = {}
    ) {
        fun build(inputNumCaps: Int, inputCapDims: Int) =
            CapsuleLayer(
                inputNumCaps, inputCapDims,
                capsules, capsuleDimensions, routings,
                name, layerBuilder
            )
    }

    companion object {
        private const val WEIGHT_KEY = "weights"
        private const val BIAS_KEY = "bias"

        operator fun invoke(
            capsules: Int,
            capsuleDimensions: Int,
            routings: Int,
            name: String = "capsules", layerBuilder: CapsuleLayer.() -> Unit = {}
        ) = ConfigHolder(capsules, capsuleDimensions, routings, name, layerBuilder)

    }

    override val weightParams = mapOf(
        WEIGHT_KEY to listOf(
            1, inputNumCaps, capsules * capsuleDimensions, inputCapDims, 1
            //inputNumCaps, inputCapDims, channels * capsuleDimensions
        )
    )

    override val biasParams = mapOf(
        BIAS_KEY to listOf(1, 1, capsules, capsuleDimensions, 1)
    )

    override fun initializeParams(params: MutableMap<String, INDArray>) {
        val weights by params
        params[WEIGHT_KEY] = weights.assign(0) //TODO initialize w/ property
        val bias by params
        params[BIAS_KEY] = bias.assign(0)
    }

    override fun SameDiffDef.defineLayer(input: SDVariable): SDVariable {

        val miniBatch = input.shape[0].toInt()

        // [mb, inputNumCaps, inputCapDims]

        val expanded = SD.expandDims(SD.expandDims(input, 2), 4) // [mb, inputNumCaps, 1, inputCapDims, 1]
        val tiled = SD.tile(
            expanded,
            intArrayOf(1, 1, capsules * capsuleDimensions, 1, 1)
        ) // [mb, inputNumCaps, capsules  * capsuleDimensions, inputCapDims, 1]

        val weights by params
        val bias by params

        var uHat = SD.sum(weights * tiled, true, 3)
            .reshape(
                miniBatch,
                inputNumCaps,
                capsules,
                capsuleDimensions,
                1
            )// [mb, inputNumCaps, capsules, capsuleDimensions, 1]


        var b = SD.zero("b", miniBatch, inputNumCaps, capsules, 1, 1)

        lateinit var v: SDVariable
        for (i in 0 until routings) { //TODO do I want backprop on inner iterations?  can I freeze it?

            val permuteForSoftmax = intArrayOf(2, 0, 1, 3, 4)

            val c = SD.nn.softmax(b.permute(*permuteForSoftmax))
                .permute(*ArrayUtil.invertPermutation(*permuteForSoftmax))

            v = squash(SD.sum(c * uHat, true, 1)/* + bias*/) // [mb, 1, capsules, capsuleDimensions, 1]

            val vTiled =
                SD.tile(v, intArrayOf(1, inputNumCaps, 1, 1, 1)) // [mb, inputNumCaps, capsules, capsuleDimensions, 1]

            val uMakesV = SD.sum(uHat * vTiled, true, 3) // [mb, inputNumCaps, capsules, 1, 1]

            b += uMakesV

        }

        return SD.squeeze(SD.squeeze(v, 1), 3)
    }

    override fun outputShape(inputShape: List<Int>) = listOf(capsules, capsuleDimensions)

    init {
        layerBuilder()
    }
}

class PrimaryCapsules(
    val channels: Int,
    val capsuleDimensions: Int,
    val kernelSize: Param2 = 9.p2,
    val stride: Param2 = 2.p2,
    val padding: Param2 = 0.p2
) : IHasLayers {
    override val layers = listOf(
        Convolution2DLayer(
            channels * capsuleDimensions,
            kernelSize,
            stride,
            padding,
            name = "primarycaps_conv"
        ), //TODO optional activation & settings
        SameDiffLambdaLayer(
            "primarycaps_samediff",
            { listOf(it.product() / capsuleDimensions, capsuleDimensions) }) { input ->
            //TODO check shapes
            val shaped = input.reshape(input.shape[0].toInt(), -1, capsuleDimensions)
            squash(shaped)
        }
    )
}

fun CapsuleStrength() = SameDiffLambdaLayer("caps_strength", { listOf(it.first()) }) { input ->
    SD.norm2("length", input, 2)
}


