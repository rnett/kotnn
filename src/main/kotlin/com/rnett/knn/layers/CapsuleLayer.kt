package com.rnett.knn.layers

import com.rnett.knn.Param2
import com.rnett.knn.layers.convolutional.Convolution2DLayer
import com.rnett.knn.layers.samediff.SameDiffDef
import com.rnett.knn.layers.samediff.SameDiffLambdaLayer
import com.rnett.knn.layers.samediff.SameDiffLayer
import com.rnett.knn.layers.util.product
import com.rnett.knn.models.Tensor
import com.rnett.knn.p2
import org.deeplearning4j.nn.weights.WeightInitUtil
import org.nd4j.autodiff.samediff.SDIndex
import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.util.ArrayUtil


//TODO optimize.  it is really close to x / scale
//TODO check that this works with the dimensions/shapes its given
fun SameDiff.squash(x: SDVariable, dim: Int): SDVariable {
    val squaredNorm = math.square(x).sum(true, dim)//squaredNorm(x, true, dim)
    val scale = math.sqrt(squaredNorm + 1e-5)
    return x * squaredNorm / (scale * (squaredNorm + 1.0))
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

    //TODO implement bias
//    override val biasParams = mapOf(
//        BIAS_KEY to listOf(1, 1, capsules, capsuleDimensions, 1)
//    )

    override fun initializeParams(params: MutableMap<String, INDArray>) {
        WeightInitUtil.initWeights(
            (inputNumCaps * inputCapDims).toDouble(),
            (capsules * capsuleDimensions).toDouble(),
            longArrayOf(1, inputNumCaps.toLong(), (capsules * capsuleDimensions).toLong(), inputCapDims.toLong(), 1),
            this.weightInit,
            null,
            'c',
            params[WEIGHT_KEY]
        )
//        val bias by params
//        params[BIAS_KEY] = bias.assign(0)
    }

    override fun SameDiffDef.defineLayer(input: SDVariable): SDVariable {

        SD.enableDebugMode()

        // [mb, inputNumCaps, inputCapDims]

        val expanded = SD.expandDims(SD.expandDims(input, 2), 4) // [mb, inputNumCaps, 1, inputCapDims, 1]
        val tiled = SD.tile(
            expanded,
            1, 1, capsules * capsuleDimensions, 1, 1
        ) // [mb, inputNumCaps, capsules  * capsuleDimensions, inputCapDims, 1]

        val weights by params
        //val bias by params

        val uHat = (weights * tiled).sum(true, 3)
            .reshape(
                -1,
                inputNumCaps,
                capsules,
                capsuleDimensions,
                1
            )// [mb, inputNumCaps, capsules, capsuleDimensions, 1]


        var b =
            SD.zerosLike(uHat)[SDIndex.all(), SDIndex.all(), SDIndex.all(), SDIndex.interval(0, 1), SDIndex.interval(
                0,
                1
            )]

        //SD.

        val permuteForSoftmax = intArrayOf(2, 0, 1, 3, 4)

//        val loopCounter = SD.zero("loopCounter", 1)
//
//        val loopV = SD.zero("v", miniBatch, 1, capsules, capsuleDimensions, 1)
//
//        val loopOut = SD.whileStatement(
//            { SD, body, inputVars ->
//                inputVars[0].lt(routings.toDouble())
//            },
//            { SD, inputs, params ->
//                params
//            },
//            { SD, inputs, params ->
//                val loopB = params[2]
//                val loopUHat = params[3]
//
//                val c = SD.nn.softmax(loopB.permute(*permuteForSoftmax))
//                    .permute(*ArrayUtil.invertPermutation(*permuteForSoftmax))
//
//                val v = SD.squash((c * loopUHat).sum(true, 1))
//
//                val vTiled = SD.tile(v, intArrayOf(1, inputNumCaps, 1, 1, 1))
//
//                arrayOf(params[0], loopB + (loopUHat * vTiled).sum(true, 3), v)
//            },
//            arrayOf(loopCounter + 1, loopV, b, uHat)
//        ).inputVars[1]
//        return SD.squeeze(SD.squeeze(loopOut, 1), 3)

        lateinit var v: SDVariable
        for (i in 0 until routings) { //TODO do I want backprop on inner iterations?  can I freeze it?

            val c = SD.nn.softmax(b.permute(*permuteForSoftmax))
                .permute(*ArrayUtil.invertPermutation(*permuteForSoftmax))

            val pre = (c * uHat).sum(true, 1)
            v = SD.squash(pre, 3) // [mb, 1, capsules, capsuleDimensions, 1]


            if (i == routings - 1)
                break

            val vTiled = SD.tile(v, 1, inputNumCaps, 1, 1, 1)
            // [mb, inputNumCaps, capsules, capsuleDimensions, 1]

            // [mb, inputNumCaps, capsules, 1, 1]
            b += (uHat * vTiled).sum(true, 3)
        }

        return SD.squeeze(SD.squeeze(v, 1), 3)
    }

    override fun outputShape(inputShape: List<Int>) = listOf(capsules, capsuleDimensions)

    init {
        layerBuilder()
    }
}

fun Tensor.PrimaryCapsules(
    channels: Int,
    capsuleDimensions: Int,
    kernelSize: Param2 = 9.p2,
    stride: Param2 = 2.p2,
    padding: Param2 = 0.p2
): SameDiffLayer {
    +Convolution2DLayer(
        channels * capsuleDimensions,
        kernelSize,
        stride,
        padding,
        name = "primarycaps_conv"
    )
    val capsules = shape.product() / capsuleDimensions
    println(capsuleDimensions)
    return SameDiffLambdaLayer(
        "primarycaps_samediff",
        { listOf(it.product() / capsuleDimensions, capsuleDimensions) }) { input ->
        //TODO check shapes
        val shaped = input.reshape(
            -1,
            capsules, //nessecary for variable minibatch
            capsuleDimensions
        )
        SD.squash(shaped, 2)
    }
}

fun CapsuleStrength() = SameDiffLambdaLayer("caps_strength", { listOf(it.first()) }) { input ->
    SD.norm2("length", input, 2)
}


