package com.rnett.knn.layers

import com.rnett.knn.Param2
import com.rnett.knn.layers.convolutional.Convolution2DLayer
import com.rnett.knn.layers.samediff.SameDiffDef
import com.rnett.knn.layers.samediff.SameDiffLambdaDef
import com.rnett.knn.layers.samediff.SameDiffLambdaLayer
import com.rnett.knn.layers.samediff.SameDiffLayer
import com.rnett.knn.p2
import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.linalg.api.ndarray.INDArray

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

    companion object {
        private const val WEIGHT_KEY = "weights"
    }

    override val weightParams = mapOf(
        WEIGHT_KEY to listOf(
            capsules, inputNumCaps, capsuleDimensions, inputCapDims
            //inputNumCaps, inputCapDims, capsules * capsuleDimensions
        )
    )

    override fun initializeParams(params: MutableMap<String, INDArray>) {
        val weights by params
        params[WEIGHT_KEY] = weights.assign(0) //TODO initialize w/ property
    }

    override fun SameDiffDef.defineLayer(input: SDVariable): SDVariable {

        val miniBatch = input.shape[0].toInt()

        val expanded = SD.expandDims(input, 1)
        val tiled = SD.tile(expanded, intArrayOf(1, capsules, 1, 1))

        val weights by params

        //TODO I may need to permute the last two dims for tensorMmul to work (cause its W x C-t)

        //TODO can I do with a single mmul?
        val hat = /*SD.stack(0, *SD.unstack(tiled, 0).map {
            SD.tensorMmul(it,weights, arrayOf(intArrayOf(1, 2), intArrayOf(2, 2, 3)) )
        }.toTypedArray())*/
            SD.tensorMmul(tiled, weights, arrayOf(intArrayOf(1, 2), intArrayOf(2, 2, 3)))

        var b = SD.zero("b", miniBatch, capsules, inputNumCaps)

        lateinit var output: SDVariable

        for (i in 0 until routings) {

            b = b.permute(1, 0, 2)
            val c = SD.nn.softmax(b) //TODO only axis 1
            b = b.permute(1, 0, 2)

            output = squash(SD.tensorMmul(c, hat, arrayOf(intArrayOf(1, 2), intArrayOf(2, 2, 3))))


            if (i < routings - 1) {
                b += SD.tensorMmul(output, hat, arrayOf(intArrayOf(1, 2), intArrayOf(2, 2, 3)))
            }

        }

        return output
    }

    override fun outputShape(inputShape: List<Int>) = listOf(capsules, capsuleDimensions)

    init {
        layerBuilder()
    }
}

class PrimaryCapsules(
    val capsules: Int,
    val capsuleDimensions: Int,
    val kernelSize: Param2 = 9.p2,
    val stride: Param2 = 2.p2,
    val padding: Param2 = 0.p2
) : IHasLayers {
    override val layers = listOf(
        Convolution2DLayer(capsules * capsuleDimensions, kernelSize, stride, padding, name = "primarycaps_conv"),
        SameDiffLambdaLayer("primarycaps_samediff", { listOf(capsules, capsuleDimensions) }) { input ->
            //TODO check shapes
            val shaped = input.reshape(input.shape[0].toInt(), -1, capsuleDimensions)
            squash(shaped)
        }
    )
}

fun CapsuleStrength() = SameDiffLambdaLayer("caps_strength", { listOf(it.first()) }) { input ->
    SD.norm2("length", input, 2)
}


