package com.rnett.knn.layers.sizing

import com.rnett.knn.layers.samediff.SameDiffDef
import com.rnett.knn.layers.samediff.SameDiffLayer
import org.deeplearning4j.nn.graph.vertex.impl.PreprocessorVertex
import org.nd4j.autodiff.samediff.SDVariable

//TODO check that things end up in the same spots

class DistributeDimensionLayer(
    var dimension: Int, name: String = "distributedimension",
    layerBuilder: DistributeDimensionLayer.() -> Unit = {}
) : SameDiffLayer(name) {

    override fun SameDiffDef.defineLayer(input: SDVariable): SDVariable {

        val shape = input.shape.map { it.toInt() }

        val newOrder = listOf(0, dimension + 1) + (1 until shape.size).filter { it != dimension + 1 }

        val newShape = longArrayOf(shape[0].toLong() * shape[dimension + 1], *shape
            .filterIndexed { index, i -> index != dimension + 1 }
            .map { it.toLong() }.drop(1).toLongArray()
        )

        return SD.reshape(SD.permute(input, *newOrder.toIntArray()), *newShape)
    }

    override fun outputShape(inputShape: List<Int>) =
        (0 until inputShape.size).filter { it != dimension }.map { inputShape[it] }

    init {
        layerBuilder()
    }
}

class CollateDimensionLayer(
    var dimension: Int, var size: Int, name: String = "collatedimension",
    layerBuilder: CollateDimensionLayer.() -> Unit = {}
) : SameDiffLayer(name) {

    override fun SameDiffDef.defineLayer(input: SDVariable): SDVariable {
        val newShape = longArrayOf(
            -1,
            size.toLong(), *input.shape.drop(1).toLongArray()
        )

        val newDims = intArrayOf(
            0,
            *(2..(dimension+1)).toList().toIntArray(),
            1,
            *((dimension+2) until input.shape.size+1).toList().toIntArray()
        )

        return SD.permute(SD.reshape(input, *newShape), *newDims)
    }

    override fun outputShape(inputShape: List<Int>) =
        (0 until dimension).map { inputShape[it] } +
                listOf(size) +
                (dimension until inputShape.size).map { inputShape[it] }

    init {
        layerBuilder()
    }
}