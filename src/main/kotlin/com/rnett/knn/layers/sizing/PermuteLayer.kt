package com.rnett.knn.layers.sizing

import com.rnett.knn.layers.samediff.SameDiffDef
import com.rnett.knn.layers.samediff.SameDiffLayer
import org.nd4j.autodiff.samediff.SDVariable

/**
 * "Pulls" dimensions.  e.g. a 2 in dimensions will move the 2nd dimension in the input to there (0 indexed)
 * @param dimensions The input dimension to put in each place
 */
class PermuteLayer(
    var dimensions: List<Int>, name: String = "permute",
    layerBuilder: PermuteLayer.() -> Unit = {}
) : SameDiffLayer(name) {

    constructor(
        vararg dimensions: Int, name: String = "permute",
        layerBuilder: PermuteLayer.() -> Unit = {}
    ) : this(dimensions.toList(), name, layerBuilder)

    override fun SameDiffDef.defineLayer(input: SDVariable): SDVariable {
        return SD.permute(input, *(listOf(0) + dimensions.map { it + 1 }).toIntArray())
    }

    override fun outputShape(inputShape: List<Int>) = dimensions.map { inputShape[it] }

    init {
        layerBuilder()
    }
}

/**
 * Brings the target dimension to the "front", leaving others unchanged
 */
class BringToFrontLayer(
    var dimension: Int, name: String = "bringdimtofront",
    layerBuilder: BringToFrontLayer.() -> Unit = {}
) : SameDiffLayer(name) {

    override fun SameDiffDef.defineLayer(input: SDVariable): SDVariable {
        val newdims  = (listOf(0, dimension) + (1..input.shape.size-1).filter { it != dimension })
        return SD.permute(input, *newdims.toIntArray())
    }

    override fun outputShape(inputShape: List<Int>) =
        listOf(inputShape[dimension]) + inputShape.filterIndexed { index, i -> index != dimension }

    init {
        layerBuilder()
    }
}