package com.rnett.knn.layers.sizing

import com.rnett.knn.layers.samediff.Index
import com.rnett.knn.layers.samediff.SameDiffDef
import com.rnett.knn.layers.samediff.SameDiffLayer
import org.nd4j.autodiff.samediff.SDVariable

class SubsetLayer(
    var subset: List<Index>,
    name: String = "subset",
    layerBuilder: SubsetLayer.() -> Unit = {}
) : SameDiffLayer(name){

    val dimensions = subset.size

    override fun SameDiffDef.defineLayer(input: SDVariable): SDVariable {
        return input[subset]
    }

    override fun outputShape(inputShape: List<Int>): List<Int> {
        if(inputShape.size != dimensions)
            throw IllegalArgumentException("Dimensions of input (${inputShape.size}) are not equal to dimensions of cropping ($dimensions)")

        return inputShape.zip(subset).map {
            val bounds = it.second.resolve(it.first)
            bounds.second - bounds.first
        }
    }
    init{
        layerBuilder()
    }
}