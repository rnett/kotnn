package com.rnett.knn.layers.samediff

import com.rnett.knn.layers.inferInputType
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.samediff.SDLayerParams
import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.linalg.api.ndarray.INDArray

//TODO serialization support
class SameDiffImpl(
    builder: KNNSameDiffLayerBuilder,
    val defineLayer: SameDiffDef.(input: SDVariable) -> SDVariable,
    val weightParams: Map<String, List<Int>> = emptyMap(),
    val biasParams: Map<String, List<Int>> = emptyMap(),
    val initalizeParams: (MutableMap<String, INDArray>) -> Unit,
    val getOutputShape: (List<Int>) -> List<Int>
) : org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayer(builder) {
    override fun defineLayer(
        sameDiff: SameDiff,
        layerInput: SDVariable,
        paramTable: MutableMap<String, SDVariable>
    ): SDVariable {
        return SameDiffDef(sameDiff, paramTable).defineLayer(layerInput)
    }

    override fun defineParameters(params: SDLayerParams) {
        params.clear()
        weightParams.forEach { key, shape ->
            params.addWeightParam(key, *shape.map { it.toLong() }.toLongArray())
        }
        biasParams.forEach { key, shape ->
            params.addBiasParam(key, *shape.map { it.toLong() }.toLongArray())
        }
    }

    override fun getOutputType(layerIndex: Int, inputType: InputType?) =
        getOutputShape(inputType!!.shape.toList().map { it.toInt() }).inferInputType()

    override fun initializeParameters(params: MutableMap<String, INDArray>) {
        initalizeParams(params)
    }
}

class KNNSameDiffLayerBuilder(
    val layer: SameDiffLayer
) :
    org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayer.Builder<KNNSameDiffLayerBuilder>() {

    override fun <E : org.deeplearning4j.nn.conf.layers.Layer?> build(): E {
        return SameDiffImpl(
            this,
            { input ->
                layer.run {
                    defineLayer(input)
                }
            },
            layer.weightParams,
            layer.biasParams,
            layer::initializeParams,
            layer::outputShape
        ) as E
    }
}