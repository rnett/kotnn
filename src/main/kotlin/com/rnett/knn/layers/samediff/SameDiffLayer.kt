package com.rnett.knn.layers.samediff

import com.rnett.knn.layers.Layer
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.learning.config.IUpdater
import org.deeplearning4j.nn.conf.layers.samediff.AbstractSameDiffLayer as DL4JAbstractSameDiffLayer
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayer as DL4JSameDiffLayer

abstract class AbstractSameDiffLayer<T : DL4JAbstractSameDiffLayer, S : AbstractSameDiffLayer<T, S>>(name: String = "samediff") :
    Layer<T, S>(name) {

    var l1: Double? = null
    var l2: Double? = null
    var l1Bias: Double? = null
    var l2Bias: Double? = null

    var updater: IUpdater? = null
    var biasUpdater: IUpdater? = null

    override fun apply(builder: org.deeplearning4j.nn.conf.layers.Layer.Builder<*>) {
        super.apply(builder)
        builder as DL4JAbstractSameDiffLayer.Builder



        builder.l1(l1 ?: Double.NaN)
        builder.l2(l2 ?: Double.NaN)
        builder.l1Bias(l1Bias ?: Double.NaN)
        builder.l2Bias(l2Bias ?: Double.NaN)

        builder.updater(updater)
        builder.biasUpdater(biasUpdater)
    }
}

abstract class SameDiffLayer(
    name: String = "samediff",
    layerBuilder: SameDiffLayer.() -> Unit = {}
) :
    AbstractSameDiffLayer<DL4JSameDiffLayer, SameDiffLayer>(name) {

    var weightInit = WeightInit.XAVIER

    open val weightParams: Map<String, List<Int>> = emptyMap()
    open val biasParams: Map<String, List<Int>> = emptyMap()

    open fun initializeParams(params: Map<String, INDArray>) {}

    abstract fun SameDiffDef.defineLayer(input: SDVariable): SDVariable

    override fun builder() = KNNSameDiffLayerBuilder(this)

    override fun apply(builder: org.deeplearning4j.nn.conf.layers.Layer.Builder<*>) {
        super.apply(builder)

        builder as KNNSameDiffLayerBuilder
        builder.weightInit(weightInit)
    }

    companion object {
        operator fun invoke(
            defineLayer: SameDiffDef.(input: SDVariable) -> SDVariable,
            weightParams: Map<String, List<Int>> = emptyMap(),
            biasParams: Map<String, List<Int>> = emptyMap(),
            outputShape: (List<Int>) -> List<Int> = { it },
            initializeParams: (Map<String, INDArray>) -> Unit = {},
            name: String = "samediff",
            layerBuilder: SameDiffLayer.() -> Unit = {}
        ) = object : SameDiffLayer(name, layerBuilder) {
            override fun SameDiffDef.defineLayer(input: SDVariable) = defineLayer(input)
            override fun outputShape(inputShape: List<Int>) = outputShape(inputShape)

            override val weightParams = weightParams
            override val biasParams = biasParams

            override fun initializeParams(params: Map<String, INDArray>) = initializeParams(params)
        }
    }

    init {
        layerBuilder()
    }
}

fun SameDiffLambdaLayer(
    name: String = "lambda",
    outputShape: (List<Int>) -> List<Int>,
    defineLayer: SameDiffLambdaDef.(input: SDVariable) -> SDVariable
) =
    object : SameDiffLayer(name, {}) {
        override fun SameDiffDef.defineLayer(input: SDVariable) = defineLayer(input)

        override fun outputShape(inputShape: List<Int>) = outputShape(inputShape)

    }

fun SameDiffLambdaLayer(
    name: String = "lambda",
    defineLayer: SameDiffLambdaDef.(input: SDVariable) -> SDVariable
) =
    SameDiffLambdaLayer(name, { it }, defineLayer)