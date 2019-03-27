package com.rnett.knn.layers

import org.deeplearning4j.nn.api.layers.LayerConstraint
import org.deeplearning4j.nn.conf.dropout.Dropout
import org.deeplearning4j.nn.conf.dropout.IDropout
import org.deeplearning4j.nn.conf.inputs.InputType
import org.nd4j.linalg.factory.Nd4j
import javax.naming.OperationNotSupportedException
import org.deeplearning4j.nn.conf.layers.Layer as DL4JLayer

interface IHasLayers {
    val layers: List<ILayer>

    fun buildAll(): List<DL4JLayer> = layers.map { it.buildLayer() }

    fun outputShape(inputShape: List<Int>): List<Int> {
        var shape = inputShape
        for (layer in layers) {
            shape = layer.outputShape(shape)
        }
        return shape
    }
}

interface ILayer : IHasLayers {
    override fun buildAll() = listOf(buildLayer())
    fun buildLayer(): DL4JLayer

    override fun outputShape(inputShape: List<Int>): List<Int>

    val name: String

    override val layers: List<ILayer>
        get() = listOf(this)
}

class DL4JWrapper<out E : DL4JLayer>(val layer: E) : ILayer {

    override fun buildLayer() = layer

    override fun outputShape(inputShape: List<Int>): List<Int> = layer.getOutputType(
        0,
        inputShape.inferInputType()
    ).shape.toList().map { it.toInt() }

    override val name get() = layer.layerName
}

fun List<Int>.NDArrayOfShape() = Nd4j.ones(*this.map {
    if (it <= 0)
        1.toLong()
    else
        it.toLong()
}.toLongArray())

fun List<Int>.inferInputType(addBatch: Boolean = true): InputType{
    val list = (if(addBatch)
        listOf(-1)
    else
        listOf()) + this
    return InputType.inferInputType(list.NDArrayOfShape())
}

fun <E : DL4JLayer> wrapper(layer: E) = DL4JWrapper(layer)
fun <T : DL4JLayer.Builder<T>, E : DL4JLayer> wrapper(builder: T) = DL4JWrapper<E>(builder.build())

interface IKNNLayer<out T : DL4JLayer> : ILayer {
    override fun buildLayer() = build()

    fun build(): T = builder().also { apply(it) }.build() as T
    fun builder(): DL4JLayer.Builder<*>
    fun apply(builder: DL4JLayer.Builder<*>)
}

private val nameCount = mutableMapOf<String, Int>().withDefault { 0 }

internal fun makeNameUnique(name: String) =
    run {
        if (name in nameCount)
            "${name}_${nameCount[name]!!}"
        else
            name
    }.also {
        nameCount[name] = nameCount.getOrDefault(name, 0) + 1
    }

abstract class Layer<out T : DL4JLayer, S : Layer<T, S>>(
    givenName: String
) : IKNNLayer<T> {

    private var trueName: String = run {
        if (givenName in nameCount)
            "${givenName}_${nameCount[givenName]!!}"
        else
            givenName
    }.also {
        nameCount[givenName] = nameCount.getOrDefault(givenName, 0) + 1
    }

    override var name: String
        get() = trueName
        set(givenName) {
            trueName = makeNameUnique(givenName)
        }

    var dropout: IDropout? = null

    var paramConstraints: MutableList<LayerConstraint> = mutableListOf()
    var weightConstraints: MutableList<LayerConstraint> = mutableListOf()
    var biasConstraints: MutableList<LayerConstraint> = mutableListOf()

    var dropoutProbability: Double
        get() = throw OperationNotSupportedException("Can't get value")
        set(v) {
            dropout = Dropout(v)
        }

    override fun apply(builder: DL4JLayer.Builder<*>) {
        builder.name(name)
        builder.dropOut(dropout)
        builder.constrainAllParameters(*paramConstraints.toTypedArray())
        builder.constrainWeights(*weightConstraints.toTypedArray())
        builder.constrainBias(*biasConstraints.toTypedArray())
    }

    operator fun invoke(builder: S.() -> Unit): S {
        builder(this as S)
        return this
    }

}