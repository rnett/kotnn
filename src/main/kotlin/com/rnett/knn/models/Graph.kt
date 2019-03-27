package com.rnett.knn.models

import com.rnett.knn.components.Optimizers
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.nd4j.linalg.learning.config.IUpdater
import org.deeplearning4j.nn.graph.ComputationGraph as DL4JComputationGraph


@DslMarker
annotation class GraphBuilderDSL

fun reversePermutation(vararg dimension: Int): IntArray {
    if (dimension.sorted() != (0 until dimension.size).toList())
        throw IllegalArgumentException("${dimension.toList()} is not a valid permutation order, elements must be all of 0 .. ${dimension.size - 1}")

    if (dimension.size <= 1)
        return dimension
    else if (dimension.size == 2)
        return dimension

    val new = MutableList(dimension.size) { 0 }
    dimension.forEachIndexed { index, x ->
        new[x] = index
    }
    return new.toIntArray()
}



@GraphBuilderDSL
fun graph(body: ComputationGraph.() -> Unit) = ComputationGraph(body)

class ComputationGraph() {

    constructor(body: ComputationGraph.() -> Unit) : this() {
        invoke(body)
    }

    //TODO graph vars
    var optimizer: IUpdater = Optimizers.SGD

    private val inputs = mutableSetOf<Input>()

    val outputs: MutableList<Tensor> = mutableListOf()

    @GraphBuilderDSL
    fun outputs(vararg outputs: Tensor) {
        this.outputs.addAll(outputs.toList())
    }

    @GraphBuilderDSL
    operator fun Tensor.not() = outputs(this)

    @GraphBuilderDSL
    fun input(name: String, shape: List<Int>) = Input(name, shape).let {
        inputs.add(it)
        Tensor(it)
    }

    @GraphBuilderDSL
    fun input(name: String, vararg shape: Int) = input(name, shape.toList())

    @GraphBuilderDSL
    fun input(name: String, vararg shape: Pair<String, Int>) =
        input(name, shape.toList().map { it.second }).also {
            it.define(*shape.map { it.first }.toTypedArray())
        }

    operator fun invoke(body: ComputationGraph.() -> Unit) = this.apply(body)

    fun build(graphConfig: ComputationGraphConfiguration.GraphBuilder.() -> Unit = {}): DL4JComputationGraph {
        val vertices = mutableListOf<GraphVertex>()

        val temp = mutableSetOf(*outputs.map { it.vertex }.toTypedArray())
        val temp2 = mutableSetOf<GraphVertex>()

        do {
            temp2.clear()

            temp.forEach {
                vertices.add(it)
                temp2.addAll((it.inputs ?: emptyList()).filter { it !in vertices })
            }

            temp.clear()
            temp.addAll(temp2)
        } while (temp.isNotEmpty())

        val graphBuilder = NeuralNetConfiguration.Builder()
            .updater(optimizer)
            .graphBuilder()
            .apply(graphConfig)

        vertices.reverse()

        inputs.forEach {
            it.add(graphBuilder)
        }

        (vertices.filter { it !in inputs }).forEach {
            it.add(graphBuilder)
        }

        if (outputs.isEmpty())
            throw IllegalStateException("No outputs defined")
        else
            graphBuilder.setOutputs(*outputs.map { it.vertex.name }.toTypedArray())

        return DL4JComputationGraph(graphBuilder.build())
    }
}