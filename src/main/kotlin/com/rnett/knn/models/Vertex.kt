package com.rnett.knn.models

import com.rnett.knn.enforceLength
import com.rnett.knn.layers.ILayer
import com.rnett.knn.layers.inferInputType
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration
import org.deeplearning4j.nn.conf.InputPreProcessor
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex as DL4JElementWiseVertex
import org.deeplearning4j.nn.conf.graph.GraphVertex as DL4JVertex
import org.deeplearning4j.nn.conf.graph.MergeVertex as DL4JMergeVertex
import org.deeplearning4j.nn.conf.graph.PreprocessorVertex as DL4JPreprocessorVertex
import org.deeplearning4j.nn.conf.graph.ReshapeVertex as DL4JReshapeVertex
import org.deeplearning4j.nn.conf.graph.SubsetVertex as DL4JSubsetVertex

sealed class GraphVertex(open val name: String, open val inputs: List<GraphVertex>?) {
    open fun add(graph: ComputationGraphConfiguration.GraphBuilder) {
        graph.addVertex(name, makeVertex(), *(inputs ?: emptyList()).map { it.name }.toTypedArray())
    }

    protected open fun makeVertex(): DL4JVertex =
        throw NotImplementedError("This vertex does not supply a DL4J GraphVertex")

    val outputShape: List<Int> get() = outputShape((inputs ?: emptyList()).map { it.outputShape })
    abstract fun outputShape(inputShapes: List<List<Int>>): List<Int>
}

data class LayerVertex(val layer: ILayer, override val inputs: List<GraphVertex>) : GraphVertex(layer.name, inputs) {
    override fun add(graph: ComputationGraphConfiguration.GraphBuilder) {
        graph.addLayer(name, layer.buildLayer(), *inputs.map { it.name }.toTypedArray())
    }

    override fun outputShape(inputShapes: List<List<Int>>): List<Int> {
        if (inputShapes.map { it.size }.toSet().size > 1)
            throw IllegalArgumentException("All inputs to layer vertex $name must have the same dimensionality")

        if (inputShapes.map { it.drop(1) }.toSet().size > 1)
            throw IllegalArgumentException("All inputs to layer vertex $name must have the same size in all dimensions except the first")

        val rest = inputShapes.first().drop(1)
        val first = inputShapes.sumBy { it.first() }

        val merged = listOf(first) + rest

        return layer.outputShape(merged)
    }
}

data class Input(override val name: String, val shape: List<Int>) : GraphVertex(name, null) {
    override fun add(graph: ComputationGraphConfiguration.GraphBuilder) {
        graph.addInputs(listOf(name))
        val type = shape.inferInputType()
        graph.setInputTypes(type)
    }

    override fun outputShape(inputShapes: List<List<Int>>) = shape
}

private var vertexCount = 0

data class VertexWrapper(
    val vertex: WrapperVertex<*>,
    override val inputs: List<GraphVertex>
) : GraphVertex(vertex.name, inputs) {
    override fun makeVertex() = vertex.makeVertex(inputs.map {
        it.outputShape
    })

    override fun outputShape(inputShapes: List<List<Int>>) = vertex.outputShape(inputShapes)

}

sealed class WrapperVertex<T : DL4JVertex>(val name: String = "wrapper_${vertexCount++}") {
    abstract fun outputShape(inputShapes: List<List<Int>>): List<Int>
    open fun makeVertex(inputShapes: List<List<Int>>): T = makeVertex()
    abstract fun makeVertex(): T
}

class PreprocessorVertex(val preprocessor: InputPreProcessor, name: String = "preprocessor_${vertexCount++}") :
    WrapperVertex<DL4JPreprocessorVertex>(name) {

    override fun outputShape(inputShapes: List<List<Int>>) =
        preprocessor.getOutputType(inputShapes.first().inferInputType()).shape.map { it.toInt() }

    override fun makeVertex() = DL4JPreprocessorVertex(preprocessor)

}

class MergeVertex(name: String = "merge_${vertexCount++}") :
    WrapperVertex<DL4JMergeVertex>(name) {
    override fun makeVertex() = DL4JMergeVertex()

    override fun outputShape(inputShapes: List<List<Int>>): List<Int> {
        if (inputShapes.map { it.size }.toSet().size > 1)
            throw IllegalArgumentException("All inputs to MergeVertex $name must have the same dimensionality")

        if (inputShapes.map { it.drop(1) }.toSet().size > 1)
            throw IllegalArgumentException("All inputs to MergeVertex $name must have the same size in all dimensions except the first")

        val rest = inputShapes.first().drop(1)
        val first = inputShapes.sumBy { it.first() }

        return listOf(first) + rest
    }
}

/**
 * end is exclusive (dl4j's is not)
 */
class SubsetVertex(
    val start: Int,
    val end: Int,
    name: String = "subset_${vertexCount++}"
) :
    WrapperVertex<DL4JSubsetVertex>(name) {

    override fun makeVertex() = DL4JSubsetVertex(start, end - 1)
    override fun outputShape(inputShapes: List<List<Int>>): List<Int> {
        inputShapes.enforceLength(1, "SubsetVertex $name must have one input")

        val inputShape = inputShapes.first()

        return listOf(end - start) + inputShape.drop(1)
    }
}

class ElementWiseVertex(
    val operation: DL4JElementWiseVertex.Op,
    name: String = "elemtwise_${operation}_${vertexCount++}"
) :
    WrapperVertex<DL4JElementWiseVertex>(name) {
    override fun makeVertex() = DL4JElementWiseVertex(operation)
    override fun outputShape(inputShapes: List<List<Int>>): List<Int> {

        if (inputShapes.toSet().size > 1)
            throw IllegalArgumentException("All inputs to an ElementWiseVertex $name must have the same dimensions")

        if (operation == org.deeplearning4j.nn.conf.graph.ElementWiseVertex.Op.Subtract)
            inputShapes.enforceLength(2, "ElementWiseVertex $name with op Subtract can only have two inputs")

        return inputShapes.first()
    }
}

object AddVertex {
    operator fun invoke(name: String = "elemtwise_Add_${vertexCount++}") =
        ElementWiseVertex(org.deeplearning4j.nn.conf.graph.ElementWiseVertex.Op.Add, name)
}

object SubtractVertex {
    operator fun invoke(name: String = "elemtwise_Subtract_${vertexCount++}") =
        ElementWiseVertex(org.deeplearning4j.nn.conf.graph.ElementWiseVertex.Op.Subtract, name)
}

object MultiplyVertex {
    operator fun invoke(name: String = "elemtwise_Product_${vertexCount++}") =
        ElementWiseVertex(org.deeplearning4j.nn.conf.graph.ElementWiseVertex.Op.Product, name)
}

object AverageVertex {
    operator fun invoke(name: String = "elemtwise_Average_${vertexCount++}") =
        ElementWiseVertex(org.deeplearning4j.nn.conf.graph.ElementWiseVertex.Op.Average, name)
}

object MaxVertex {
    operator fun invoke(name: String = "elemtwise_Max_${vertexCount++}") =
        ElementWiseVertex(org.deeplearning4j.nn.conf.graph.ElementWiseVertex.Op.Max, name)
}

//TODO lambdas using SameDiffVertex

class ReshapeVertex(
    val shape: List<Int>,
    val reshapeOrder: ReshapeOrder = ReshapeOrder.RowMajor,
    name: String = "reshape_${vertexCount++}"
) :
    WrapperVertex<DL4JReshapeVertex>(name) {

    constructor(vararg shape: Int, name: String = "reshape_${vertexCount++}") : this(
        shape.toList(),
        ReshapeOrder.RowMajor,
        name
    )


    enum class ReshapeOrder(val code: Char) {
        RowMajor('c'), ColumnMajor('f');
    }

    override fun makeVertex() = DL4JReshapeVertex(reshapeOrder.code, (listOf(-1) + shape).toIntArray(), null)
    override fun outputShape(inputShapes: List<List<Int>>): List<Int> {
        inputShapes.enforceLength(1, "ReshapeVertex $name requires exactly one input")

        val total = inputShapes.first().reduce { i, acc -> acc * i }
        val newTotal = shape.reduce { i, acc -> acc * i }

        if (total != newTotal)
            throw IllegalArgumentException(
                "ReshapeVertex $name requires the input and output shapes to have the same " +
                        "number of parameters, but the input has $total and the output has $newTotal"
            )

        return shape
    }
}

class FlattenVertex(
    val reshapeOrder: ReshapeVertex.ReshapeOrder = ReshapeVertex.ReshapeOrder.RowMajor,
    name: String = "reshape_${vertexCount++}"
) :
    WrapperVertex<DL4JReshapeVertex>(name) {

    override fun makeVertex() = throw NotImplementedError()

    override fun makeVertex(inputShapes: List<List<Int>>) =
        DL4JReshapeVertex(
            reshapeOrder.code,
            (listOf(-1) + inputShapes.first().reduce { acc, i -> acc * i }).toIntArray(),
            null
        )

    override fun outputShape(inputShapes: List<List<Int>>): List<Int> {
        inputShapes.enforceLength(1, "ReshapeVertex $name requires exactly one input")

        val total = inputShapes.first().reduce { i, acc -> acc * i }
        return listOf(total)
    }
}

interface CustomVertex {
    fun create(inputs: List<GraphVertex>): GraphVertex
}

class AddDimensionVertex(
    val reshapeOrder: ReshapeVertex.ReshapeOrder = ReshapeVertex.ReshapeOrder.RowMajor,
    val name: String = "reshape_${vertexCount++}"
) : CustomVertex {
    override fun create(inputs: List<GraphVertex>) = VertexWrapper(
        ReshapeVertex(
            inputs.first().outputShape + listOf(1),
            reshapeOrder, name
        ), inputs
    )
}


class RemoveDegenerateDimensionVertex(
    val reshapeOrder: ReshapeVertex.ReshapeOrder = ReshapeVertex.ReshapeOrder.RowMajor,
    val name: String = "reshape_${vertexCount++}"
) : CustomVertex {

    override fun create(inputs: List<GraphVertex>): GraphVertex {
        if (inputs.first().outputShape.last() != 1)
            throw IllegalArgumentException(
                "Can't use RemoveDegenerateDimensionVertex $name on " +
                        "input with last dimension != 1 (was ${inputs.first().outputShape.last()})"
            )

        return VertexWrapper(
            ReshapeVertex(inputs.first().outputShape.dropLast(1), reshapeOrder, name)
            , inputs
        )
    }
}