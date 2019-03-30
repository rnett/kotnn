package com.rnett.knn.models

import com.rnett.knn.P
import com.rnett.knn.Param2
import com.rnett.knn.layers.CapsuleLayer
//import com.rnett.knn.layers.CapsuleLayer
import com.rnett.knn.layers.IHasLayers
import com.rnett.knn.layers.ILayer
import com.rnett.knn.layers.samediff.SameDiffLambdaDef
import com.rnett.knn.layers.samediff.SameDiffLambdaLayer
import com.rnett.knn.layers.sizing.CollateDimensionLayer
import com.rnett.knn.layers.sizing.CroppingLayer
import com.rnett.knn.layers.sizing.DistributeDimensionLayer
import com.rnett.knn.layers.sizing.PermuteLayer
import com.rnett.knn.p2
import org.nd4j.autodiff.samediff.SDVariable

operator fun Int.times(dimension: Tensor.Dimension) = this * dimension.size
operator fun Int.div(dimension: Tensor.Dimension) = this / dimension.size

class Tensor internal constructor(vertex: GraphVertex) {


    internal constructor(layer: ILayer, inputs: List<Tensor>) : this(LayerVertex(layer, inputs.map { it.vertex }))
    internal constructor(vertex: WrapperVertex<*>, inputs: List<Tensor>) : this(
        VertexWrapper(
            vertex,
            inputs.map { it.vertex })
    )

    internal constructor(vertex: CustomVertex, inputs: List<Tensor>) : this(vertex.create(inputs.map { it.vertex }))

    internal constructor(layer: ILayer, vararg inputs: Tensor) : this(layer, inputs.toList())
    internal constructor(vertex: WrapperVertex<*>, vararg inputs: Tensor) : this(vertex, inputs.toList())
    internal constructor(vertex: CustomVertex, vararg inputs: Tensor) : this(vertex, inputs.toList())

    var vertex: GraphVertex = vertex
        internal set

    val shape get() = vertex.outputShape
    val rank get() = shape.size

    @GraphBuilderDSL
    fun dup() = Tensor(vertex)

    // index/dimension stuff

    internal val dimNames = mutableMapOf<String, Int>()

    val dimensionMap get() = dimNames.toMap()
    val dimensions get() = dimensionMap.toList().sortedBy { it.second }.map { it.first }

    inner class Dimension internal constructor(val name: String) {
        val index
            get() = run {
                val ind = dimNames[name]
                    ?: throw IllegalArgumentException("No dimension with name $name.  Use Tensor.define() to name dimensions.")

                if (ind >= rank)
                    throw IllegalArgumentException("Can not get dimension $name with index $ind for a tensor of rank $rank")

                ind
            }

        init {
            if (index >= rank)
                throw IllegalArgumentException("Can not get dimension $name with index $index for a tensor of rank $rank")

        }

        fun crop(cropping: Param2, name: String = "cropping") = this@Tensor.crop(index, cropping, name)

        fun crop(before: Int, after: Int, name: String = "cropping") = this@Tensor.crop(index, P(before, after), name)

        val size get() = this@Tensor.shape[index]

        operator fun times(other: Dimension) = size * other.size
        operator fun times(other: Int) = size * other

        operator fun div(other: Dimension) = size / other.size
        operator fun div(other: Int) = size / other

    }

    fun reversePermutation(vararg dimension: Dimension) =
        reversePermutation(*dimension.map { it.index }.toIntArray()).map { this[it] }

    fun reversePermutation(vararg dimension: String) =
        reversePermutation(*dimension.map { this[it].index }.toIntArray()).map { this[it] }

    @GraphBuilderDSL
    fun define(vararg names: String): List<Dimension> {
        if (names.size != rank)
            throw IllegalArgumentException(
                "Number of names given must equal the rank of the tensor.  " +
                        "Got ${names.size} names for a tensor of rank ${rank}"
            )

        dimNames.clear()
        return names.mapIndexed { i, s ->
            dimNames[s] = i
            Dimension(s)
        }
    }

    operator fun get(name: String) = if (name in dimNames) Dimension(name) else
        throw IllegalArgumentException("Dimension $name not defined.  Use Tensor.define() to name dimensions.")

    operator fun get(index: Int) = dimNames.entries
        .firstOrNull { it.value == index }?.let { Dimension(it.key) }
        ?: throw IllegalArgumentException("Dimension with index $index not defined.  Use Tensor.define() to name dimensions.")

    operator fun component1() =
        this[0]

    operator fun component2() =
        this[1]

    operator fun component3() =
        this[2]

    operator fun component4() =
        this[3]

    operator fun component5() =
        this[4]

    operator fun component6() =
        this[5]


    // updating/adding vertices

    @GraphBuilderDSL
    fun updateVertex(update: (GraphVertex) -> GraphVertex) {
        vertex = update(vertex)
    }

    @GraphBuilderDSL
    fun updateVertex(update: IHasLayers, vararg otherInputs: GraphVertex = emptyArray()) {
        update.layers.forEach {
            updateVertex { vertex ->
                LayerVertex(it, listOf(vertex) + otherInputs)
            }
        }
    }

    @GraphBuilderDSL
    fun updateVertex(update: WrapperVertex<*>, vararg otherInputs: GraphVertex = emptyArray()) {
        updateVertex {
            VertexWrapper(update, listOf(it) + otherInputs)
        }
    }

    @GraphBuilderDSL
    fun updateVertex(update: CustomVertex, vararg otherInputs: GraphVertex = emptyArray()) {
        updateVertex {
            update.create(listOf(it) + otherInputs)
        }
    }


    @GraphBuilderDSL
    operator fun IHasLayers.unaryPlus() = updateVertex(this)

    @GraphBuilderDSL
    operator fun WrapperVertex<*>.unaryPlus() = updateVertex(this)

    @GraphBuilderDSL
    operator fun CustomVertex.unaryPlus() = updateVertex(this)


    @GraphBuilderDSL
    operator fun get(vararg layers: IHasLayers) = layers.forEach { +it }

    @GraphBuilderDSL
    operator fun get(vararg vertices: WrapperVertex<*>) = vertices.forEach { +it }

    @GraphBuilderDSL
    operator fun get(vararg vertices: CustomVertex) = vertices.forEach { +it }

    @GraphBuilderDSL
    operator fun invoke(body: Tensor.() -> Unit) {
        this.apply(body)
    }

    //TODO extend
    @GraphBuilderDSL
    fun addInParallel(vararg layers: IHasLayers) = layers.map { this.dup().updateVertex(it) }

    // shape operations

    @GraphBuilderDSL
    fun reshape(vararg shape: Int, name: String = "reshape") = +ReshapeVertex(*shape, name = name)

    @GraphBuilderDSL
    fun reshape(vararg shape: Pair<String, Int>, name: String = "reshape"): List<Dimension> {
        +ReshapeVertex(*shape.map { it.second }.toIntArray(), name = name)
        return define(*shape.map { it.first }.toTypedArray())
    }

    @GraphBuilderDSL
    fun flatten(
        reshapeOrder: ReshapeVertex.ReshapeOrder = ReshapeVertex.ReshapeOrder.RowMajor,
        name: String = "flatten"
    ) = +FlattenVertex(reshapeOrder, name)

    @GraphBuilderDSL
    fun permute(vararg shape: Int, name: String = "permute") = +PermuteLayer(*shape, name = name)

    @GraphBuilderDSL
    fun permute(vararg shape: String, name: String = "permute") = +PermuteLayer(
        *shape.map {
            this[it].index
        }.toIntArray(),
        name = name
    )

    @GraphBuilderDSL
    fun permute(vararg shape: Dimension, name: String = "permute") =
        +PermuteLayer(*shape.map { it.index }.toIntArray(), name = name)

    @GraphBuilderDSL
    fun cropIndexed(vararg croppings: Pair<Int, Param2>, name: String = "cropping") {
        val map = croppings.toMap()
        +CroppingLayer((0 until rank).map { if (it in map) map.getValue(it) else 0.p2 }, name)
    }

    @GraphBuilderDSL
    fun cropNamed(vararg croppings: Pair<String, Param2>, name: String = "cropping") {
        val map = croppings.toMap().mapKeys {
            this[it.key].index
        }
        cropIndexed(*map.toList().toTypedArray(), name = name)
    }

    @GraphBuilderDSL
    fun crop(vararg croppings: Pair<Dimension, Param2>, name: String = "cropping") {
        val map = croppings.toMap().mapKeys {
            it.key.index
        }
        cropIndexed(*map.toList().toTypedArray(), name = name)
    }

    @GraphBuilderDSL
    fun crop(dimension: Int, cropping: Param2, name: String = "cropping") =
        cropIndexed(dimension to cropping, name = name)

    @GraphBuilderDSL
    fun crop(dimension: String, cropping: Param2, name: String = "cropping") =
        cropNamed(dimension to cropping, name = name)

    @GraphBuilderDSL
    fun crop(dimension: Dimension, cropping: Param2, name: String = "cropping") =
        crop(dimension to cropping, name = name)

    @GraphBuilderDSL
    fun crop(dimension: Int, before: Int, after: Int, name: String = "cropping") =
        cropIndexed(dimension to P(before, after), name = name)

    @GraphBuilderDSL
    fun crop(dimension: String, before: Int, after: Int, name: String = "cropping") =
        cropNamed(dimension to P(before, after), name = name)

    @GraphBuilderDSL
    fun crop(dimension: Dimension, before: Int, after: Int, name: String = "cropping") =
        crop(dimension to P(before, after), name = name)

    @GraphBuilderDSL
    fun distribute(dimension: Int, name: String = "distribute", body: Tensor.() -> Unit) {
        val size = shape[dimension]
        +DistributeDimensionLayer(dimension, name = name + "_distribute")
        body()
        +CollateDimensionLayer(dimension, size, name + "_collate")
    }

    // samediff lambdas

    fun samediff(
        outputShape: (List<Int>) -> List<Int> = { it },
        name: String = "lambda",
        lambda: SameDiffLambdaDef.(SDVariable) -> SDVariable
    ) =
        +SameDiffLambdaLayer(name, outputShape, lambda)

    //TODO multiple input samediffs (needs implementation)

    // merging inplace

    operator fun plusAssign(other: Tensor) = updateVertex(AddVertex(), other.vertex)
    operator fun minusAssign(other: Tensor) = updateVertex(SubtractVertex(), other.vertex)
    operator fun timesAssign(other: Tensor) = updateVertex(MultiplyVertex(), other.vertex)

    @GraphBuilderDSL
    infix fun inplaceAverage(other: Tensor) = updateVertex(AverageVertex(), other.vertex)

    @GraphBuilderDSL
    infix fun inplaceMax(other: Tensor) = updateVertex(MaxVertex(), other.vertex)

    operator fun remAssign(other: Tensor) = inplaceConcat(other)
    operator fun remAssign(other: Pair<Int, Tensor>) = inplaceConcat(other.first, other.second)

    @GraphBuilderDSL
    infix fun inplaceConcat(other: Tensor) {
        updateVertex(MergeVertex(), other.vertex)
    }

    @GraphBuilderDSL
    fun inplaceConcat(dimension: Int, other: Tensor) {
        val dims = (0 until rank).toList()
        permute(dimension, *dims.minus(dimension).toIntArray())
        val o = other.dup().apply { permute(dimension, *dims.minus(dimension).toIntArray()) }
        inplaceConcat(o)
        permute(*reversePermutation(dimension, *dims.minus(dimension).toIntArray()))
    }

    @GraphBuilderDSL
    fun inplaceConcat(dimension: Dimension, other: Tensor) = inplaceConcat(dimension.index, other)

    @GraphBuilderDSL
    fun inplaceConcat(dimension: String, other: Tensor) = inplaceConcat(this[dimension].index, other)

    operator fun plusAssign(others: List<Tensor>) = updateVertex(AddVertex(), *others.map { it.vertex }.toTypedArray())

    operator fun minusAssign(others: List<Tensor>) =
        updateVertex(SubtractVertex(), *others.map { it.vertex }.toTypedArray())

    operator fun timesAssign(others: List<Tensor>) =
        updateVertex(MultiplyVertex(), *others.map { it.vertex }.toTypedArray())

    @GraphBuilderDSL
    infix fun inplaceAverage(others: List<Tensor>) =
        updateVertex(AverageVertex(), *others.map { it.vertex }.toTypedArray())

    @GraphBuilderDSL
    infix fun inplaceMax(others: List<Tensor>) = updateVertex(MaxVertex(), *others.map { it.vertex }.toTypedArray())

    operator fun remAssign(others: List<Tensor>) = inplaceConcat(others)

    @GraphBuilderDSL
    infix fun inplaceConcat(others: List<Tensor>) {
        updateVertex(MergeVertex(), *others.map { it.vertex }.toTypedArray())
    }

    @GraphBuilderDSL
    fun inplaceConcat(dimension: Int, others: List<Tensor>) {
        val dims = (0 until rank).toList()
        permute(dimension, *dims.minus(dimension).toIntArray())
        val o = others.map { it.dup().apply { permute(dimension, *dims.minus(dimension).toIntArray()) } }
        inplaceConcat(o)
        permute(*reversePermutation(dimension, *dims.minus(dimension).toIntArray()))
    }

    @GraphBuilderDSL
    fun inplaceConcat(dimension: Dimension, others: List<Tensor>) = inplaceConcat(dimension.index, others)

    @GraphBuilderDSL
    fun inplaceConcat(dimension: String, others: List<Tensor>) = inplaceConcat(this[dimension].index, others)

    // merging not in place

    operator fun plus(other: Tensor) = Tensor(AddVertex(), this, other)
    operator fun minus(other: Tensor) = Tensor(SubtractVertex(), this, other)
    operator fun times(other: Tensor) = Tensor(MultiplyVertex(), this, other)

    @GraphBuilderDSL
    infix fun average(other: Tensor) = Tensor(AverageVertex(), this, other)

    @GraphBuilderDSL
    infix fun max(other: Tensor) = Tensor(AverageVertex(), this, other)


    @GraphBuilderDSL
    infix fun concat(other: Tensor) = Tensor(MergeVertex(), this, other)

    @GraphBuilderDSL
    fun concat(dimension: Int, other: Tensor) = dup().apply {
        inplaceConcat(dimension, other)
    }

    @GraphBuilderDSL
    fun concat(dimension: Dimension, other: Tensor) = dup().apply {
        inplaceConcat(dimension, other)
    }

    @GraphBuilderDSL
    fun concat(dimension: String, other: Tensor) = dup().apply {
        inplaceConcat(dimension, other)
    }

    operator fun rem(other: Tensor) = concat(other)
    operator fun rem(other: Pair<Int, Tensor>) = concat(other.first, other.second)


    operator fun plus(others: List<Tensor>) = Tensor(AddVertex(), listOf(this) + others)
    operator fun minus(others: List<Tensor>) = Tensor(SubtractVertex(), listOf(this) + others)
    operator fun times(others: List<Tensor>) = Tensor(MultiplyVertex(), listOf(this) + others)

    @GraphBuilderDSL
    infix fun average(others: List<Tensor>) = Tensor(AverageVertex(), listOf(this) + others)

    @GraphBuilderDSL
    infix fun max(others: List<Tensor>) = Tensor(AverageVertex(), listOf(this) + others)


    @GraphBuilderDSL
    infix fun concat(others: List<Tensor>) = Tensor(MergeVertex(), this, *others.toTypedArray())

    @GraphBuilderDSL
    fun concat(dimension: Int, others: List<Tensor>) = dup().apply {
        inplaceConcat(dimension, others)
    }

    @GraphBuilderDSL
    fun concat(dimension: Dimension, others: List<Tensor>) = dup().apply {
        inplaceConcat(dimension, others)
    }

    @GraphBuilderDSL
    fun concat(dimension: String, others: List<Tensor>) = dup().apply {
        inplaceConcat(dimension, others)
    }

    operator fun rem(others: List<Tensor>) = concat(others)


    operator fun CapsuleLayer.ConfigHolder.unaryPlus() = +this.build(shape[0], shape[1])


}