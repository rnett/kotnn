package com.rnett.knn.models

import com.rnett.knn.layers.IHasLayers
import com.rnett.knn.layers.ILayer
import com.rnett.knn.layers.samediff.SameDiffDef
import com.rnett.knn.layers.samediff.SameDiffLambdaLayer
import com.rnett.knn.layers.sizing.CollateDimensionLayer
import com.rnett.knn.layers.sizing.DistributeDimensionLayer
import com.rnett.knn.layers.wrapper
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.linalg.learning.config.IUpdater
import org.deeplearning4j.nn.conf.graph.GraphVertex as DL4JVertex
import org.deeplearning4j.nn.conf.layers.Layer as DL4JLayer

@DslMarker
annotation class GraphBuilderDSL

fun List<ILayer>.toLayers() = object : IHasLayers {
    override val layers = this@toLayers
}

typealias SameDiffLambda = SameDiffDef.(input: SDVariable) -> SDVariable

class ComplexNetwork() {

    constructor(builder: ComplexNetwork.() -> Unit) : this() {
        invoke(builder)
    }

    //TODO graph vars
    var optimizer: IUpdater = Updater.SGD.iUpdaterWithDefaultConfig

    private var _lastAdded: GraphVertex? = null

    @GraphBuilderDSL
    var currentVertex
        get() = _lastAdded ?: throw IllegalStateException("No vertices have been added")
        set(v){ _lastAdded = v }

    @GraphBuilderDSL
    fun from(vertex: GraphVertex, isolate: Boolean = true, builder: ComplexNetwork.() -> Unit){
        if(isolate){
            val old = currentVertex
            currentVertex = vertex
            builder()
            currentVertex = old
        } else {
            currentVertex = vertex
            builder()
        }
    }

    @GraphBuilderDSL
    infix fun GraphVertex.thenChain(builder: ComplexNetwork.() -> Unit) = from(this, true, builder)

    @GraphBuilderDSL
    infix fun GraphVertex.thenChainNonIsolated(builder: ComplexNetwork.() -> Unit) = from(this, false, builder)

    private val inputs = mutableSetOf<Input>()

    private val vertices = mutableMapOf<String, GraphVertex>()

    operator fun get(name: String) = vertices[name]
    fun getByName(name: String) = this[name]

    val outputs: MutableList<GraphVertex> = mutableListOf()

    @GraphBuilderDSL
    fun outputs(vararg outputs: GraphVertex) {
        this.outputs.addAll(outputs.toList())
    }

    @GraphBuilderDSL
    operator fun GraphVertex.not(): GraphVertex = this.also { outputs(this) }

    @GraphBuilderDSL
    operator fun invoke(builder: ComplexNetwork.() -> Unit): ComplexNetwork = this.also(builder)

    @GraphBuilderDSL
    fun input(name: String, shape: List<Int>) = +Input(name, shape).also {
        inputs.add(it)
    }

    @GraphBuilderDSL
    fun input(name: String, vararg shape: Int) = input(name, shape.toList())

    @GraphBuilderDSL
    operator fun Input.unaryPlus(): Input {
        this@ComplexNetwork.inputs.add(this)
        addVertex(this)
        return this
    }

    @GraphBuilderDSL
    fun addVertex(vertex: GraphVertex): GraphVertex {
        if (vertex.name in vertices)
            throw IllegalArgumentException("Vertex with name ${vertex.name} already present in graph")

        vertices[vertex.name] = vertex
        _lastAdded = vertex
        return vertex
    }

    @GraphBuilderDSL
    fun addVertex(vertex: WrapperVertex<*>, inputs: List<GraphVertex>): GraphVertex =
        addVertex(VertexWrapper(vertex, inputs))

    @GraphBuilderDSL
    fun addVertex(vertex: CustomVertex, inputs: List<GraphVertex>): GraphVertex =
        addVertex(vertex.create(inputs))


    @GraphBuilderDSL
    fun addVertex(vertex: WrapperVertex<*>, vararg inputs: GraphVertex): GraphVertex =
        addVertex(vertex, inputs.toList())

    @GraphBuilderDSL
    fun addVertex(vertex: CustomVertex, vararg inputs: GraphVertex): GraphVertex =
        addVertex(vertex, inputs.toList())

    @GraphBuilderDSL
    operator fun WrapperVertex<*>.unaryPlus() = addVertex(this, currentVertex)

    @GraphBuilderDSL
    operator fun CustomVertex.unaryPlus() = addVertex(this, currentVertex)


    @GraphBuilderDSL
    fun List<GraphVertex>.then(vertex: WrapperVertex<*>) = addVertex(vertex, this)

    @GraphBuilderDSL
    fun GraphVertex.then(vertex: WrapperVertex<*>) = addVertex(vertex, this)

    @GraphBuilderDSL
    infix fun WrapperVertex<*>.from(inputs: List<GraphVertex>) = addVertex(this, inputs)

    @GraphBuilderDSL
    infix fun WrapperVertex<*>.from(inputs: GraphVertex) = addVertex(this, inputs)

    @GraphBuilderDSL
    fun WrapperVertex<*>.from(vararg inputs: GraphVertex) = addVertex(this, *inputs)


    @GraphBuilderDSL
    fun List<GraphVertex>.then(vertex: CustomVertex) = addVertex(vertex, this)

    @GraphBuilderDSL
    fun GraphVertex.then(vertex: CustomVertex) = addVertex(vertex, this)

    @GraphBuilderDSL
    infix fun CustomVertex.from(inputs: List<GraphVertex>) = addVertex(this, inputs)

    @GraphBuilderDSL
    infix fun CustomVertex.from(inputs: GraphVertex) = addVertex(this, inputs)

    @GraphBuilderDSL
    fun CustomVertex.from(vararg inputs: GraphVertex) = addVertex(this, *inputs)


    @GraphBuilderDSL
    operator fun GraphVertex.unaryPlus(): GraphVertex = addVertex(this)

    @GraphBuilderDSL
    fun addLayer(layer: ILayer, inputs: List<GraphVertex>): LayerVertex {
        return LayerVertex(layer, inputs).also { +it }
    }

    @GraphBuilderDSL
    fun addLayer(layer: ILayer, vararg inputs: GraphVertex) = addLayer(layer, inputs.toList())

    @GraphBuilderDSL
    fun addLayers(layers: List<ILayer>, inputs: List<GraphVertex>): LayerVertex {
        var layerInput = LayerVertex(layers.first(), inputs)

        addVertex(layerInput)

        layers.drop(1).forEach {
            layerInput = it from layerInput
        }

        return layerInput
    }

    @GraphBuilderDSL
    fun addLayers(layers: List<ILayer>, vararg inputs: GraphVertex) = addLayers(layers, inputs.toList())

    @GraphBuilderDSL
    fun addLayers(layers: IHasLayers, inputs: List<GraphVertex>) = addLayers(layers.layers, inputs)

    @GraphBuilderDSL
    fun addLayers(layers: IHasLayers, vararg inputs: GraphVertex) = addLayers(layers, inputs.toList())

    @GraphBuilderDSL
    fun addLayers(inputs: List<GraphVertex>, vararg layers: ILayer) = addLayers(layers.toList(), inputs)


    @GraphBuilderDSL
    operator fun IHasLayers.unaryPlus() = addLayers(this, currentVertex)


    @GraphBuilderDSL
    fun List<GraphVertex>.then(vararg layers: ILayer) = addLayers(this, *layers)

    @GraphBuilderDSL
    fun GraphVertex.then(vararg layers: ILayer) = addLayers(listOf(this), *layers)

    @GraphBuilderDSL
    infix fun List<GraphVertex>.then(layers: List<ILayer>) = addLayers(layers, this)

    @GraphBuilderDSL
    infix fun GraphVertex.then(layers: List<ILayer>) = addLayers(layers, this)

    @GraphBuilderDSL
    infix fun List<GraphVertex>.then(layers: IHasLayers) = addLayers(layers, this)

    @GraphBuilderDSL
    infix fun GraphVertex.then(layers: IHasLayers) = addLayers(layers, this)


    @GraphBuilderDSL
    infix fun List<ILayer>.from(inputs: List<GraphVertex>) = addLayers(this, inputs)

    @GraphBuilderDSL
    infix fun List<ILayer>.from(inputs: GraphVertex) = addLayers(this, inputs)

    @GraphBuilderDSL
    infix fun IHasLayers.from(inputs: List<GraphVertex>) = addLayers(this, inputs)

    @GraphBuilderDSL
    infix fun IHasLayers.from(inputs: GraphVertex) = addLayers(this, inputs)

    @GraphBuilderDSL
    fun IHasLayers.from(vararg inputs: GraphVertex) = addLayers(this, *inputs)

    @GraphBuilderDSL
    fun lambda(name: String = "lambda", lambda: SameDiffLambda) =
        +SameDiffLambdaLayer(name, lambda)


    @GraphBuilderDSL
    fun distributedOver(input: GraphVertex, dimension: Int, body: ComplexNetwork.() -> Unit): GraphVertex {
        input then DistributeDimensionLayer(dimension)
        val size = input.outputShape[dimension]
        body()
        return +CollateDimensionLayer(dimension, size)
    }

    @GraphBuilderDSL
    fun GraphVertex.thenDistributedOver(dimension: Int, body: ComplexNetwork.() -> Unit) =
        distributedOver(this, dimension, body)

    @GraphBuilderDSL
    fun distributedOver(dimension: Int, body: ComplexNetwork.() -> Unit) =
            distributedOver(currentVertex, dimension, body)

    fun ends() = vertices.values.filter { vert ->
        vertices.values.none { it.inputs != null && vert in it.inputs!! }
    }

    //TODO can I get vertex by name?  probably not

    fun makeGraph(config: ComputationGraphConfiguration.GraphBuilder.() -> ComputationGraphConfiguration.GraphBuilder = { this }):
            ComputationGraphConfiguration {
        val graphBuilder = NeuralNetConfiguration.Builder().updater(optimizer).graphBuilder().config()

        inputs.forEach {
            it.add(graphBuilder)

        }

        (vertices.values - inputs).forEach {
            it.add(graphBuilder)
        }

        if (outputs.isEmpty())
            throw IllegalStateException("No outputs defined")
        else
            graphBuilder.setOutputs(*outputs.map { it.name }.toTypedArray())

        return graphBuilder.build()
    }

//    fun summary(): String{
//        //TODO output methods
//    }
}