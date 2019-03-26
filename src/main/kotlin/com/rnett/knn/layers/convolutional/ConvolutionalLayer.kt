package com.rnett.knn.layers.convolutional

import com.rnett.knn.*
import com.rnett.knn.layers.feedforeward.FeedForwardLayer
import org.deeplearning4j.nn.conf.ConvolutionMode
import org.deeplearning4j.nn.conf.layers.Convolution1DLayer as DL4JConvolution1DLayer
import org.deeplearning4j.nn.conf.layers.Convolution3D
import kotlin.math.ceil
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer as DL4JConvolutionLayer

abstract class BaseConvolutionLayer<out T : DL4JConvolutionLayer, D : NParams, S : BaseConvolutionLayer<T, D, S>>(
    val dimensions: Int,
    filters: Int,
    name: String
) : FeedForwardLayer<T, S>(name, filters) {

    var hasBias: Boolean = true
    var convolutionMode: ConvolutionMode = ConvolutionMode.Truncate
    var cudnnAllowFallback = true

    var cudnnAlgoMode = DL4JConvolutionLayer.AlgoMode.PREFER_FASTEST
    var cudnnFwdAlgo: DL4JConvolutionLayer.FwdAlgo? = null
    var cudnnBwdFilterAlgo: DL4JConvolutionLayer.BwdFilterAlgo? = null
    var cudnnBwdDataAlgo: DL4JConvolutionLayer.BwdDataAlgo? = null

    abstract var kernelSize: D
    abstract var stride: D
    abstract var padding: D
    abstract var dilation: D

    override fun apply(builder: org.deeplearning4j.nn.conf.layers.Layer.Builder<*>) {
        super.apply(builder)
        builder as DL4JConvolutionLayer.BaseConvBuilder

        builder.hasBias(hasBias)
        builder.convolutionMode(convolutionMode)
        builder.cudnnAllowFallback(cudnnAllowFallback)

        builder.cudnnAlgoMode(cudnnAlgoMode)
        builder.cudnnFwdMode(cudnnFwdAlgo)
        builder.cudnnBwdFilterMode(cudnnBwdFilterAlgo)
        builder.cudnnBwdDataMode(cudnnBwdDataAlgo)

        if (dimensions == 1) {
            builder.kernelSize(*kernelSize.toIntArray(), 1)
            builder.stride(*stride.toIntArray(), 1)
            builder.padding(*padding.toIntArray(), 0)
            builder.dilation(*dilation.toIntArray())
        } else {
            builder.kernelSize(*kernelSize.toIntArray())
            builder.stride(*stride.toIntArray())
            builder.padding(*padding.toIntArray())
            builder.dilation(*dilation.toIntArray())
        }

    }
}

class Convolution1DLayer(
    filters: Int,
    kernelSize: Int = 0,
    stride: Int = 1,
    padding: Int = 0,
    dilation: Int = 1,
    name: String = "conv1d",
    layerBuilder: Convolution1DLayer.() -> Unit = {}
) : BaseConvolutionLayer<DL4JConvolution1DLayer, Param1, Convolution1DLayer>(1, filters, name) {

    override fun outputShape(inputShape: List<Int>): List<Int> {

        inputShape.enforceLength(2, "Input of Convolution1DLayer must have two dimensions")

        val effectiveKernel = if (dilation.first == 1) {
            kernelSize.first
        } else {
            kernelSize.first + (kernelSize.first - 1) * (dilation.first - 1)
        }

        val outputLength = if (convolutionMode == ConvolutionMode.Same)
            ceil(inputShape[1].toDouble() / stride.first).toInt()
        else
            (inputShape[1] - effectiveKernel + 2 * padding.first) / stride.first + 1

        return listOf(nOut, outputLength)
    }

    override var kernelSize by nonNegativeParams<Param1>(kernelSize)
    override var stride by nonNegativeParams<Param1>(stride)
    override var padding by nonNegativeParams<Param1>(padding)
    override var dilation by nonNegativeParams<Param1>(dilation)

    var intKernelSize
        get() = kernelSize.first
        set(v){kernelSize = P(v)}

    var intStride
        get() = stride.first
        set(v){stride = P(v)}

    var intPadding
        get() = padding.first
        set(v){padding = P(v)}

    var intDilation
        get() = dilation.first
        set(v){dilation = P(v)}

    override fun builder(): org.deeplearning4j.nn.conf.layers.Layer.Builder<*> = DL4JConvolution1DLayer.Builder()

    init {
        layerBuilder()
    }
}

class Convolution2DLayer(
    filters: Int,
    kernelSize: Param2 = Param2(5, 5),
    stride: Param2 = Param2(1, 1),
    padding: Param2 = Param2(0, 0),
    dilation: Param2 = Param2(1, 1),
    name: String = "conv2d",
    layerBuilder: Convolution2DLayer.() -> Unit = {}
) : BaseConvolutionLayer<DL4JConvolutionLayer, Param2, Convolution2DLayer>(2, filters, name) {

    override var kernelSize by nonNegativeParams(kernelSize)
    override var stride by nonNegativeParams(stride)
    override var padding by nonNegativeParams(padding)
    override var dilation by nonNegativeParams(dilation)

    override fun builder(): org.deeplearning4j.nn.conf.layers.Layer.Builder<*> = DL4JConvolutionLayer.Builder()

    init {
        layerBuilder()
    }

    override fun outputShape(inputShape: List<Int>): List<Int> {
        inputShape.enforceLength(3, "Input of Convolution2DLayer must have three dimensions")

        val inHeight = inputShape[1]
        val inWidth = inputShape[2]

        val (padH, padW) = padding
        var (kH, kW) = kernelSize

        if (dilation.first != 1) {
            //Use *effective* kernel size, accounting for dilation
            kH = kH + (kH - 1) * (dilation[0] - 1)
        }
        if (dilation.second != 1) {
            kW = kW + (kW - 1) * (dilation[1] - 1)
        }

        val (sH, sW) = stride

        if (convolutionMode == ConvolutionMode.Same) {

            val outH = Math.ceil(inHeight / (stride[0].toDouble())).toInt()
            val outW = Math.ceil(inWidth / (stride[1].toDouble())).toInt()

            return listOf(nodes, outH, outW)
        }

        val hOut = (inHeight - kH + 2 * padH) / sH + 1
        val wOut = (inWidth - kW + 2 * padW) / sW + 1

        return listOf(nodes, hOut, wOut)
    }
}

class Convolution3DLayer(
    filters: Int,
    kernelSize: Param3 = Param3(2, 2, 2),
    stride: Param3 = Param3(1, 1, 1),
    padding: Param3 = Param3(0, 0, 0),
    dilation: Param3 = Param3(1, 1, 1),
    name: String = "conv3d",
    layerBuilder: Convolution3DLayer.() -> Unit = {}
) : BaseConvolutionLayer<Convolution3D, Param3, Convolution3DLayer>(3, filters, name) {

    override var kernelSize by nonNegativeParams(kernelSize)
    override var stride by nonNegativeParams(stride)
    override var padding by nonNegativeParams(padding)
    override var dilation by nonNegativeParams(dilation)

    var dataFormat: Convolution3D.DataFormat = Convolution3D.DataFormat.NCDHW

    override fun builder(): org.deeplearning4j.nn.conf.layers.Layer.Builder<*> = Convolution3D.Builder()

    override fun apply(builder: org.deeplearning4j.nn.conf.layers.Layer.Builder<*>) {
        super.apply(builder)
        builder as Convolution3D.Builder
        builder.dataFormat(dataFormat)
    }

    init {
        layerBuilder()
    }

    override fun outputShape(inputShape: List<Int>): List<Int> {

        val (_, inDepth, inWidth, inHeight) = if (dataFormat == Convolution3D.DataFormat.NCDHW) {
            inputShape
        } else
            listOf(inputShape[3], inputShape[0], inputShape[1], inputShape[2])

        val (padD, padH, padW) = padding

        var (kD, kH, kW) = kernelSize


        if (dilation[0] != 1) {
            //Use *effective* kernel size, accounting for dilation
            kD += (kD - 1) * (dilation[0] - 1)
        }
        if (dilation[1] != 1) {
            kH += (kH - 1) * (dilation[1] - 1)
        }
        if (dilation[2] != 1) {
            kW += (kW - 1) * (dilation[2] - 1)
        }

        val sD = stride[0]
        val sH = stride[1]
        val sW = stride[1]

        //Strict mode: require exactly the right size...
        if (convolutionMode == ConvolutionMode.Same) {

            val outD = Math.ceil(inDepth / (sD.toDouble())).toInt()
            val outH = Math.ceil(inHeight / (sH.toDouble())).toInt()
            val outW = Math.ceil(inWidth / (sW.toDouble())).toInt()


            if(dataFormat == Convolution3D.DataFormat.NCDHW)
                return listOf(nodes, outD, outH, outW)
            else
                return listOf(outD, outH, outW, nodes)
        }

        val dOut = (inDepth - kD + 2 * padD) / sD + 1
        val hOut = (inHeight - kH + 2 * padH) / sH + 1
        val wOut = (inWidth - kW + 2 * padW) / sW + 1

        if(dataFormat == Convolution3D.DataFormat.NCDHW)
            return listOf(nodes, dOut, hOut, wOut)
        else
            return listOf(dOut, hOut, wOut, nodes)

    }
}