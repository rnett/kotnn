package com.rnett.knn.layers.convolutional

import com.rnett.knn.*
import com.rnett.knn.layers.Layer
import org.deeplearning4j.nn.conf.ConvolutionMode
import org.deeplearning4j.nn.conf.layers.LocallyConnected1D
import org.deeplearning4j.nn.conf.layers.LocallyConnected2D
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayer
import org.deeplearning4j.util.Convolution1DUtils
import org.nd4j.linalg.activations.Activation
import kotlin.math.ceil

//TODO check sizes

abstract class BaseLocallyConnectedLayer<T : SameDiffLayer, S : BaseLocallyConnectedLayer<T, S, D>, D : NParams>(
    nodes: Int, val dimensions: Int, name: String = "locallyconnected"
) :
    Layer<T, S>(name) {

    open var nOut: Int = 0

    var nodes
        get() = nOut
        set(v) {
            nOut = v
        }

    init {
        this.nodes = nodes
    }

    var activation: Activation = Activation.TANH

    abstract var kernelSize: D
    abstract var stride: D
    abstract var padding: D
    abstract var dilation: D

    var convolutionMode = ConvolutionMode.Same

    var hasBias = true

    override fun apply(builder: org.deeplearning4j.nn.conf.layers.Layer.Builder<*>) {
        super.apply(builder)

        if (builder is LocallyConnected1D.Builder) {
            builder.nOut(nOut)
            builder.activation(activation)
            builder.kernelSize(kernelSize[0])
            builder.stride(stride[0])
            builder.padding(padding[0])
            builder.dilation(dilation[0])
            builder.convolutionMode(convolutionMode)
            builder.hasBias(hasBias)
        } else if (builder is LocallyConnected2D.Builder) {
            builder.nOut(nOut)
            builder.activation(activation)
            builder.kernelSize(kernelSize[0], kernelSize[1])
            builder.stride(stride[0], stride[1])
            builder.padding(padding[0], padding[1])
            builder.dilation(dilation[0], dilation[1])
            builder.convolutionMode(convolutionMode)
            builder.hasBias(hasBias)
        }
    }
}

class LocallyConnected1DLayer(
    nodes: Int,
    kernelSize: Int = 2,
    stride: Int = 1,
    padding: Int = 0,
    dilation: Int = 1,
    name: String = "locallyconnected1d",
    layerBuilder: LocallyConnected1DLayer.() -> Unit = {}
) : BaseLocallyConnectedLayer<LocallyConnected1D, LocallyConnected1DLayer, Param1>(nodes, 1, name) {
    override var kernelSize by nonNegativeParams(kernelSize.p1)
    override var stride by nonNegativeParams(stride.p1)
    override var padding by nonNegativeParams(padding.p1)
    override var dilation by nonNegativeParams(dilation.p1)

    override fun outputShape(inputShape: List<Int>): List<Int> {
        val inH = inputShape[1]
        val eKernel = if (dilation[0] == 1) {
            kernelSize[0]
        } else {
            kernelSize[0] + (kernelSize[0] - 1) * (dilation[0] - 1)
        }

        val tempOut = if (convolutionMode == ConvolutionMode.Same) {
            Math.ceil(inH / stride[0].toDouble()).toInt()
        } else {
            (inH - eKernel + 2 * 0) / stride[0] + 1
        }

        val tempPadding = if (convolutionMode == ConvolutionMode.Same) {
            ((tempOut - 1) * stride[0] + eKernel - inputShape[1]) / 2
        } else padding[0]

        val outputLength = if (convolutionMode == ConvolutionMode.Same)
            ceil(inputShape[1].toDouble() / stride.first).toInt()
        else
            (inputShape[1] - eKernel + 2 * tempPadding) / stride.first + 1

        return listOf(nOut, outputLength)
    }

    override fun builder() = LocallyConnected1D.Builder()

    init {
        layerBuilder()
    }
}

class LocallyConnected2DLayer(
    nodes: Int,
    kernelSize: Param2 = 2.p2,
    stride: Param2 = 1.p2,
    padding: Param2 = 0.p2,
    dilation: Param2 = 1.p2,
    name: String = "locallyconnected2d",
    layerBuilder: LocallyConnected2DLayer.() -> Unit = {}
) : BaseLocallyConnectedLayer<LocallyConnected2D, LocallyConnected2DLayer, Param2>(nodes, 2, name) {
    override var kernelSize by nonNegativeParams(kernelSize)
    override var stride by nonNegativeParams(stride)
    override var padding by nonNegativeParams(padding)
    override var dilation by nonNegativeParams(dilation)

    override fun outputShape(inputShape: List<Int>): List<Int> {
        inputShape.enforceLength(3, "Input of Convolution2DLayer must have three dimensions")

        val inHeight = inputShape[1]
        val inWidth = inputShape[2]

        var (padH, padW) = Pair(0, 0)
        var (kH, kW) = kernelSize

        if (dilation.first != 1) {
            //Use *effective* kernel size, accounting for dilation
            kH = kH + (kH - 1) * (dilation[0] - 1)
        }
        if (dilation.second != 1) {
            kW = kW + (kW - 1) * (dilation[1] - 1)
        }

        val (sH, sW) = stride

        val tempOut = if (convolutionMode == ConvolutionMode.Same) {

            val outH = Math.ceil(inHeight / (stride[0].toDouble())).toInt()
            val outW = Math.ceil(inWidth / (stride[1].toDouble())).toInt()

            listOf(outH, outW)
        } else {
            val hOut = (inHeight - kH + 2 * padH) / sH + 1
            val wOut = (inWidth - kW + 2 * padW) / sW + 1

            listOf(hOut, wOut)
        }

        val tempPadding = if (convolutionMode == ConvolutionMode.Same) {
            val eKernel = if (dilation[0] == 1 && dilation[1] == 1) {
                kernelSize.toList()
            } else {
                listOf(
                    kernelSize[0] + (kernelSize[0] - 1) * (dilation[0] - 1),
                    kernelSize[1] + (kernelSize[1] - 1) * (dilation[1] - 1)
                )
            }
            listOf(
                ((tempOut[0] - 1) * stride[0] + eKernel[0] - inputShape[0]) / 2,
                ((tempOut[1] - 1) * stride[1] + eKernel[1] - inputShape[1]) / 2
            )

        } else padding.toList()

        padH = tempPadding[0]
        padW = tempPadding[1]

        if (dilation.first != 1) {
            //Use *effective* kernel size, accounting for dilation
            kH = kH + (kH - 1) * (dilation[0] - 1)
        }
        if (dilation.second != 1) {
            kW = kW + (kW - 1) * (dilation[1] - 1)
        }

        if (convolutionMode == ConvolutionMode.Same) {

            val outH = Math.ceil(inHeight / (stride[0].toDouble())).toInt()
            val outW = Math.ceil(inWidth / (stride[1].toDouble())).toInt()

            return listOf(nodes, outH, outW)
        }

        val hOut = (inHeight - kH + 2 * padH) / sH + 1
        val wOut = (inWidth - kW + 2 * padW) / sW + 1

        return listOf(nodes, hOut, wOut)
    }

    override fun builder() = LocallyConnected2D.Builder()

    init {
        layerBuilder()
    }
}