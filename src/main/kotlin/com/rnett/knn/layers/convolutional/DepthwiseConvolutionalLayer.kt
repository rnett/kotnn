package com.rnett.knn.layers.convolutional

import com.rnett.knn.NParams
import com.rnett.knn.Param2
import com.rnett.knn.enforceLength
import com.rnett.knn.nonNegativeParams
import org.deeplearning4j.nn.conf.ConvolutionMode
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.conf.layers.DepthwiseConvolution2D
import org.deeplearning4j.nn.conf.layers.Layer
import org.deeplearning4j.nn.conf.layers.DepthwiseConvolution2D as DL4JDepthwiseConvolution2DLayer

abstract class BaseDepthwiseConvolutionLayer<out T : ConvolutionLayer, D : NParams, S : BaseDepthwiseConvolutionLayer<T, D, S>>(
    dimensions: Int,
    filters: Int,
    name: String
) : BaseConvolutionLayer<T, D, S>(dimensions, filters, name)

/*class DepthwiseConvolution1DLayer(
    filters: Int,
    kernelSize: Int = 0,
    stride: Int = 1,
    padding: Int = 0,
    dilation: Int = 1,
    name: String = "conv1d",
    layerBuilder: DepthwiseConvolution1DLayer.() -> Unit = {}
) : BaseDepthwiseConvolutionLayer<org.deeplearning4j.nn.conf.layers.DepthwiseConvolution1DLayer, Param1, DepthwiseConvolution1DLayer>(1, filters, name) {

    override fun outputShape(inputShape: List<Int>): List<Int> {

        inputShape.enforceLength(2, "Input of DepthwiseConvolution1DLayer must have two dimensions")

        val effectiveKernel = if (dilation.first == 1) {
            kernelSize.first
        } else {
            kernelSize.first + (kernelSize.first - 1) * (dilation.first - 1)
        }

        val outputLength = if (depthwiseConvolutionMode == DepthwiseConvolutionMode.Same)
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
        set(v){kernelSize = P(v)
        }

    var intStride
        get() = stride.first
        set(v){stride = P(v)
        }

    var intPadding
        get() = padding.first
        set(v){padding = P(v)
        }

    var intDilation
        get() = dilation.first
        set(v){dilation = P(v)
        }

    override fun builder(): org.deeplearning4j.nn.conf.layers.Layer.Builder<*> = org.deeplearning4j.nn.conf.layers.DepthwiseConvolution1DLayer.Builder()

    init {
        layerBuilder()
    }
}*/

class DepthwiseConvolution2DLayer(
    filters: Int,
    kernelSize: Param2 = Param2(5, 5),
    stride: Param2 = Param2(1, 1),
    padding: Param2 = Param2(0, 0),
    dilation: Param2 = Param2(1, 1),
    name: String = "depthwiseconv2d",
    layerBuilder: DepthwiseConvolution2DLayer.() -> Unit = {}
) : BaseDepthwiseConvolutionLayer<DepthwiseConvolution2D, Param2, DepthwiseConvolution2DLayer>(2, filters, name) {

    override var kernelSize by nonNegativeParams(kernelSize)
    override var stride by nonNegativeParams(stride)
    override var padding by nonNegativeParams(padding)
    override var dilation by nonNegativeParams(dilation)

    var depthMultiplier: Int = 1

    override fun apply(builder: Layer.Builder<*>) {
        super.apply(builder)

        builder as DepthwiseConvolution2D.Builder
        builder.depthMultiplier(depthMultiplier)
    }

    override fun builder(): org.deeplearning4j.nn.conf.layers.Layer.Builder<*> = DepthwiseConvolution2D.Builder()

    init {
        layerBuilder()
    }

    override fun outputShape(inputShape: List<Int>): List<Int> {
        inputShape.enforceLength(3, "Input of DepthwiseConvolution2DLayer must have three dimensions")

        val inHeight = inputShape[1]
        val inWidth = inputShape[2]

        val (padH, padW) = padding
        var (kH, kW) = kernelSize

        if (dilation.first != 1) {
            //Use *effective* kernel size, accounting for dilation
            kH += (kH - 1) * (dilation[0] - 1)
        }
        if (dilation.second != 1) {
            kW += (kW - 1) * (dilation[1] - 1)
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

/*
class DepthwiseConvolution3DLayer(
    filters: Int,
    kernelSize: Param3 = Param3(2, 2, 2),
    stride: Param3 = Param3(1, 1, 1),
    padding: Param3 = Param3(0, 0, 0),
    dilation: Param3 = Param3(1, 1, 1),
    name: String = "conv3d",
    layerBuilder: DepthwiseConvolution3DLayer.() -> Unit = {}
) : BaseDepthwiseConvolutionLayer<DepthwiseConvolution3D, Param3, DepthwiseConvolution3DLayer>(3, filters, name) {

    override var kernelSize by nonNegativeParams(kernelSize)
    override var stride by nonNegativeParams(stride)
    override var padding by nonNegativeParams(padding)
    override var dilation by nonNegativeParams(dilation)

    var dataFormat: DepthwiseConvolution3D.DataFormat = DepthwiseConvolution3D.DataFormat.NCDHW

    override fun builder(): org.deeplearning4j.nn.conf.layers.Layer.Builder<*> = DepthwiseConvolution3D.Builder()

    override fun apply(builder: org.deeplearning4j.nn.conf.layers.Layer.Builder<*>) {
        super.apply(builder)
        builder as DepthwiseConvolution3D.Builder
        builder.dataFormat(dataFormat)
    }

    init {
        layerBuilder()
    }

    override fun outputShape(inputShape: List<Int>): List<Int> {

        val (_, inDepth, inWidth, inHeight) = if (dataFormat == DepthwiseConvolution3D.DataFormat.NCDHW) {
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
        if (depthwiseConvolutionMode == DepthwiseConvolutionMode.Same) {

            val outD = Math.ceil(inDepth / (sD.toDouble())).toInt()
            val outH = Math.ceil(inHeight / (sH.toDouble())).toInt()
            val outW = Math.ceil(inWidth / (sW.toDouble())).toInt()


            if(dataFormat == DepthwiseConvolution3D.DataFormat.NCDHW)
                return listOf(nodes, outD, outH, outW)
            else
                return listOf(outD, outH, outW, nodes)
        }

        val dOut = (inDepth - kD + 2 * padD) / sD + 1
        val hOut = (inHeight - kH + 2 * padH) / sH + 1
        val wOut = (inWidth - kW + 2 * padW) / sW + 1

        if(dataFormat == DepthwiseConvolution3D.DataFormat.NCDHW)
            return listOf(nodes, dOut, hOut, wOut)
        else
            return listOf(dOut, hOut, wOut, nodes)

    }
}*/
