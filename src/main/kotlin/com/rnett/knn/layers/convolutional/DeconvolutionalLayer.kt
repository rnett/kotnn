package com.rnett.knn.layers.convolutional

import com.rnett.knn.*
import com.rnett.knn.layers.feedforeward.FeedForwardLayer
import org.deeplearning4j.exception.DL4JInvalidConfigException
import org.deeplearning4j.nn.conf.ConvolutionMode
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.Deconvolution2D as DL4JDeconvolution2DLayer
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import kotlin.math.ceil

abstract class BaseDeconvolutionLayer<out T : ConvolutionLayer, D : NParams, S : BaseDeconvolutionLayer<T, D, S>>(
    dimensions: Int,
    filters: Int,
    name: String
) : BaseConvolutionLayer<T, D, S>(dimensions, filters, name) {

}

/*class Deconvolution1DLayer(
    filters: Int,
    kernelSize: Int = 0,
    stride: Int = 1,
    padding: Int = 0,
    dilation: Int = 1,
    name: String = "conv1d",
    layerBuilder: Deconvolution1DLayer.() -> Unit = {}
) : BaseDeconvolutionLayer<org.deeplearning4j.nn.conf.layers.Deconvolution1DLayer, Param1, Deconvolution1DLayer>(1, filters, name) {

    override fun outputShape(inputShape: List<Int>): List<Int> {

        inputShape.enforceLength(2, "Input of Deconvolution1DLayer must have two dimensions")

        val effectiveKernel = if (dilation.first == 1) {
            kernelSize.first
        } else {
            kernelSize.first + (kernelSize.first - 1) * (dilation.first - 1)
        }

        val outputLength = if (deconvolutionMode == DeconvolutionMode.Same)
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

    override fun builder(): org.deeplearning4j.nn.conf.layers.Layer.Builder<*> = org.deeplearning4j.nn.conf.layers.Deconvolution1DLayer.Builder()

    init {
        layerBuilder()
    }
}*/

class Deconvolution2DLayer(
    filters: Int,
    kernelSize: Param2 = Param2(5, 5),
    stride: Param2 = Param2(1, 1),
    padding: Param2 = Param2(0, 0),
    dilation: Param2 = Param2(1, 1),
    name: String = "deconv2d",
    layerBuilder: Deconvolution2DLayer.() -> Unit = {}
) : BaseDeconvolutionLayer<DL4JDeconvolution2DLayer, Param2, Deconvolution2DLayer>(2, filters, name) {

    override var kernelSize by nonNegativeParams(kernelSize)
    override var stride by nonNegativeParams(stride)
    override var padding by nonNegativeParams(padding)
    override var dilation by nonNegativeParams(dilation)

    override fun builder(): org.deeplearning4j.nn.conf.layers.Layer.Builder<*> = DL4JDeconvolution2DLayer.Builder()

    init {
        layerBuilder()
    }

    override fun outputShape(inputShape: List<Int>): List<Int> {
        inputShape.enforceLength(3, "Input of Deconvolution2DLayer must have three dimensions")

        val hIn = inputShape[1]
        val wIn = inputShape[2]

        val padH = if (convolutionMode == ConvolutionMode.Same) 0 else padding[0]
        val padW = if (convolutionMode == ConvolutionMode.Same) 0 else padding[1]
        var kH = kernelSize[0]
        var kW = kernelSize[1]
        if (dilation[0] != 1) {
            kH += (kH - 1) * (dilation[0] - 1)
        }
        if (dilation[1] != 1) {
            kW += (kW - 1) * (dilation[1] - 1)
        }

        val sH = stride[0]
        val sW = stride[1]

        if (convolutionMode == ConvolutionMode.Same) {
            val hOut = stride[0] * hIn
            val wOut = stride[1] * wIn
            return listOf(nodes, hOut, wOut)
        }

        val hOut = sH * (hIn - 1) + kH - 2 * padH
        val wOut = sW * (wIn - 1) + kW - 2 * padW

        return listOf(nodes, hOut, wOut)
    }
}

/*
class Deconvolution3DLayer(
    filters: Int,
    kernelSize: Param3 = Param3(2, 2, 2),
    stride: Param3 = Param3(1, 1, 1),
    padding: Param3 = Param3(0, 0, 0),
    dilation: Param3 = Param3(1, 1, 1),
    name: String = "conv3d",
    layerBuilder: Deconvolution3DLayer.() -> Unit = {}
) : BaseDeconvolutionLayer<Deconvolution3D, Param3, Deconvolution3DLayer>(3, filters, name) {

    override var kernelSize by nonNegativeParams(kernelSize)
    override var stride by nonNegativeParams(stride)
    override var padding by nonNegativeParams(padding)
    override var dilation by nonNegativeParams(dilation)

    var dataFormat: Deconvolution3D.DataFormat = Deconvolution3D.DataFormat.NCDHW

    override fun builder(): org.deeplearning4j.nn.conf.layers.Layer.Builder<*> = Deconvolution3D.Builder()

    override fun apply(builder: org.deeplearning4j.nn.conf.layers.Layer.Builder<*>) {
        super.apply(builder)
        builder as Deconvolution3D.Builder
        builder.dataFormat(dataFormat)
    }

    init {
        layerBuilder()
    }

    override fun outputShape(inputShape: List<Int>): List<Int> {

        val (_, inDepth, inWidth, inHeight) = if (dataFormat == Deconvolution3D.DataFormat.NCDHW) {
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
        if (deconvolutionMode == DeconvolutionMode.Same) {

            val outD = Math.ceil(inDepth / (sD.toDouble())).toInt()
            val outH = Math.ceil(inHeight / (sH.toDouble())).toInt()
            val outW = Math.ceil(inWidth / (sW.toDouble())).toInt()


            if(dataFormat == Deconvolution3D.DataFormat.NCDHW)
                return listOf(nodes, outD, outH, outW)
            else
                return listOf(outD, outH, outW, nodes)
        }

        val dOut = (inDepth - kD + 2 * padD) / sD + 1
        val hOut = (inHeight - kH + 2 * padH) / sH + 1
        val wOut = (inWidth - kW + 2 * padW) / sW + 1

        if(dataFormat == Deconvolution3D.DataFormat.NCDHW)
            return listOf(nodes, dOut, hOut, wOut)
        else
            return listOf(dOut, hOut, wOut, nodes)

    }
}*/
