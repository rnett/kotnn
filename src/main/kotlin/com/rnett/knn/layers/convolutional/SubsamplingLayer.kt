package com.rnett.knn.layers.convolutional

import com.rnett.knn.*
import com.rnett.knn.layers.Layer
import org.deeplearning4j.nn.conf.ConvolutionMode
import org.deeplearning4j.nn.conf.layers.*
import kotlin.math.ceil
import org.deeplearning4j.nn.conf.layers.Subsampling1DLayer as DL4JSubsampling1DLayer
import org.deeplearning4j.nn.conf.layers.Subsampling3DLayer as DL4JSubsampling3DLayer
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer as DL4JSubsampling2DLayer

abstract class BaseSubsamplingLayer<T : NoParamLayer, S : BaseSubsamplingLayer<T, S, D>, D : NParams>(
    val dimensions: Int,
    name: String = "subsampling"
) :
    Layer<T, S>(name) {

    var poolingType: PoolingType = PoolingType.MAX

    abstract var kernelSize: D
    abstract var stride: D
    abstract var padding: D
    abstract var dilation: D

    var convolutionMode: ConvolutionMode? = null
    var pnorm: Int? = null
    var eps: Double = 1e-8
    var cudnnAllowFallback = true

    override fun apply(builder: org.deeplearning4j.nn.conf.layers.Layer.Builder<*>) {
        super.apply(builder)

        when (builder) {
            is DL4JSubsampling1DLayer.Builder -> {
                builder.poolingType(DL4JSubsampling2DLayer.PoolingType.valueOf(poolingType.name))
                builder.kernelSize(kernelSize[0])
                builder.stride(stride[0])
                builder.padding(padding[0])

                builder.convolutionMode(convolutionMode)
                if (pnorm != null)
                    builder.pnorm(pnorm!!)
                builder.eps(eps)
                builder.cudnnAllowFallback(cudnnAllowFallback)
            }
            is DL4JSubsampling2DLayer.Builder -> {
                builder.poolingType(DL4JSubsampling2DLayer.PoolingType.valueOf(poolingType.name))
                builder.kernelSize(kernelSize[0], kernelSize[1])
                builder.stride(stride[0], stride[1])
                builder.padding(padding[0], padding[1])
                builder.dilation(dilation[0], dilation[1])

                builder.convolutionMode(convolutionMode)
                if (pnorm != null)
                    builder.pnorm(pnorm!!)
                builder.eps(eps)
                builder.cudnnAllowFallback(cudnnAllowFallback)
            }
            is DL4JSubsampling3DLayer.Builder -> {
                builder.poolingType(DL4JSubsampling3DLayer.PoolingType.valueOf(poolingType.name))
                builder.kernelSize(kernelSize[0], kernelSize[1], kernelSize[2])
                builder.stride(stride[0], stride[1], stride[2])
                builder.padding(padding[0], padding[1], padding[2])
                builder.dilation(dilation[0], dilation[1], dilation[2])

                builder.convolutionMode(convolutionMode)

                builder.cudnnAllowFallback(cudnnAllowFallback)
            }
        }
    }
}

class Subsampling1DLayer(
    poolingType: PoolingType = PoolingType.MAX,
    kernelSize: Int = 2,
    stride: Int = 1,
    padding: Int = 0,
    name: String = "subsampling1d",
    layerBuilder: Subsampling1DLayer.() -> Unit = {}
) : BaseSubsamplingLayer<DL4JSubsampling1DLayer, Subsampling1DLayer, Param1>(1, name) {
    override var kernelSize by nonNegativeParams(kernelSize.p1)
    override var stride by nonNegativeParams(stride.p1)
    override var padding by nonNegativeParams(padding.p1)
    override var dilation by nonNegativeParams(1.p1)

    init {
        this.poolingType = poolingType
    }

    override fun outputShape(inputShape: List<Int>): List<Int> {
        inputShape.enforceLength(2, "Input of Subsampling1DLayer must have two dimensions")

        val effectiveKernel = if (dilation.first == 1) {
            kernelSize.first
        } else {
            kernelSize.first + (kernelSize.first - 1) * (dilation.first - 1)
        }

        val outputLength = if (convolutionMode == ConvolutionMode.Same)
            ceil(inputShape[1].toDouble() / stride.first).toInt()
        else
            (inputShape[1] - effectiveKernel + 2 * padding.first) / stride.first + 1

        return listOf(inputShape[0], outputLength)
    }

    override fun builder() = DL4JSubsampling1DLayer.Builder()

    init {
        layerBuilder()
    }
}

class Subsampling2DLayer(
    poolingType: PoolingType = PoolingType.MAX,
    kernelSize: Param2 = P(1, 1),
    stride: Param2 = P(2, 2),
    padding: Param2 = P(0, 0),
    dilation: Param2 = P(1, 1),
    name: String = "subsampling2d",
    layerBuilder: Subsampling2DLayer.() -> Unit = {}
) : BaseSubsamplingLayer<DL4JSubsampling2DLayer, Subsampling2DLayer, Param2>(2, name) {
    override var kernelSize by nonNegativeParams(kernelSize)
    override var stride by nonNegativeParams(stride)
    override var padding by nonNegativeParams(padding)
    override var dilation by nonNegativeParams(dilation)

    init {
        this.poolingType = poolingType
    }

    override fun outputShape(inputShape: List<Int>): List<Int> {
        inputShape.enforceLength(3, "Input of Subsampling2DLayer must have three dimensions")

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

            return listOf(inputShape[0], outH, outW)
        }

        val hOut = (inHeight - kH + 2 * padH) / sH + 1
        val wOut = (inWidth - kW + 2 * padW) / sW + 1

        return listOf(inputShape[0], hOut, wOut)
    }

    override fun builder() = DL4JSubsampling2DLayer.Builder()

    init {
        layerBuilder()
    }
}

class Subsampling3DLayer(
    poolingType: PoolingType,
    kernelSize: Param3 = P[1],
    stride: Param3 = P[2],
    padding: Param3 = P[0],
    dilation: Param3 = P[1],
    name: String = "subsampling3d",
    layerBuilder: Subsampling3DLayer.() -> Unit = {}
) : BaseSubsamplingLayer<DL4JSubsampling3DLayer, Subsampling3DLayer, Param3>(2, name) {
    override var kernelSize by nonNegativeParams(kernelSize)
    override var stride by nonNegativeParams(stride)
    override var padding by nonNegativeParams(padding)
    override var dilation by nonNegativeParams(dilation)

    var dataFormat: Convolution3D.DataFormat = Convolution3D.DataFormat.NCDHW

    init {
        this.poolingType = poolingType
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


            if (dataFormat == Convolution3D.DataFormat.NCDHW)
                return listOf(inputShape[0], outD, outH, outW)
            else
                return listOf(outD, outH, outW, inputShape[0])
        }

        val dOut = (inDepth - kD + 2 * padD) / sD + 1
        val hOut = (inHeight - kH + 2 * padH) / sH + 1
        val wOut = (inWidth - kW + 2 * padW) / sW + 1

        if (dataFormat == Convolution3D.DataFormat.NCDHW)
            return listOf(inputShape[0], dOut, hOut, wOut)
        else
            return listOf(dOut, hOut, wOut, inputShape[0])

    }

    override fun builder() = DL4JSubsampling1DLayer.Builder()

    init {
        layerBuilder()
    }
}