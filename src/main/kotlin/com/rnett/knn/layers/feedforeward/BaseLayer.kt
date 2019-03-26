package com.rnett.knn.layers.feedforeward

import com.rnett.knn.layers.Layer
import org.deeplearning4j.nn.conf.GradientNormalization
import org.deeplearning4j.nn.conf.distribution.Distribution
import org.deeplearning4j.nn.conf.weightnoise.IWeightNoise
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.IActivation
import org.nd4j.linalg.learning.config.IUpdater
import org.deeplearning4j.nn.conf.layers.BaseLayer as DL4JBaseLayer

abstract class BaseLayer<out T : DL4JBaseLayer, S : BaseLayer<T, S>>(name: String) : Layer<T, S>(name) {


    var activation: IActivation? = null

    var weightInit: WeightInit? = null
    var biasInit: Double? = null
    var distribution: Distribution? = null

    var l1: Double? = null
    var l2: Double? = null
    var l1Bias: Double? = null
    var l2Bias: Double? = null

    var updater: IUpdater? = null
    var biasUpdater: IUpdater? = null

    var gradientNormalization: GradientNormalization? = null
    var gradientNormalizationThreshold: Double? = null

    var weightNoise: IWeightNoise? = null

    override fun apply(builder: org.deeplearning4j.nn.conf.layers.Layer.Builder<*>) {
        super.apply(builder)

        builder as org.deeplearning4j.nn.conf.layers.BaseLayer.Builder

        builder.activation(activation)

        if(weightInit != null)
            builder.weightInit(weightInit)

        builder.biasInit(biasInit ?: Double.NaN)

        if(distribution != null)
            builder.dist(distribution)

        builder.l1(l1 ?: Double.NaN)
        builder.l2(l2 ?: Double.NaN)
        builder.l1Bias(l1Bias ?: Double.NaN)
        builder.l2Bias(l2Bias ?: Double.NaN)

        builder.updater(updater)
        builder.biasUpdater(biasUpdater)

        builder.gradientNormalization(gradientNormalization)
        builder.gradientNormalizationThreshold(gradientNormalizationThreshold ?: Double.NaN)

        builder.weightNoise(weightNoise)
    }

}