package com.rnett.knn.layers.feedforeward

import org.deeplearning4j.nn.api.layers.LayerConstraint
import org.deeplearning4j.nn.conf.layers.Layer
import org.deeplearning4j.nn.conf.layers.BatchNormalization as DL4JBatchNormalizationLayer

//TODO what use is nodes/nOut?
class BatchNormalizationLayer(name: String = "batchnormalization",
                              layerBuilder: BatchNormalizationLayer.() -> Unit = {}) :
    FeedForwardLayer<DL4JBatchNormalizationLayer, BatchNormalizationLayer>(name) {
    override fun outputShape(inputShape: List<Int>) = inputShape

    override fun builder() = DL4JBatchNormalizationLayer.Builder()

    var decay: Double = 0.9
    var eps: Double = 1e-5
    var isMinibatch = true
    var lockGammaBeta = false
    var gamma: Double = 1.0
    var beta: Double = 0.0
    var betaConstraints = mutableListOf<LayerConstraint>()
    var gammaConstraints = mutableListOf<LayerConstraint>()
    var cudnnAllowFallback = true

    override fun apply(builder: Layer.Builder<*>) {
        super.apply(builder)

        builder as DL4JBatchNormalizationLayer.Builder

        builder.decay(decay)
        builder.eps(eps)
        builder.minibatch(isMinibatch)
        builder.lockGammaBeta(lockGammaBeta)
        builder.gamma(gamma)
        builder.beta(beta)
        builder.constrainBeta(*betaConstraints.toTypedArray())
        builder.constrainGamma(*gammaConstraints.toTypedArray())
        builder.cudnnAllowFallback(cudnnAllowFallback)
    }

    init{
        layerBuilder()
    }
}