package com.rnett.knn.components

import org.nd4j.linalg.learning.config.*

object Optimizers {
    val SGD = org.nd4j.linalg.learning.config.Sgd()
    val Adam = org.nd4j.linalg.learning.config.Adam()
    val AdaMax = org.nd4j.linalg.learning.config.AdaMax()
    val AdaDelta = org.nd4j.linalg.learning.config.AdaDelta()
    val Nesterovs = org.nd4j.linalg.learning.config.Nesterovs()
    val NAdam = Nadam()
    val AdaGrad = org.nd4j.linalg.learning.config.AdaGrad()
    val RmsProp = org.nd4j.linalg.learning.config.RmsProp()
    val None = NoOp()
}

operator fun Sgd.invoke(learningRate: Double = 1e-3) = Sgd(learningRate)
operator fun Adam.invoke(
    learningRate: Double = 1e-3,
    beta1: Double = 0.9,
    beta2: Double = 0.999,
    epsilon: Double = 1e-8
) =
    Adam(learningRate, beta1, beta2, epsilon)

operator fun AdaMax.invoke(
    learningRate: Double = 1e-3,
    beta1: Double = 0.9,
    beta2: Double = 0.999,
    epsilon: Double = 1e-8
) =
    AdaMax(learningRate, beta1, beta2, epsilon)

operator fun AdaDelta.invoke(rho: Double = 0.95, epsilon: Double = 1e-6) =
    AdaDelta(rho, epsilon)

operator fun Nesterovs.invoke(learningRate: Double = 0.1, momentum: Double = 0.9) =
    Nesterovs(learningRate, momentum)

operator fun Nadam.invoke(
    learningRate: Double = 1e-3,
    beta1: Double = 0.9,
    beta2: Double = 0.999,
    epsilon: Double = 1e-8
) =
    Nadam(learningRate, beta1, beta2, epsilon)

operator fun AdaGrad.invoke(learningRate: Double = 1e-1, epsilon: Double = 1e-6) =
    AdaGrad(learningRate, epsilon)

operator fun RmsProp.invoke(learningRate: Double = 1e-1, rmsDecay: Double = 0.95, epsilon: Double = 1e-8) =
    RmsProp(learningRate, rmsDecay, epsilon)