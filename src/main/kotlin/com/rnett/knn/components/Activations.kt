package com.rnett.knn.components

import org.nd4j.linalg.activations.impl.*
import org.nd4j.linalg.api.ndarray.INDArray

object Activations {
    val LeakyReLU = ActivationLReLU()
    val ReLU = ActivationReLU()
    val RandomizedReLU = ActivationRReLU()
    val ELU = ActivationELU()
    //TODO snapshot only val GELU = ActivationGELU()
    fun PReLU(alpha: INDArray, vararg sharedAxes: Long) = ActivationPReLU(
        alpha,
        if (sharedAxes.isEmpty()) null else sharedAxes
    )

    val ReLU6 = ActivationReLU6()
    val SELU = ActivationSELU()
    val ThresholdReLU = ActivationThresholdedReLU()

    val Cube = ActivationCube()
    val HardSigmoid = ActivationHardSigmoid()
    val HardTanh = ActivationHardTanH()
    val Identity = ActivationIdentity()
    val RationalTanh = ActivationRationalTanh()
    val RectifiedTanh = ActivationRectifiedTanh()
    val Sigmoid = ActivationSigmoid()
    val Softmax = ActivationSoftmax()
    val SoftPlus = ActivationSoftPlus()
    val SoftSign = ActivationSoftSign()
    val Swish = ActivationSwish()
    val TanH = ActivationTanH()
}

operator fun ActivationLReLU.invoke(alpha: Double) = ActivationLReLU(alpha)
operator fun ActivationRReLU.invoke(l: Double, u: Double) = ActivationRReLU(l, u)
operator fun ActivationELU.invoke(alpha: Double) = ActivationELU(alpha)
//TODO snapshot only operator fun ActivationGELU.invoke(percise: Boolean) = ActivationGELU(percise)
operator fun ActivationThresholdedReLU.invoke(theta: Double) = ActivationThresholdedReLU(theta)