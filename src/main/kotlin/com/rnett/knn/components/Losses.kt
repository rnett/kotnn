package com.rnett.knn.components

import org.nd4j.linalg.lossfunctions.impl.*

object Losses {
    val BinaryCrossentropy = LossBinaryXENT()
    val CosineProximity = LossCosineProximity()
    val FMeasure = LossFMeasure()
    val Hinge = LossHinge()
    val KLDivergence = LossKLD()
    val L1 = LossL1()
    val MeanAbsoluteError = LossMAE()
    val L2 = LossL2()
    val MeanSquaredError = LossMSE()

    val MeanAbsolutePercentageError = LossMAPE()
    val CategoricalCrossentropy = LossMCXENT()
    val NegativeLogLikelihood = LossNegativeLogLikelihood()
    val MixtureDensity = LossMixtureDensity()
    val MeanSquaredLogarithmicError = LossMSLE()
    val MultiLabel = LossMultiLabel()
    val Poisson = LossPoisson()
    val SquaredHinge = LossSquaredHinge()

}

operator fun LossFMeasure.invoke(beta: Double) = LossFMeasure(beta)

