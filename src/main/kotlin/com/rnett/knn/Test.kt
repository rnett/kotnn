package com.rnett.knn

import com.rnett.knn.components.Activations
import com.rnett.knn.components.Losses
import com.rnett.knn.components.Optimizers
import com.rnett.knn.layers.CapsuleLayer
import com.rnett.knn.layers.CapsuleStrength
import com.rnett.knn.layers.PrimaryCapsules
import com.rnett.knn.layers.convolutional.Convolution2DLayer
import com.rnett.knn.layers.feedforeward.loss.LossLayer
import com.rnett.knn.layers.util.ActivationLayer
import com.rnett.knn.models.graph
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.impl.ActivationLReLU
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.nativeblas.NativeOpsHolder
import org.nd4j.nativeblas.Nd4jBlas
import kotlin.system.measureTimeMillis

object Runner {
    @JvmStatic
    fun main(args: Array<String>) {
        val nd4jBlas = Nd4j.factory().blas() as Nd4jBlas
        nd4jBlas.maxThreads = 6

        val instance = NativeOpsHolder.getInstance()
        val deviceNativeOps = instance.deviceNativeOps
        deviceNativeOps.setOmpNumThreads(6)

        val arr = Nd4j.linspace(1, 100, 100).reshape(10L, 10L)

        println(arr.toFloatMatrix().toList().map { it.toList() })

        val model = graph {

            optimizer = Optimizers.Adam

            val image = input("image", "depth" to 1, "height" to 28, "depth" to 28)

            image.reshape(1, 28, 28)

            println(image.shape)

            image {
                +Convolution2DLayer(16, 9.p2, 3.p2) { activation = ActivationLReLU() }
                println("Conv: ${image.shape}")
                +PrimaryCapsules(8, 8, 7.p2, 2.p2)
                println("PC: ${image.shape}")
                +CapsuleLayer(10, 16, 3)
                println("Caps: ${image.shape}")
                +CapsuleStrength()

                +ActivationLayer(Activations.Softmax)

                +LossLayer(Losses.NegativeLogLikelihood)
            }

            outputs(image)

        }.build()
        model.init()
        println(model.summary())


        val rngSeed = 12345
        val mnistTrain = MnistDataSetIterator(64, true, rngSeed)
        val mnistTest = MnistDataSetIterator(64, false, rngSeed)


        val nEpochs = 5

        (1..nEpochs).forEach { epoch ->
            val time = measureTimeMillis {
                try {
                    model.fit(mnistTrain)
                } catch (e: Exception) {
                    e.printStackTrace()
                }
            }

            println("Epoch " + epoch + " complete, took ${time / 1000} seconds")
        }

        val eval: Evaluation = model.evaluate(mnistTest)
        println(eval.stats())
    }
}