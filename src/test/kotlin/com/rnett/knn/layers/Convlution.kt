package com.rnett.knn.layers

import com.rnett.knn.layers.convolutional.Convolution2DLayer
import com.rnett.knn.layers.convolutional.Subsampling2DLayer
import com.rnett.knn.layers.feedforeward.DenseLayer
import com.rnett.knn.layers.feedforeward.output.OutputLayer
import com.rnett.knn.models.graph
import com.rnett.knn.p2
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.junit.Test
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.impl.ActivationLReLU
import org.nd4j.linalg.activations.impl.ActivationSoftmax
import org.nd4j.linalg.lossfunctions.impl.LossNegativeLogLikelihood
import kotlin.system.measureTimeMillis


class ConvTests {

    @Test
    fun `test mnist`() {

        val model = graph {
            val image = input("image", "depth" to 1, "height" to 28, "depth" to 28)

            image.reshape(1, 28, 28)

            image {
                +Convolution2DLayer(20, 5.p2) { activation = ActivationLReLU() }
                +Subsampling2DLayer(kernelSize = 2.p2, stride = 2.p2)

                +Convolution2DLayer(20, 5.p2) { activation = ActivationLReLU() }
                +Subsampling2DLayer(kernelSize = 2.p2, stride = 2.p2)

                flatten()

                +DenseLayer(200) { activation = ActivationLReLU() }

                +OutputLayer(10, loss = LossNegativeLogLikelihood()) { activation = ActivationSoftmax() }
            }

            outputs(image)
        }.build()
        model.init()


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
        println(model.summary())

        val eval: Evaluation = model.evaluate(mnistTest)
        println(eval.stats())

    }

//    @Test
//    fun `test mnist java`() {
//
//        val conf = NeuralNetConfiguration.Builder()
//            .updater(Updater.SGD.iUpdaterWithDefaultConfig)
//            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//            .list()
//            .layer(0, ConvolutionLayer.Builder()
//                .kernelSize(5, 5).nOut(20).activation(ActivationLReLU())
//                .build())
//            .layer(1, SubsamplingLayer.Builder()
//                .kernelSize(2, 2).stride(2 ,2)
//                .build())
//            .layer(2, ConvolutionLayer.Builder()
//                .kernelSize(5, 5).nOut(20).activation(ActivationLReLU())
//                .build())
//            .layer(3, SubsamplingLayer.Builder()
//                .kernelSize(2, 2).stride(2 ,2)
//                .build())
//            .layer(4, org.deeplearning4j.nn.conf.layers.DenseLayer.Builder()
//                .nOut(200).activation(ActivationLReLU())
//                .build())
//            .layer(5, org.deeplearning4j.nn.conf.layers.OutputLayer.Builder()
//                .nOut(10).activation(ActivationSoftmax()).lossFunction(LossNegativeLogLikelihood())
//                .build())
//            .setInputType(InputType.convolutionalFlat(28, 28, 1)).build()
//
//        val model = MultiLayerNetwork(conf)
//
//        //model.init()
//
//        val rngSeed = 12345
//        val mnistTrain = MnistDataSetIterator(64, true, rngSeed)
//        val mnistTest = MnistDataSetIterator(64, false, rngSeed)
//
//
//        val nEpochs = 5
//
//        (1..nEpochs).forEach { epoch ->
//            val time = measureTimeMillis {
//                try {
//
//                    model.fit(mnistTrain)
//                } catch (e: Exception) {
//                    e.printStackTrace()
//                }
//            }
//
//            println("Epoch " + epoch + " complete, took ${time / 1000} seconds")
//        }
//        println(model.summary())
//
//        val eval: Evaluation = model.evaluate(mnistTest)
//        println(eval.stats())
//
//    }

}