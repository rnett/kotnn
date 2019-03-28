package com.rnett.knn.layers

import com.rnett.knn.components.Activations
import com.rnett.knn.components.Losses
import com.rnett.knn.layers.convolutional.Convolution2DLayer
import com.rnett.knn.layers.convolutional.Subsampling2DLayer
import com.rnett.knn.layers.feedforeward.DenseLayer
import com.rnett.knn.layers.feedforeward.loss.LossLayer
import com.rnett.knn.layers.feedforeward.output.OutputLayer
import com.rnett.knn.layers.util.ActivationLayer
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

    /*@Test
    fun `same diff tests`() {
        val sd = SameDiff.create()

        fun test(a: SDVariable, b: SDVariable, vararg dims: List<Int>): SDVariable{

            if(dims.size == 1) {
                val axes = dims.first().first()

                val aShape = a.shape.toList()
                val bShape = b.shape.toList()

                val outShape = aShape.subList(0, aShape.size - axes) + bShape.subList(axes + 1, bShape.size)

                val newAShape = mutableListOf(aShape[0].toInt(), 1, 1)
                val newBShape = mutableListOf(bShape[0].toInt(), 1, 1)

                for(i in 1..axes){
                    newAShape[2] *= aShape[aShape.size - i].toInt()
                    newBShape[1] *= bShape[i].toInt()
                }
                for(i in 1 until (aShape.size - axes)){
                    newAShape[1] *= aShape[i].toInt()
                }
                for(i in 1 until (bShape.size - axes)){
                    newBShape[2] *= bShape[bShape.size - i].toInt()
                }

                val aR = sd.reshape(a, *newAShape.toIntArray())
                val bR = sd.reshape(b, *newBShape.toIntArray())

                val dot = sd.unstack(aR, 0, newAShape[0]).zip(sd.unstack(bR, 0, newBShape[0])).map {
                    sd.mmul(it.first, it.second, MMulTranspose(false, false, false))
                }.let{
                    sd.stack(0, *it.toTypedArray())
                }

                //TODO NPE
                //val dot = sd.stack(0, *sd.batchMmul(sd.unstack(aR, 0, newAShape[0]), sd.unstack(bR, 0, newBShape[0]))

                return sd.reshape(dot, *outShape.toLongArray())

            } else {
                if(dims[0].size != dims[1].size)
                    throw IllegalArgumentException("Must give the same number of axes for each input")

                val aShape = a.shape.toList()
                val bShape = b.shape.toList()

                val otherAs = (1 until aShape.size).filter { it !in dims[0] }
                val otherBs = (1 until bShape.size).filter { it !in dims[1] }

                val pA = sd.permute(a, *(listOf(0) + otherAs + dims[0]).toIntArray())
                val pB = sd.permute(b, *(listOf(0) + dims[1] + otherBs).toIntArray())

                return test(pA, pB, listOf(dims[0].size))
            }

        }
        val x = sd.`var`(
            Nd4j.create(
                arrayOf(
                    floatArrayOf(1F, 2F),
                    floatArrayOf(3F, 4F)
                )
            )
        )
        val y = sd.`var`(
            Nd4j.create(
                arrayOf(
                    floatArrayOf(5F, 6F),
                    floatArrayOf(7F, 8F)
                )
            )
        )

        println(sd.batchDot(x, listOf(2, 2), y, listOf(2, 2), 1, 1).eval().toIntVector().toList())


        val a = sd.one("a", 10, 160, 8)
        val b = sd.one("b", 10, 160, 16, 8)

        println(sd.batchDot(a, listOf(10, 160, 8), b, listOf(10, 160, 16, 8), 2, 3).eval().shape().toList())

    }*/

    @Test
    fun `test mnist capsnet`() {
        val model = graph {
            val image = input("image", "depth" to 1, "height" to 28, "depth" to 28)

            image.reshape(1, 28, 28)

            println(image.shape)

            image {
                +Convolution2DLayer(20, 9.p2) { activation = ActivationLReLU() }
                +PrimaryCapsules(10, 8, 5.p2, 4.p2)
                +CapsuleLayer(10, 16, 5)
                +CapsuleStrength()

//                +Convolution2DLayer(20, 5.p2) { activation = ActivationLReLU() }
//                +Subsampling2DLayer(kernelSize = 2.p2, stride = 2.p2)
//
//                +Convolution2DLayer(20, 5.p2) { activation = ActivationLReLU() }
//                +Subsampling2DLayer(kernelSize = 2.p2, stride = 2.p2)
//
//                flatten()
//
//                +DenseLayer(10) { activation = ActivationLReLU() }

                +ActivationLayer(Activations.Softmax)

                println(image.shape)
                +LossLayer(Losses.NegativeLogLikelihood)
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