package com.rnett.knn.layers

import com.rnett.knn.P
import com.rnett.knn.layers.convolutional.Convolution2DLayer
import com.rnett.knn.layers.convolutional.Subsampling2DLayer
import com.rnett.knn.layers.feedforeward.DenseLayer
import com.rnett.knn.layers.feedforeward.output.OutputLayer
import com.rnett.knn.layers.sizing.CollateDimensionLayer
import com.rnett.knn.layers.sizing.DistributeDimensionLayer
import com.rnett.knn.layers.sizing.PermuteLayer
import com.rnett.knn.models.ComplexNetwork
import com.rnett.knn.models.FlattenVertex
import com.rnett.knn.models.ReshapeVertex
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.graph.ComputationGraph
import org.junit.Test
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.impl.ActivationReLU
import org.nd4j.linalg.activations.impl.ActivationSoftmax
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.impl.LossNegativeLogLikelihood
import kotlin.system.measureTimeMillis

class ConvTests {

    @Test
    fun `test mnist`() {

        val graph = ComplexNetwork {

            optimizer = Updater.SGD.iUpdaterWithDefaultConfig

            input("image", 3, 4, 5)

            //+ReshapeVertex(1, 28, 28)

            //+DistributeDimensionLayer(2, name = "test1")
            //+CollateDimensionLayer(2, 28, name = "test2")

            !distributedOver(2){
                Unit
            }

//            +Convolution2DLayer(20, kernelSize = P(5, 5)) { activation = ActivationReLU() }
//            +Subsampling2DLayer(kernelSize = P(2, 2), stride = P(2, 2))
//
//            +Convolution2DLayer(20, kernelSize = P(5, 5)) { activation = ActivationReLU() }
//            +Subsampling2DLayer(kernelSize = P(2, 2), stride = P(2, 2))
//
//            +FlattenVertex()
//            +DenseLayer(200) { activation = ActivationReLU() }
//
//            !+OutputLayer(10, loss = LossNegativeLogLikelihood()) {
//                activation = ActivationSoftmax()
//            }
        }.makeGraph()

        /*val out = graph.vertices["test1"]!!.getOutputType(0, InputType.convolutional(1, 28, 28)).getShape(true).toList()
        println(out)

        val out2 = graph.vertices["test2"]!!.getOutputType(0, InputType.recurrent(1, 28)).getShape(true).toList()
        println(out2)*/

        val model = ComputationGraph(graph)
        model.init()

//        val uiServer = UIServer.getInstance()
//        val statsStorage = InMemoryStatsStorage()
//        uiServer.attach(statsStorage)
//        model.setListeners(StatsListener(statsStorage))

        //println(model.summary())

        val rngSeed = 12345
        val mnistTrain = MnistDataSetIterator(64, true, rngSeed)
        val mnistTest = MnistDataSetIterator(64, false, rngSeed)

        val scaler: DataNormalization = ImagePreProcessingScaler(0.0, 1.0)
        scaler.fit(mnistTrain)

        mnistTrain.preProcessor = scaler
        mnistTest.preProcessor = scaler


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
}