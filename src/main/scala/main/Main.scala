package main
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.client.{FeedMap, Session}
import org.platanios.tensorflow.api.ops.variables.ConstantInitializer
import org.platanios.tensorflow.api.tf


object Main extends App {
  TensorFlowExample.sessionRun()
}

object TensorFlowExample {
  def sessionRun(): Unit = {
    val x = tf.placeholder[Float](shape=Shape(1))
    val w = tf.variable[Float]("w", Shape(1), ConstantInitializer(Tensor(2.0f)))
    val z = x + w
    val session = Session()
    session.run(targets = tf.globalVariablesInitializer())
    val result = session.run(
      targets = z,
      feeds = Map(x -> Tensor(0.6f)))
    println(result)
  }

  def helloTensorFlow = {
    val t1 = Tensor(1.2, 4.5)
    val t2 = Tensor(-0.2, 1.1)
    println(t1)
    println(t2)
    println(t1 + t2 == Tensor(1.0, 5.6))

    val inputs = tf.placeholder[Float](Shape(-1, 10))
    val outputs = tf.placeholder[Float](Shape(-1, 10))
    val predictions = tf.nameScope("Linear") {
      val weights = tf.variable[Float](
        "weights",
        Shape(10, 1),
        tf.ZerosInitializer)
      tf.matmul(inputs, weights)
    }
    val loss = tf.sum(tf.square(predictions - outputs))
    val optimizer = tf.train.AdaGrad(1.0f)
    val trainOp = optimizer.minimize(loss)

    println(trainOp)
  }
}
