package main
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.tf

object Main extends App {
  val t1 = Tensor(1.2, 4.5)
  val t2 = Tensor(-0.2, 1.1)
  println(t1)
  println(t2)
  println(t1 + t2 == Tensor(1.0, 5.6))

  val inputs      = tf.placeholder[Float](Shape(-1, 10))
  val outputs     = tf.placeholder[Float](Shape(-1, 10))
  val predictions = tf.nameScope("Linear") {
    val weights = tf.variable[Float]("weights", Shape(10, 1), tf.ZerosInitializer)
    tf.matmul(inputs, weights)
  }
  val loss        = tf.sum(tf.square(predictions - outputs))
  val optimizer   = tf.train.AdaGrad(1.0f)
  val trainOp     = optimizer.minimize(loss)
  println(trainOp)
}
