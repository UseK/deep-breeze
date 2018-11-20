package example

import breeze.linalg._
import nlp.Corpus

object Hello extends Greeting with App {
  val corpus = Corpus.preprocess("You say goodbye and I say hello.")
  println(corpus)
}


object PracticeBreeze {
  def broadcast(): Unit = {
    val A = DenseMatrix(
      (1, 2, 3),
      (3, 4, 6))
    println(A(::, *) * 10)
    println()
    println(A(::, *) * DenseVector(10, 100))
    println()
    println(A(*, ::) * DenseVector(10, 100, 1000)) // equivalent to numpy
  }

  def dotProduct(): Unit = {
    val w = DenseMatrix(
      (1.0, 2.0, 3.0),
      (4.0, 5.0, 6.0))
    val x = DenseMatrix(
      (0.0, 1.0, 2.0),
      (3.0, 4.0, 5.0))
    println(w.t * x)
  }

  def zeros(): Unit = {
    val five_zeros = DenseVector.zeros[Double](5)
    println(five_zeros)
    val m_zeros = DenseMatrix.zeros[Double](4, 6)
    println(m_zeros)
  }

  def elementWise(): Unit = {
    val w = DenseMatrix(
      (1, 2, 3),
      (4, 5, 6))
    val x = DenseMatrix(
      (0, 1, 2),
      (3, 4, 5))
    println(w + x)
    println(w *:* x)
  }
}

trait Greeting {
  lazy val greeting: String = "hello"
}
