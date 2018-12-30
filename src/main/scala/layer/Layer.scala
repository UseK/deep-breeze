package layer

import breeze.linalg.DenseMatrix
import breeze.linalg._
import breeze.numerics.{exp, log}


trait Layer {
  type T = Double
  def forward(x: DenseMatrix[T]): DenseMatrix[T]
  def backward(dOut: DenseMatrix[T]): DenseMatrix[T]
  def showParams()
}


class Sigmoid extends  Layer {
  var outCache: Option[DenseMatrix[T]] = None
  override def forward(x: DenseMatrix[T]): DenseMatrix[T] = {
    val forwarded = x.map(forwardCalculate)
    outCache = Option(forwarded)
    forwarded
  }


  def forwardCalculate(x: T): T = {
    1.0 / (1.0 + Math.exp(-x))
  }

  override def backward(dOut: DenseMatrix[T]): DenseMatrix[T] = {
    dOut *:* (1.0 - outCache.get) *:* outCache.get
  }

  override def showParams(): Unit = {}
}
object Sigmoid {
  def apply(): Sigmoid = new Sigmoid()
}


class Affine(w: DenseMatrix[Double], b: DenseMatrix[Double]) extends Layer {
  override def forward(x: DenseMatrix[T]): DenseMatrix[T] = {
    val dotted = (x * w)
    dotted+ b
  }

  override def backward(dOut: DenseMatrix[T]): DenseMatrix[T] = {
    // TODO Not Yet Implemented
    dOut
  }

  override def showParams(): Unit = {
    println(w)
    println(b)
  }
}

object Affine {
  def initByOne(nVec: Int, nSample: Int): Affine = {
    new Affine(
      DenseMatrix.ones[Double](nVec, nVec),
      DenseMatrix.zeros[Double](nSample, nVec)
    )
  }
}

class SoftmaxWithCrossEntropyError {
  var softmaxForwardCache: Option[DenseMatrix[T]] = None
  var tCache: Option[DenseMatrix[T]] = None

  type T = Double
  def forward(x: DenseMatrix[T], t: DenseMatrix[T]): DenseMatrix[T] = {
    tCache = Option(t)
    val forwarded = softmaxForward(x)
    softmaxForwardCache = Option(forwarded)
    crossEntropyErrorForward(forwarded, t)
  }

  def backward(): DenseMatrix[T] = {
    (softmaxForwardCache.get - tCache.get).map(v => v / tCache.get.rows)
  }

  def softmaxForward(x: DenseMatrix[T]): DenseMatrix[T] = {
    val softmaxMapped = (0 until x.rows).map { i =>
      softmax(x(i, ::).t)
    }
    DenseMatrix(softmaxMapped: _*)
  }

  def crossEntropyErrorForward(x: DenseMatrix[T],
                               t: DenseMatrix[T]): DenseMatrix[T] = {
    val ceeMapped = (0 until x.rows).map { i =>
      val sumOfCEE = crossEntropyError(x(i, ::).t, t(i, ::).t)
      sumOfCEE / t.rows
    }
    DenseMatrix(ceeMapped: _*)
  }

  def crossEntropyError(x: DenseVector[T],
                        t: DenseVector[T]): T = {
    -sum(t *:* log(x))
  }

  def softmax(row: DenseVector[T]): DenseVector[T] = {
    val cMinusEx = exp(row - max(row))
    cMinusEx / sum(cMinusEx)
  }
}

