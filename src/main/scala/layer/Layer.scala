package layer

import breeze.linalg.DenseMatrix
import breeze.linalg._


trait Layer {
  type T = Double
  def forward(x: DenseMatrix[T]): DenseMatrix[T]
  def backward(dOut: DenseMatrix[T]): DenseMatrix[T]
}


class Sigmoid extends  Layer {
  var out: Option[DenseMatrix[T]] = None
  override def forward(x: DenseMatrix[T]): DenseMatrix[T] = {
    val forwarded = x.map(forwardCalculate)
    out = Option(forwarded)
    forwarded
  }


  def forwardCalculate(x: T): T = {
    1.0 / (1.0 + Math.exp(-x))
  }

  override def backward(dOut: DenseMatrix[T]): DenseMatrix[T] = {
    dOut *:* (1.0 - out.get) *:* out.get
  }
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
    // Not Yet Implemented
    dOut
  }
}
