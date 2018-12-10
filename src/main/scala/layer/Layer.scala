package layer

import breeze.linalg.DenseMatrix


trait Layer {
  type T = Double
  def forward(x: DenseMatrix[T]): DenseMatrix[T]
}

class Sigmoid extends  Layer {
  override def forward(x: DenseMatrix[T]): DenseMatrix[T] = {
    x.map(calculate)
  }

  def calculate(x: T): T = {
    1.0 / (1.0 + Math.exp(-x))
  }
}

object Sigmoid {
  def apply(): Sigmoid = new Sigmoid()
}
