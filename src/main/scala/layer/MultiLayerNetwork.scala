package layer

import breeze.linalg.DenseMatrix

class MultiLayerNetwork(val layers: List[Layer]) {
  type T = Double
  def predict(x: DenseMatrix[T]): DenseMatrix[T] = {
    layers.foldLeft(x)((forwarded, layer) => layer.forward(forwarded))
  }
}
