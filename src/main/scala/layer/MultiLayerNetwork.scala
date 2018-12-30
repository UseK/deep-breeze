package layer

import breeze.linalg.DenseMatrix

class MultiLayerNetwork(val layers: List[Layer]) {
  type T = Double
  def predict(x: DenseMatrix[T]): DenseMatrix[T] = {
    layers.foldLeft(x)((forwarded, layer) => layer.forward(forwarded))
  }
}

object MultiLayerNetwork {
  def affineOnly(nVec: Int, nSample: Int): MultiLayerNetwork = {
    new MultiLayerNetwork(
      List(Affine.initByOne(nVec, nSample))
    )
  }

  def affineSigmoid(nVec: Int, nSample: Int): MultiLayerNetwork = {
    new MultiLayerNetwork(
      List(Affine.initByOne(nVec, nSample), Sigmoid())
    )
  }
}
