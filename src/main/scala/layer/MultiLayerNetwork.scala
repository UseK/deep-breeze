package layer

import breeze.linalg.{DenseMatrix, max, min, sum}

class MultiLayerNetwork(val layers: List[Layer]) {
  val lossFunction = new SoftmaxWithCrossEntropyError()
  type T = Double
  def loss(x: DenseMatrix[T], t: DenseMatrix[T]): DenseMatrix[T] = {
    val predicted = predict(x)
    lossFunction.forward(predicted, t)
  }
  def predict(x: DenseMatrix[T]): DenseMatrix[T] = {
    layers.foldLeft(x)((forwarded, layer) => layer.forward(forwarded))
  }
  def backward(): Unit = {
    var dOut = lossFunction.backward()
    layers.reverse.foreach { layer =>
      dOut = layer.backward(dOut)
    }
  }
  def updateParams(learningRate: Double): Unit = {
    layers.foreach {
      case l: ParamUpdatable => l.updateParams(learningRate)
      case _ => Unit
    }
  }
  def learn(x: DenseMatrix[Double],
            t: DenseMatrix[Double],
            nIter: Int,
            learningRate: Double): Unit = {
    (1 to nIter).foreach { i =>
      println(i)
      val ls = loss(x, t)
      println(sum(ls))
      println(max(ls))
      println(min(ls))
      println()
      backward()
      updateParams(learningRate)
    }
  }
}

object MultiLayerNetwork {
  def affineOnly(nVec: Int): MultiLayerNetwork = {
    new MultiLayerNetwork(
      List(Affine.initByOne(nVec))
    )
  }

  def affineSigmoid(nVec: Int): MultiLayerNetwork = {
    new MultiLayerNetwork(
      List(Affine.initByOne(nVec), Sigmoid())
    )
  }

  def twoLayerNetwork(nVec: Int): MultiLayerNetwork = {
    new MultiLayerNetwork(
      List(
        Affine.initByOne(nVec),
        Sigmoid(),
        Affine.initByOne(nVec),
      )
    )
  }
}
