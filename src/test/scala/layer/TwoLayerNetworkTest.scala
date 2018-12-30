package layer

import breeze.linalg.DenseMatrix
import org.scalatest.FunSuite

class TwoLayerNetworkTest extends FunSuite {
  val x = DenseMatrix(
    (0.1, 0.2, 0.3, 0.4),
    (0.9, 0.8, 0.7, 0.6),
  )
  val t = DenseMatrix(
    (0.0, 0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0, 0.0),
  )

  test("backward") {
    val net = MultiLayerNetwork.twoLayerNetwork(4, 2)
    net.loss(x, t)
    net.layers(0).showParams()
    net.backward()
  }
}
