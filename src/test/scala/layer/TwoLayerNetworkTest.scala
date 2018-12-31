package layer

import breeze.linalg.{DenseMatrix, sum}
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

    val firstLoss = net.loss(x, t)
    net.backward()
    net.updateParams(0.1)

    val secondLoss = net.loss(x, t)
    net.backward()
    net.updateParams(0.1)

    val thirdLoss = net.loss(x, t)
    assert(sum(firstLoss) > sum(secondLoss))
    assert(sum(secondLoss) > sum(thirdLoss))

  }
}
