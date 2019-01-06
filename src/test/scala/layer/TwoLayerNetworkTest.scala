package layer

import breeze.linalg.{DenseMatrix, sum}
import org.scalatest.FunSuite

class TwoLayerNetworkTest extends FunSuite {
  private val x = DenseMatrix(
    (0.1, 0.2, 0.3, 0.4),
    (0.9, 0.8, 0.7, 0.6),
  )
  private val t = DenseMatrix(
    (0.0, 0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0, 0.0),
  )
  private val x2 = DenseMatrix(
    (0.1, 5.0, 3.0, 2.0),
    (0.8, 0.7, 5.9, 0.6),
  )
  private val t2 = DenseMatrix(
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
  )

  test("backward") {
    val net = MultiLayerNetwork.twoLayerNetwork(4)

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
  test("learn") {
    val net = MultiLayerNetwork.twoLayerNetwork(4)
    net.learn(x, t, 10000, 0.1)
    val expected = DenseMatrix(
      (-0.37807558581455547,  -1.5461798782345912,
        -1.5461798782345912,  6.004984427950525),
      (11.273705092086715, -1.0531644743138475,
        -1.0531644743138475,  4.471575126911721)
    )
    assert(net.predict(x) == expected)
  }
  test("learn2") {
    val net = MultiLayerNetwork.twoLayerNetwork(4)
    net.learn(x2, t2, 100000, 1.0)
    val expected = DenseMatrix(
      (-2.717025033381218,  16.752136959563572,  4.565931421266416, -2.717025033381218),
      (-3.2041079379467243, -2.5235010458222322, 8.983374687103948, -3.2041079379467243)
    )
    assert(net.predict(x2) == expected)
  }
}
