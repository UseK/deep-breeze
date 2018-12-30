package layer

import breeze.linalg.DenseMatrix
import org.scalatest.FunSuite

class MultiLayerNetworkTest extends FunSuite {
  val affineSigmoid = MultiLayerNetwork.affineSigmoid(4, 2)
  val affineOnly = MultiLayerNetwork.affineOnly(4, 2)
  private val inputMatrix = DenseMatrix(
    (0.1, 0.2, 0.3, 0.4),
    (0.9, 0.8, 0.7, 0.6),
  )
  test("predict in affine only") {
    val result = affineOnly.predict(inputMatrix)
    val three = 3.0000000000000004
    val expected =  DenseMatrix(
      (1.0, 1.0,  1.0, 1.0),
      (three, three, three, three),
    )
    assert(result == expected)
  }
  test("predict in affine and sigmoid") {
    val result = affineSigmoid.predict(inputMatrix)
    val fromOne = 0.7310585786300049
    val fromThree = 0.9525741268224334
    val expected =  DenseMatrix(
      (fromOne, fromOne, fromOne, fromOne),
      (fromThree, fromThree, fromThree, fromThree),
    )
    assert(result == expected)
  }
  test("numerical gradient") {
    val correctLabel = DenseMatrix(
      (0.01, 0.04, 0.09, 0.16),
      (0.81, 0.64, 0.49, 0.36),
    )
    val affineSigmoid = MultiLayerNetwork.affineSigmoid(4, 2)
  }
}
