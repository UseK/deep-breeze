package layer

import breeze.linalg.DenseMatrix
import org.scalatest.FunSuite

class SigmoidTest extends FunSuite {
  type T = Double
  val eps = 0.001
  val sigmoid = Sigmoid()

  test("calculate") {
    assertAround(sigmoid.forwardCalculate(0.0),0.5)
    assertAround(sigmoid.forwardCalculate(1.0),0.731)
  }

  test("forward") {
    val result = sigmoid.forward(DenseMatrix(
      (1.0, 2.0, 3.0),
      (4.0, 5.0, 6.0)
    ))
    val expected = DenseMatrix(
      (0.7310585786300049, 0.8807970779778823, 0.9525741268224334),
      (0.9820137900379085, 0.9933071490757153, 0.9975273768433653)
    )
    assert(result == expected)
  }

  test("backward") {
    val grad = 0.0001
    val dx = sigmoid.backward(
      DenseMatrix(
        (grad, grad, grad),
        (grad, grad, grad),
      )
    )
    assert(dx.data(0) > dx.data(1))
    assert(dx.data(4) > dx.data(5))
  }

  def assertAround(result: T, expected: T): Unit = {
    assert(result <= expected + eps)
    assert(result >= expected - eps)
  }
}

class AffineTest extends FunSuite {
  test("forward") {
    val src = DenseMatrix(
      (1.0, 2.0, 3.0, 4.0),
      (5.0, 6.0, 7.0, 8.0),
    )
    val affine = new Affine(
      w = DenseMatrix(
        (1.0, 2.0, 10.0, 10.0),
        (3.0, 4.0, 10.0, 10.0),
        (5.0, 6.0, 10.0, 10.0),
        (7.0, 8.0, 10.0, 10.0),
      ),
      b = DenseMatrix(
        (100.0, 200.0, 300.0, 400.0),
        (500.0, 600.0, 700.0, 800.9),
      )
    )

    val result = affine.forward(src)
    val expected = DenseMatrix(
      (150.0, 260.0, 400.0, 500.0),
      (614.0, 740.0, 960.0, 1060.9)
    )
    assert(result == expected)
  }
}

class SoftmaxWithCrossEntropyErrorTest extends FunSuite {
  test("softmaxForward") {
    val inputMatrix = DenseMatrix(
      (0.1, 0.2, 0.3, 0.4),
      (0.9, 0.8, 0.7, 0.6),
    )
    val softmax = new SoftmaxWithCrossEntropyError()
    val result = softmax.softmaxForward(inputMatrix)
    val expected = DenseMatrix(
      (0.21383822036598443,  0.23632778232153764,
        0.26118259215507555,  0.28865140515740234),
      (0.28865140515740234,  0.2611825921550756,
        0.23632778232153764,  0.21383822036598443)
    )
    assert(result == expected)
  }

  test("crossEntropyErrorForward") {
    val inputMatrix = DenseMatrix(
      (0.21383822036598443,  0.23632778232153764,
        0.26118259215507555,  0.28865140515740234),
      (0.28865140515740234,  0.2611825921550756,
        0.23632778232153764,  0.21383822036598443)
    )
    val correctSet = DenseMatrix(
      (1.0, 0.0, 0.0, 0.0),
      (1.0, 0.0, 0.0, 0.0),
    )
    val softmax = new SoftmaxWithCrossEntropyError()
    val result = softmax.crossEntropyErrorForward(inputMatrix, correctSet)
    val expected = DenseMatrix(
      (0.7712677647275814),
      (0.6212677647275814),
    )
    assert(result == expected)
  }
}

