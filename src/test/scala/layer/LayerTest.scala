package layer

import breeze.linalg.DenseMatrix
import org.scalatest.FunSuite

class SigmoidTest extends FunSuite {
  type T = Double
  val eps = 0.001
  val sigmoid = Sigmoid()

  test("calculate") {
    assertAround(sigmoid.calculate(0.0),0.5)
    assertAround(sigmoid.calculate(1.0),0.731)
  }

  test("forward") {
    val result = sigmoid.forward(DenseMatrix(
      (1.0, 2.0, 3.0),
      (4.0, 5.0, 6.0)
    ))
    val expected = DenseMatrix(
      (0.7310585786300049, 0.8807970779778823,  0.9525741268224334),
      (0.9820137900379085, 0.9933071490757153, 0.9975273768433653)
    )
    assert(result == expected)
  }

  def assertAround(result: T, expected: T): Unit = {
    assert(result <= expected + eps)
    assert(result >= expected - eps)
  }
}
