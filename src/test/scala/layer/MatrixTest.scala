package layer

import breeze.linalg._
import org.scalatest.FunSuite

class MatrixTest extends FunSuite {
  test("broadcast") {
    val mat = DenseMatrix(
      (1.0, 2.0, 3.0),
      (4.0, 5.0, 6.0),
    )
    val bias = DenseVector(100.0, 200.0, 300.0)
    val result = mat(*, ::) + bias
    val expected = DenseMatrix(
      (101.0, 202.0, 303.0),
      (104.0, 205.0, 306.0),
    )
    assert(result == expected)
  }
}
