package layer

import breeze.linalg._
import org.scalatest.FunSuite

class MatrixTest extends FunSuite {
  private val mat = DenseMatrix(
    (1.0, 2.0, 3.0),
    (4.0, 5.0, 6.0),
  )
  test("broadcast") {
    val bias = DenseVector(100.0, 200.0, 300.0)
    val result = mat(*, ::) + bias
    val expected = DenseMatrix(
      (101.0, 202.0, 303.0),
      (104.0, 205.0, 306.0),
    )
    assert(result == expected)
  }

  test("sum") {
    assert(sum(mat(::, *)).t == DenseVector(5.0, 7.0, 9.0))
  }

  test("broadcast mult") {
    assert(mat *:* 0.1 == DenseMatrix(
      (0.1, 0.2, 0.30000000000000004),
      (0.4, 0.5, 0.6000000000000001),
    )
    )
  }
}
