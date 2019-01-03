package nlp

import breeze.linalg.DenseVector
import org.scalatest.FunSuite

class StringEncoderTest extends FunSuite {
  val sVec = StringEncoder.fromStrings(List("aiee", "foo"))
  test("characters") {
    val expected = List('a', 'e', 'f', 'i', 'o', ' ')
    assert(sVec.characters == expected)
  }
  test("encode char") {
    assert(sVec.encode('e') == DenseVector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0))
  }
  test("encode string") {
    assert(sVec.encode("ae") ==
      DenseVector(
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
      ))
  }
  test("decoding") {
    val vec = DenseVector(
      1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    )
    assert(sVec.decode(vec) == "aiee")
  }
}
