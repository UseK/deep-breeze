package nlp

import breeze.linalg.{DenseMatrix, DenseVector}

class StringEncoder(val characters: List[Char]) {
  val nChars = characters.length
  private def toVectors(m: DenseMatrix[Double]): List[DenseVector[Double]] = {
    (0 until m.rows).map { i =>
      m(i, ::).t
    }.toList
  }
  private val identityMatrix = DenseMatrix.eye[Double](nChars)
  private val char2Vector = toVectors(identityMatrix).zipWithIndex.map {t =>
    val (vec, i) = t
    (characters(i), vec)
  }.toMap
  def encode(c: Char): DenseVector[Double] = {
    char2Vector(c)
  }
  def encode(s: String): DenseVector[Double] = {
    s.iterator.map(encode).reduce {(concated, vec) =>
      DenseVector.vertcat(concated, vec)
    }
  }
  def decode(vec: DenseVector[Double]): String = {
    vec.toArray.grouped(nChars).toList.map {vec4char =>
      val maxIndex = vec4char.indexOf(vec4char.max)
      characters(maxIndex)
    }.mkString
  }
}

object StringEncoder {
  def fromStrings(strings: List[String]): StringEncoder = {
    val characters = strings.flatten.iterator.toList.distinct.sorted
    new StringEncoder(characters)
  }
}
