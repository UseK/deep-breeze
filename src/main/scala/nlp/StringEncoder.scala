package nlp

import breeze.linalg.{DenseMatrix, DenseVector}

class StringEncoder(val characters: List[Char]) {
  private def toVectors(m: DenseMatrix[Double]): List[DenseVector[Double]] = {
    (0 until m.rows).map { i =>
      m(i, ::).t
    }.toList
  }
  private val identityMatrix = DenseMatrix.eye[Double](characters.length)
  private val char2Vector = toVectors(identityMatrix).zipWithIndex.map {t =>
    val (vec, i) = t
    (characters(i), vec)
  }.toMap
  def encode(c: Char): DenseVector[Double] = {
    char2Vector(c)
  }
}

object StringEncoder {
  def fromStrings(strings: List[String]): StringEncoder = {
    val characters = strings.flatten.iterator.toList.distinct.sorted
    new StringEncoder(characters)
  }
}
