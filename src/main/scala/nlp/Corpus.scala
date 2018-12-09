package nlp

import breeze.linalg._
import breeze.linalg.functions.cosineDistance

case class Corpus(corpus: List[Int], wordToId: Map[String, Int]) {
  private val vocabSize = wordToId.size
  val idToWord: Map[Int, String] = wordToId.map(t => t._2 -> t._1)
  val idToCorpusIndices: Map[Int, List[Int]] = corpus.zipWithIndex.foldLeft(
    Map[Int, List[Int]]()) {(m, t) =>
    m.updated(t._1, t._2 :: m.getOrElse(t._1, List()))
  }
  val coMatrix = createCoMatrix()

  def createCoMatrix(windowSize: Int = 1): DenseMatrix[Int] = {
    val matrix = DenseMatrix.zeros[Int](vocabSize, vocabSize)
    idToCorpusIndices.foreach{t =>
      val (wordId, indices) = t
      indices.foreach { idx =>
        val leftIdx = idx - windowSize
        if (leftIdx >= 0) {
          matrix(wordId, corpus(leftIdx)) += 1
        }
        val rightIdx = idx + windowSize
        if (rightIdx < corpus.size) {
          matrix(wordId, corpus(rightIdx)) += 1
        }
      }
    }
    matrix
  }

  def createPPMIMatrix(): DenseMatrix[Double] = {
    DenseMatrix(
      Range(0, vocabSize).toList.map { i =>
        Range(0, vocabSize).toList.map(j => ppmi(i, j))
      }: _*
    )
  }

  val N = sum(coMatrix)

  def ppmi(i: Int, j: Int) = {
    val xOccurrence = id2NOccurrence(i)
    val yOccurrence = id2NOccurrence(j)
    val co = coMatrix(i, j)
    Corpus.ppmi(co, xOccurrence, yOccurrence, N)
  }

  def id2NOccurrence(id: Int): Int = {
    sum(coMatrix(id, ::))
  }
}

object Corpus {
  def showSVD(matrix: DenseMatrix[Double]) = {
    //val s, v, d = svd(matrix)
    println(svd(matrix))
    //println(s)
    //println()
    //println(v)
    //println()
    //println(d)
  }

  def preprocess(text: String): Corpus = {
    val lowered = text.toLowerCase
    val replaced = lowered.replace(".", " .")
    val words = replaced.split(' ').toList
    val wordToId = genWordToId(words)
    val corpus = words.map(w => wordToId(w))
    new Corpus(corpus, wordToId)
  }

  def genWordToId(words: List[String]): Map[String, Int] = {
    var wordToId = Map.empty[String, Int]
    var idCounter = 0
    words.foreach { word =>
      if (!wordToId.contains(word)) {
        wordToId = wordToId.updated(word, idCounter)
        idCounter += 1
      }
    }
    wordToId
  }

  val eps: Double = Double.MinPositiveValue

  def ppmi(xy: Int, x: Int, y:Int, N: Int): Double = {
    Math.max(0.0, pmi(xy, x, y, N))
  }

  /**
    * PointWise Mutual Information
    * @param xy Number of Co-occurrence
    * @param x Number of x occurrence
    * @param y Number of y occurrence
    * @param N Number of courpus
    * @return
    */
  def pmi(xy: Int, x: Int, y:Int, N: Int): Double = {
    val antilogarithm = (xy * N) / (x * y + eps)
    log2(antilogarithm)
  }

  val lnOf2: Double = Math.log(2)
  def log2(x: Double): Double = {
    Math.log(x) / lnOf2
  }
}
