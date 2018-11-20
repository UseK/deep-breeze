package nlp

import breeze.linalg._

case class Corpus(corpus: List[Int], wordToId: Map[String, Int]) {
  private val vocabSize = wordToId.size
  val idToWord: Map[Int, String] = wordToId.map(t => t._2 -> t._1)
  val idToCorpusIndices: Map[Int, List[Int]] = corpus.zipWithIndex.foldLeft(
    Map[Int, List[Int]]()) {(m, t) =>
    m.updated(t._1, t._2 :: m.getOrElse(t._1, List()))
  }

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
}

object Corpus {
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
}
