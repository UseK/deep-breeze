package nlp

import breeze.linalg._

case class Corpus(corpus: List[Int], wordToId: Map[String, Int]) {
  private val vocabSize = wordToId.size
  val idToWord: Map[Int, String] = wordToId.map(t => t._2 -> t._1)

  def createCoMatrix(windowSize: Int = 1): DenseMatrix[Int] = {
    DenseMatrix.zeros[Int](vocabSize, vocabSize)
  }
}

object Corpus {
  def preprocess(text: String): Corpus = {
    val lowered = text.toLowerCase
    val replaced = lowered.replace(".", " .")
    val words = replaced.split(' ').toList
    var wordToId = Map.empty[String, Int]
    var idCounter = 0
    words.foreach { word =>
      if (!wordToId.contains(word)) {
        wordToId = wordToId.updated(word, idCounter)
        idCounter += 1
      }
    }
    val corpus = words.map(w => wordToId(w))
    new Corpus(corpus, wordToId)
  }
}
