package nlp

import org.scalatest.FlatSpec
import org.scalatest.Matchers

class CorpusSpec extends  FlatSpec with Matchers {
  "The Corpus object" should "preprocess" in {
    val text = "You say goodbye and I say hello."
    val corpus = Corpus.preprocess(text)
    val expected = Corpus(
      List(0, 1, 2, 3, 4, 1, 5, 6),
      Map(
        "." -> 6, "i" -> 4, "you" -> 0,
        "goodbye" -> 2, "say" -> 1,
        "hello" -> 5, "and" -> 3)
    )
    corpus shouldEqual expected
  }


}
