package ken

import java.io.File

import com.github.tototoshi.csv.CSVReader

class Converter {
}

object Converter {
  def run(): Unit = {
    val linesWith = readIntoLines("data/KEN_ALL_with.CSV")
    val linesWithout = readIntoLines("data/KEN_ALL_without.CSV")
    val pairs = linesWith.zip(linesWithout).flatMap { t =>
      val (w, wo) = t
      List((w._1, wo._1), (w._2, wo._2), (w._3, wo._3))
    }.distinct
    pairs.foreach(println)
    println(pairs.length)
    println(pairs.count(t => t._1 != t._2))
    pairs.filter(t => t._1 != t._2).foreach(println)
  }

  private def readIntoLines(pathName: String) = {
    val reader = CSVReader.open(new File(pathName))
    val lines = reader.iterator.toList
    lines.map(items =>
      (items(3), items(4), items(5))
    ).distinct
  }
}
