package ken

import java.io.File

import breeze.linalg.DenseMatrix
import com.github.tototoshi.csv.CSVReader
import layer.MultiLayerNetwork
import nlp.StringEncoder

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
    val pairsBatch = pairs.slice(0, 100)
    val encoder = StringEncoder.fromStrings(pairs.map(_._1) ++ pairs.map(_._2))
    val x = DenseMatrix(pairsBatch.map(t=> encoder.encode(t._1)):_*)
    val t = DenseMatrix(pairsBatch.map(t=> encoder.encode(t._2)):_*)
    val nVec = x(0, ::).t.length
    println(nVec)
    println(encoder.characters.length)
    val net = MultiLayerNetwork.twoLayerNetwork(nVec, 2)
    net.learn(x, t, 10000, 0.000005)
  }

  private def readIntoLines(pathName: String) = {
    val reader = CSVReader.open(new File(pathName))
    val lines = reader.iterator.toList
    lines.map(items =>
      (items(3), items(4), items(5))
    ).distinct
  }
}
