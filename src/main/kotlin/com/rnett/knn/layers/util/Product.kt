package com.rnett.knn.layers.util

fun Iterable<Int>.product() = reduce { acc, t -> acc * t }
fun Iterable<Long>.product() = reduce { acc, t -> acc * t }
fun Iterable<Float>.product() = reduce { acc, t -> acc * t }
fun Iterable<Double>.product() = reduce { acc, t -> acc * t }