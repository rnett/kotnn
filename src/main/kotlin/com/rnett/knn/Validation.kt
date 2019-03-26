package com.rnett.knn

import kotlin.properties.Delegates
import kotlin.properties.ReadWriteProperty
import kotlin.reflect.KMutableProperty0
import kotlin.reflect.KProperty
import kotlin.reflect.KProperty0

object P{
    operator fun invoke(d1: Int) = Param1(d1)
    operator fun invoke(d1: Int, d2: Int) = Param2(d1, d2)
    operator fun invoke(d1: Int, d2: Int, d3: Int) = Param3(d1, d2, d3)
    operator fun invoke(d1: Int, d2: Int, d3: Int, d4: Int) = Param4(d1, d2, d3, d4)
    operator fun invoke(d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int) = Param6(d1, d2, d3, d4, d5, d6)

    operator fun invoke(vararg data: Int) = data.toParams()
    operator fun invoke(data: List<Int>) = data.toParams()

    inline operator fun <reified T : NParams> get(data : NParams) = data.expandToType<T>()
    inline operator fun <reified T : NParams> get(vararg data: Int) = data.toParams().expandToType<T>()
    inline operator fun <reified T : NParams> get(data: List<Int>) = data.toParams().expandToType<T>()

}

sealed class NParams(val n: Int) {
    abstract fun toList(): List<Int>

    fun toIntArray() = toList().toIntArray()

    abstract fun expand(target: Int): NParams

    inline fun <reified T : NParams> expandToType() : T{
        val target = when(T::class){
            Param1::class -> 1
            Param2::class -> 2
            Param3::class -> 3
            Param4::class -> 4
            Param6::class -> 6
            else -> throw IllegalArgumentException("Unknown target type ${T::class}")
        }
        return expand(target) as T
    }

    val nonNegative get() = toList().all { it >= 0 }

    operator fun get(idx: Int) = toList()[idx]

}

data class Param1(val first: Int) : NParams(1) {
    override fun toList() = listOf(first)
    override fun expand(target: Int) =
        when (target) {
            1 -> this
            2 -> Param2(first, first)
            3 -> Param3(first, first, first)
            4 -> Param4(first, first, first, first)
            6 -> Param6(first, first, first, first, first, first)
            else -> throw IllegalArgumentException("Can't expand $n to $target")
        }

}

data class Param2(val first: Int, val second: Int) : NParams(2) {
    override fun toList() = listOf(first, second)
    override fun expand(target: Int) =
        when (target) {
            2 -> this
            4 -> Param4(first, first, second, second)
            6 -> Param6(first, second, first, second, first, second)
            else -> throw IllegalArgumentException("Can't expand $n to $target")
        }
}

data class Param3(val first: Int, val second: Int, val third: Int) : NParams(3) {
    override fun toList() = listOf(first, second, third)
    override fun expand(target: Int) =
        when (target) {
            3 -> this
            6 -> Param6(first, first, second, second, third, third)
            else -> throw IllegalArgumentException("Can't expand $n to $target")
        }
}

data class Param4(val first: Int, val second: Int, val third: Int, val fourth: Int) : NParams(4) {
    override fun toList() = listOf(first, second, third, fourth)
    override fun expand(target: Int) =
        throw IllegalArgumentException("Can't expand $n to $target")
}

data class Param6(val first: Int, val second: Int, val third: Int, val fourth: Int, val fifth: Int, val sixth: Int) :
    NParams(6) {
    override fun toList() = listOf(first, second, third, fourth, fifth, sixth)
    override fun expand(target: Int) =
        throw IllegalArgumentException("Can't expand $n to $target")
}

fun Iterable<Int>.toParams(): NParams {
    val list = toList()
    return when (count()) {
        1 -> Param1(list[0])
        2 -> Param2(list[0], list[1])
        3 -> Param3(list[0], list[1], list[2])
        4 -> Param4(list[0], list[1], list[2], list[3])
        5 -> Param6(list[0], list[1], list[2], list[3], list[4], list[5])
        else -> throw IllegalArgumentException("Can't make NParams of size ${count()}")
    }
}

fun IntArray.toParams() = toList().toParams()

fun paramsOf(vararg values: Int) = values.toList().toParams()

fun <T : NParams> nonNegativeParams(default: T) =
    Delegates.vetoable(default) { _, _, new -> new.toList().all { it >= 0 } }

fun <T : NParams> nonNegativeParams(vararg values: Int) =
    nonNegativeParams(values.toParams() as T)

fun <T : NParams?> nullableNonNegativeParams(default: T? = null) =
    Delegates.vetoable(default) { _, _, new -> new == null || new.toList().all { it >= 0 } }

fun <T : NParams?> nullableNonNegativeParams(vararg values: Int) =
    nullableNonNegativeParams(if(values.isEmpty()) null else values.toParams() as T)

inline operator fun <reified T : NParams> KMutableProperty0<T>.get(data: NParams){
    this.set(P[data])
}

inline operator fun <reified T : NParams>KMutableProperty0<T>.get(data: List<Int>){
    this[data.toParams()]
}

inline operator fun <reified T : NParams>KMutableProperty0<T>.get(vararg data: Int){
    this[data.toParams()]
}

fun <T> List<T>.enforceLength(length: Int, message: String = ""){
    if(size != length)
        throw IllegalArgumentException(message)
}

val <T> T.paired get() = Pair(this, this)

val Pair<Int, Int>.p2 get() = Param2(first, second)
val Pair<Int, Int>.p get() = p2


val Int.p get() = p1
val Int.p1 get() = Param1(this)
val Int.p2 get() = Param2(this, this)
val Int.p3 get() = Param3(this, this, this)
val Int.p4 get() = Param4(this, this, this, this)
val Int.p6 get() = Param6(this, this, this, this, this, this)

infix fun Int.x(other: Int) = this paramWith other
infix fun Int.paramWith(other: Int) = Param2(this, other)