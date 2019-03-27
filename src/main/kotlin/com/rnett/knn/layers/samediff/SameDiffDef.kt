package com.rnett.knn.layers.samediff

import org.nd4j.autodiff.samediff.SDIndex
import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.SameDiff
import kotlin.properties.ReadOnlyProperty
import kotlin.reflect.KProperty

//TODO don't allow param refs for lambda layers
open class SameDiffLambdaDef(val SD: SameDiff) {
    operator fun SDVariable.unaryMinus(): SDVariable = this.neg()

    operator fun SDVariable.plus(other: SDVariable): SDVariable = this.add(other)
    operator fun SDVariable.plus(other: Number): SDVariable = this.add(other.toDouble())

    operator fun SDVariable.minus(other: SDVariable): SDVariable = this.sub(other)
    operator fun SDVariable.minus(other: Number): SDVariable = this.sub(other.toDouble())

    operator fun SDVariable.times(other: SDVariable): SDVariable = this.mul(other)
    operator fun SDVariable.times(other: Number): SDVariable = this.mul(other.toDouble())

    operator fun SDVariable.div(other: SDVariable): SDVariable = this.div(other)
    operator fun SDVariable.div(other: Number): SDVariable = this.div(other.toDouble())

    operator fun Number.plus(other: SDVariable): SDVariable = other.add(this.toDouble())
    operator fun Number.minus(other: SDVariable): SDVariable = other.rsub(this.toDouble())
    operator fun Number.times(other: SDVariable): SDVariable = other.mul(this.toDouble())
    operator fun Number.div(other: SDVariable): SDVariable = other.rdiv(this.toDouble())

    infix fun SDVariable.mmul(other: SDVariable): SDVariable = this.mmul(other)
    operator fun SDVariable.rem(other: SDVariable): SDVariable = this mmul other

    operator fun SDVariable.get(indexes: Iterable<Any?>) = this.get(*indexes.toList().toTypedArray())

    operator fun SDVariable.get(vararg indexes: Index?) = this[(indexes as Array<Any?>)]

    operator fun SDVariable.get(vararg indexes: Any?): SDVariable = this.get(
        *indexes.mapIndexed { index, it ->
            when (it) {
                is IntRange -> {
                    if (it.start >= 0 && it.endInclusive >= 0)
                        SDIndex.interval(it.start, it.endInclusive)

                    val end = if (it.endInclusive < 0)
                        this.shape[index].toInt() + it.endInclusive
                    else
                        it.endInclusive

                    val start = if (it.start < 0)
                        this.shape[index].toInt() + it.start
                    else
                        it.start

                    SDIndex.interval(start, end)
                }
                is Index -> {

                    val start = when {
                        it._start == null -> null
                        it._start < 0 -> this.shape[index].toInt() + it._start
                        else -> it._start
                    }

                    val end = when {
                        it._end == null -> null
                        it._end < 0 -> this.shape[index].toInt() + it._end
                        else -> it._end
                    }

                    SDIndex.interval(start, end)
                }
                is Int -> SDIndex.point(it.toLong())
                is SDIndex -> it
                else -> SDIndex.all()
            }
        }.toTypedArray()
    )
}

class SameDiffDef(SD: SameDiff, val params: Map<String, SDVariable>) : SameDiffLambdaDef(SD) {
    inner class SDVarDelegate internal constructor(val name: String?) : ReadOnlyProperty<Any?, SDVariable> {
        override fun getValue(thisRef: Any?, property: KProperty<*>): SDVariable {
            return params[name ?: property.name]!!
        }

    }

    operator fun provideDelegate(thisRef: Any?, property: KProperty<*>) =
        SDVarDelegate(null)

    operator fun Map<String, SDVariable>.provideDelegate(thisRef: Any?, property: KProperty<*>) =
        SDVarDelegate(null)

    val sdVar get() = SDVarDelegate(null)
    fun sdVar(name: String) = SDVarDelegate(name)
}

data class Index(val _start: Int?, val _end: Int?) : ClosedRange<Int>{
    override val start = _start ?: 0
    override val endInclusive = (_end ?: 1) - 1

    fun resolve(dimSize: Int): Pair<Int, Int> = Pair(
        when{
            _start == null -> 0
            _start < 0 -> dimSize + _start
            else -> _start
        },
        when{
            _end == null -> dimSize - 1
            _end < 0 -> (dimSize - 1) + _end
            else -> _end
        }
    )
}

fun ClosedRange<Int>.toIndex() = Index(this.start, this.endInclusive)

operator fun Int.rangeTo(other: Int) = Index(this, other)
operator fun Int?.rangeTo(other: Int) = Index(this, other)
operator fun Int.rangeTo(other: Int?) = Index(this, other)
operator fun Int?.rangeTo(other: Int?) = Index(this, other)