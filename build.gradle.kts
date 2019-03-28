import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

buildscript {

    repositories {
        maven("https://dl.bintray.com/kotlin/kotlin-eap")
        mavenCentral()
    }

    dependencies {
        classpath("org.jetbrains.kotlin:kotlin-gradle-plugin:1.3.30-eap-45")
    }
}

repositories {
    mavenCentral()
}

plugins {
    java
    id("com.github.ben-manes.versions") version "0.20.0"
}

apply{
    plugin("kotlin")
}

group = "com.rnett.knn"
version = "1.0-SNAPSHOT"

repositories {
    maven("https://dl.bintray.com/kotlin/kotlin-eap")
    mavenCentral()
    mavenLocal()
    jcenter()
    maven( "https://dl.bintray.com/soywiz/soywiz")
    maven( "https://jitpack.io")
    maven("https://oss.sonatype.org/content/repositories/snapshots")
}

//val dl4j_version  = "1.0.0-beta3"
val dl4j_version = "1.0.0-SNAPSHOT"
val slf4j_version = "1.7.25"

dependencies {
    implementation(kotlin("stdlib-jdk8", "1.3.30-eap-45"))
    testCompile("junit", "junit", "4.12")
    compile(kotlin("reflect"))

    implementation("org.deeplearning4j:deeplearning4j-core:$dl4j_version")
    implementation("org.nd4j:nd4j-native-platform:$dl4j_version")

    implementation("org.deeplearning4j:deeplearning4j-cuda-10.0:$dl4j_version")
    implementation("org.nd4j:nd4j-cuda-10.0:$dl4j_version:windows-x86_64")

    //if(dl4j_version == "1.0.0-beta3")
    //    implementation("org.deeplearning4j:deeplearning4j-ui_2.10:1.0.0-beta3")

    implementation("org.slf4j:slf4j-simple:$slf4j_version")
    implementation("org.slf4j:slf4j-api:$slf4j_version")
}

configure<JavaPluginConvention> {
    sourceCompatibility = JavaVersion.VERSION_1_8
}
tasks.withType<KotlinCompile> {
    kotlinOptions.jvmTarget = "1.8"
}