allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

// NOTE: The custom build directory redirect to "../../build" has been removed.
// It caused the Kotlin incremental compiler to crash with
// "IllegalArgumentException: this and base files have different roots" on Windows
// because the redirected path crossed drive boundaries (D:\project vs C:\.gradle).
// Gradle now uses the default build/ directory inside each subproject, which keeps
// all paths on the same drive and allows the Kotlin daemon to compute relative paths.

subprojects {
    project.evaluationDependsOn(":app")
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
