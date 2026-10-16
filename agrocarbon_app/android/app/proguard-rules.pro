# Core Flutter Bindings
-keep class io.flutter.app.** { *; }
-keep class io.flutter.plugin.**  { *; }
-keep class io.flutter.util.**  { *; }
-keep class io.flutter.view.**  { *; }
-keep class io.flutter.**  { *; }
-keep class plugins.flutter.io.**  { *; }

# Safely preserves line numbers masking production crash stacks natively!
-keepattributes *Annotation*
-keepattributes SourceFile,LineNumberTable

# Dio Network Hooks (Avoid R8 Stripping)
-keep class retrofit2.** { *; }
-dontwarn retrofit2.**
