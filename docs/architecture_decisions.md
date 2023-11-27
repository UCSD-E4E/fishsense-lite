# Architecture Decisions

## System Requirements
* Multiarchitecture
    * aarch64
    * x86-64
* Linux
    * Ubuntu 22.04
* Performant
    * Can support CUDA

## Language Choice
In order to meet the above requirements, we aim to use an application language to write the UI and a system language for the backend.

### Front-End
#### Electron (JavaScript/TypeScript)
This technology would require hosting a web server to interact with our system language.  This would not satisfy our performance requirements.

#### Flutter (Dart)
This technology is an option.

#### JavaFX (Java)
This technology is now in community support and appears to be abandoned by Oracle.  Would not recommend.

#### WPF (C#)
This technology is Windows only.  Does not meet the cross-platform requirement.

#### Avalonia (C#)
This is a cross-platform, vector-based UI.  This is an option.  As C# is Java-like, it is the recommended go forward.

### Back-End
#### C
This would work, but may be harder to express the pipeline we wish to represent.

#### C++
This would be more expressive than C.  CMake allows us to support cross-platform.  Library management is difficult.

#### Rust
This would also be more expressive than C.  It includes memory safety.  Cargo solves library management and cross-platform.

### Final Decision
Given the above information, we will leverage Avalonia with C# as our frontend and Rust as our backed.  We will look to Seq's example of how to make this work correctly.