## TensorFlow Genetic Programming

A first attempt at implementing genetic programming with tensor operations.


## Design

Programs are represented as integer tensors with vaguely Fortran-like semantics. Each program has 
two stacks -- one for floating point values and one for integers. Likewise, inputs, outputs, and 
memory are represented as arrays with read(/write) heads, each with dual instances for floating 
point versus integer values. The execution loop computes all possible instructions in parallel, and 
then selects the path appropriate to each program based on the instruction appearing next in that
program.
