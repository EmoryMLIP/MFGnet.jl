# Multiple Dispatch in Julia

Multiple dispatch is Julia's core paradigm for organizing code. Understanding it is essential for writing idiomatic, performant Julia code.

## What is Multiple Dispatch?

Multiple dispatch selects which method to call based on the runtime types of **all** arguments, not just the first one (unlike single dispatch in OOP).

```julia
# Define multiple methods for the same function
collide(x::Rock, y::Rock) = "rocks collide"
collide(x::Rock, y::Paper) = "paper covers rock"
collide(x::Paper, y::Rock) = "paper covers rock"
collide(x::Paper, y::Paper) = "papers slide past"
collide(x::Rock, y::Scissors) = "rock crushes scissors"
# ... etc

# The runtime types of BOTH arguments determine which method is called
collide(Rock(), Paper())  # Calls method #2
collide(Paper(), Rock())  # Calls method #3
```

## Basic Patterns

### Type-Specific Methods

```julia
# Generic fallback
process(x) = error("Not implemented for type $(typeof(x))")

# Specific implementations
process(x::Int) = x + 1
process(x::Float64) = x * 2.0
process(x::String) = uppercase(x)
process(x::AbstractArray) = sum(x)

process(5)         # -> 6
process(5.0)       # -> 10.0
process("hello")   # -> "HELLO"
process([1,2,3])   # -> 6
```

### Parametric Methods

```julia
# Works for any numeric type
double(x::T) where T<:Number = 2x

# Can use the type parameter
convert_to_float(x::T) where T<:Integer = Float64(x)

# Multiple type parameters
combine(x::T, y::S) where {T<:Number, S<:Number} = promote_type(T, S)(x + y)
```

### Abstract Type Hierarchies

```julia
abstract type Animal end
abstract type Pet <: Animal end

struct Dog <: Pet
    name::String
end

struct Cat <: Pet
    name::String
end

struct Wolf <: Animal
    pack_size::Int
end

# General method for all animals
speak(a::Animal) = "Some animal sound"

# More specific for pets
speak(p::Pet) = "$(p.name) says something"

# Most specific for individual types
speak(d::Dog) = "$(d.name) says woof!"
speak(c::Cat) = "$(c.name) says meow!"

fido = Dog("Fido")
speak(fido)  # -> "Fido says woof!" (most specific method)
```

**Method specificity**: More specific methods are preferred:
1. Concrete types > abstract types
2. Subtypes > supertypes
3. Longer signature matches > shorter

## Advanced Patterns

### Holy Traits Pattern

Use types to dispatch on properties/behaviors rather than type hierarchies.

```julia
# Define trait types
abstract type ArrayStyle end
struct Dense <: ArrayStyle end
struct Sparse <: ArrayStyle end

# Associate types with traits
array_style(::Type{<:Array}) = Dense()
array_style(::Type{<:SparseArrays.SparseMatrixCSC}) = Sparse()

# Dispatch on trait
function compute_sum(A::AbstractArray)
    # Dispatch to trait-specific method
    compute_sum(array_style(typeof(A)), A)
end

# Trait-specific implementations
compute_sum(::Dense, A) = sum(A)  # Use standard sum
compute_sum(::Sparse, A) = sum(nonzeros(A))  # Optimized for sparse

# Works automatically for new types
struct MyDenseArray <: AbstractArray{Float64, 2}
    # ...
end
array_style(::Type{<:MyDenseArray}) = Dense()
```

**When to use:**
- Multiple orthogonal properties (e.g., mutability, memory layout)
- Properties that don't fit into type hierarchy
- Avoid type proliferation

### Argument-Type Computation

Dispatch based on types of arguments:

```julia
# Compute return type based on input types
function op(x::T, y::S) where {T<:Number, S<:Number}
    R = promote_type(T, S)  # Compute common type
    return R(x) + R(y)
end

op(1, 2.0)  # Returns Float64
```

### Val Types for Compile-Time Constants

Dispatch on values known at compile time:

```julia
compute(::Val{:fast}, x) = x^2  # Fast approximation
compute(::Val{:accurate}, x) = exp(log(x) * 2)  # Accurate computation

# Call with Val
compute(Val(:fast), 5.0)
```

**When to use:**
- Algorithm selection based on symbol/integer constant
- Compile-time optimization
- Avoiding branches in hot loops

### Diagonal Dispatch

Require all arguments to have the same type:

```julia
# All arguments must be the same type T
function homogeneous_add(x::T, y::T, z::T) where T
    return x + y + z
end

homogeneous_add(1, 2, 3)      # OK: all Int
homogeneous_add(1, 2.0, 3)    # ERROR: types don't match
```

### Tuple-Based Dispatch

```julia
# Different handling for different tuple structures
process(t::Tuple{Int, Float64}) = "int and float"
process(t::Tuple{String, String}) = "two strings"
process(t::Tuple{Vararg{Int}}) = "all ints"

process((1, 2.0))      # -> "int and float"
process(("a", "b"))    # -> "two strings"
process((1, 2, 3, 4))  # -> "all ints"
```

## Performance Considerations

### Method Ambiguity

```julia
# Ambiguous methods
foo(x::Int, y::Any) = 1
foo(x::Any, y::String) = 2

# What should this call?
foo(1, "hello")  # ERROR: ambiguous
```

**Solution:** Add a more specific method
```julia
foo(x::Int, y::String) = 3  # Resolves ambiguity
```

### Type Instability from Dispatch

```julia
# Type-unstable: return type depends on runtime value
function bad_dispatch(x)
    if x > 0
        return process_positive(x)  # Returns Int
    else
        return process_negative(x)  # Returns Float64
    end
end
```

**Solution 1:** Make return types consistent
```julia
function good_dispatch(x)
    if x > 0
        return Float64(process_positive(x))
    else
        return process_negative(x)
    end
end
```

**Solution 2:** Use dispatch instead of branches
```julia
process(x::Positive) = # returns Int
process(x::Negative) = # returns Int

# Single return type
function good_dispatch(x)
    return process(classify(x))
end
```

### Union Splitting

Julia optimizes small unions (2-3 types):

```julia
# Efficiently dispatched
function handle(x::Union{Int, Float64})
    # Julia generates code for both branches
end
```

**Avoid large unions** (>3 types) in performance-critical code.

### Avoiding Dynamic Dispatch

**Dynamic dispatch** (runtime lookup) is slow:

```julia
function sum_array(arr::Vector{Any})
    s = 0
    for x in arr
        s += process(x)  # Dynamic dispatch on each iteration
    end
    return s
end
```

**Solution:** Use typed containers
```julia
function sum_array(arr::Vector{Int})
    s = 0
    for x in arr
        s += process(x)  # Static dispatch
    end
    return s
end
```

Or parametric types:
```julia
function sum_array(arr::Vector{T}) where T
    s = zero(T)
    for x in arr
        s += process(x)  # Static dispatch for type T
    end
    return s
end
```

## Design Patterns

### Builder Pattern with Dispatch

```julia
# Build different objects based on input types
build(x::Int) = IntProcessor(x)
build(x::String) = StringProcessor(x)
build(x::Array) = ArrayProcessor(x)

# Generic interface
process(builder::AbstractProcessor, data) = # ...

# Usage
processor = build(my_input)  # Dispatch selects right processor
result = process(processor, data)
```

### Strategy Pattern

```julia
# Define strategy types
abstract type SortStrategy end
struct QuickSort <: SortStrategy end
struct MergeSort <: SortStrategy end
struct InsertionSort <: SortStrategy end

# Dispatch on strategy
sort_data(::QuickSort, data) = quicksort(data)
sort_data(::MergeSort, data) = mergesort(data)
sort_data(::InsertionSort, data) = insertionsort(data)

# Select strategy
function sort_data(data; strategy=QuickSort())
    sort_data(strategy, data)
end
```

### Visitor Pattern

```julia
# Define visitable types
abstract type TreeNode end
struct Leaf <: TreeNode
    value::Int
end
struct Branch <: TreeNode
    left::TreeNode
    right::TreeNode
end

# Multiple visitors
visit(::Type{SumVisitor}, node::Leaf) = node.value
visit(::Type{SumVisitor}, node::Branch) =
    visit(SumVisitor, node.left) + visit(SumVisitor, node.right)

visit(::Type{CountVisitor}, node::Leaf) = 1
visit(::Type{CountVisitor}, node::Branch) =
    visit(CountVisitor, node.left) + visit(CountVisitor, node.right)
```

## Best Practices

### 1. Start Generic, Specialize When Needed

```julia
# Generic implementation
compute(x) = x^2

# Specialize only if needed
compute(x::BigInt) = # optimized for large integers
```

### 2. Document Method Relationships

```julia
"""
    process(x::AbstractArray)

Generic array processing. Specific array types may provide optimized methods.

# Implementations
- `process(x::Vector)`: Optimized for 1D arrays
- `process(x::Matrix)`: Optimized for 2D arrays
"""
process(x::AbstractArray) = # ...
```

### 3. Avoid Over-Specialization

```julia
# TOO SPECIFIC
foo(x::Vector{Float64}) = # ...
foo(x::Vector{Float32}) = # ...
foo(x::Vector{Int64}) = # ...

# BETTER: Parametric
foo(x::Vector{T}) where T<:Number = # ...
```

### 4. Use Abstract Types for Interfaces

```julia
# Define interface via abstract type
abstract type Optimizer end

# Concrete implementations
struct GradientDescent <: Optimizer
    learning_rate::Float64
end

struct Adam <: Optimizer
    α::Float64
    β1::Float64
    β2::Float64
end

# Common interface
optimize(opt::Optimizer, f, x0) = # dispatch to specific method
```

### 5. Separate Generic and Specialized Logic

```julia
# Public API: generic
function solve(problem::Problem)
    # Common preprocessing
    preprocess!(problem)

    # Dispatch to specialized solver
    solve_specialized(problem)

    # Common postprocessing
    postprocess!(problem)
end

# Specialized implementations
solve_specialized(p::LinearProblem) = # ...
solve_specialized(p::NonlinearProblem) = # ...
```

## Common Pitfalls

### 1. Forgetting Type Parameters

```julia
# BAD: Non-concrete type
struct Container
    data::AbstractVector  # Abstract!
end

# GOOD: Parametric type
struct Container{T<:AbstractVector}
    data::T  # Concrete at instantiation
end
```

### 2. Too Many Type Parameters

```julia
# OVERKILL
struct OverEngineered{T, S, R, U, V, W}
    # ...
end

# BETTER: Only parameterize what's necessary
struct Reasonable{T}
    # ...
end
```

### 3. Dispatch on Mutable State

```julia
# BAD: Don't dispatch on mutable fields
function process(x::MyType)
    if x.is_ready  # BAD: branch on state
        # ...
    end
end

# GOOD: Use types for state
abstract type MyType end
struct Ready <: MyType end
struct NotReady <: MyType end

process(x::Ready) = # ...
process(x::NotReady) = # ...
```

## Debugging Dispatch

### See Which Method is Called

```julia
@which foo(1, 2.0)  # Shows which method will be called
```

### List All Methods

```julia
methods(foo)  # Shows all methods for foo
```

### Method Ambiguities

```julia
# Check for ambiguities
methods(foo).ms  # List of methods
# Look for WARNING: Method definition ambiguous
```

### Generated Code

```julia
@code_lowered foo(1, 2.0)  # Lowered IR
@code_typed foo(1, 2.0)    # Type-inferred IR
@code_llvm foo(1, 2.0)     # LLVM IR
@code_native foo(1, 2.0)   # Native assembly
```

## Summary

**Key principles:**
1. Dispatch on types, not values (except Val)
2. Keep hierarchies shallow and simple
3. Use traits for orthogonal properties
4. Specialize only when necessary
5. Keep methods small and focused

**Performance:**
- Concrete types in structs
- Avoid large unions
- Use parametric types
- Watch for method ambiguities

**Design:**
- Generic implementations first
- Specialize for performance or correctness
- Use abstract types for interfaces
- Document method relationships
