import numpy as np

# matmul + relu numpy implement
dtype = "float32"
a_np = np.random.rand(128, 128).astype(dtype)
b_np = np.random.rand(128, 128).astype(dtype)
c_mm_relu = np.maximum(a_np @ b_np, 0)


# low-level numpy
# use for-loop, and allocate the array explicitly
def lnumpy_mm_relu(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    # each time compute an element
    Y = np.empty((128, 128), dtype=dtype)
    for i in range(128):
        for j in range(128):
            for k in range(128):
                if k == 0:
                    Y[i, j] = 0
                Y[i, j] += A[i, k] * B[k, j]
    for i in range(128):
        for j in range(128):
            C[i, j] = max(Y[i, j], 0) # relu
c_np = np.empty((128, 128), dtype=dtype)
lnumpy_mm_relu(a_np, b_np, c_np)
np.testing.assert_allclose(c_mm_relu, c_np, rtol=1e-5) # assert the difference

# mm_relu tensorIR implement
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T


@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def mm_relu(A: T.Buffer[(128, 128), "float32"],
                B: T.Buffer[(128, 128), "float32"],
                C: T.Buffer[(128, 128), "float32"]):
        # noalias means A, B, C pointer will not  repeat
        T.func_attr({"global_symbol": "mm_relu", "tir.noalias": True})
        Y = T.alloc_buffer((128, 128), dtype="float32")
        # block and block axis
        for i, j, k in T.grid(128, 128, 128):
            with T.block("Y"): # each block is self-contained
                vi = T.axis.spatial(128, i) # bind exactly 128 with the outer gird
                vj = T.axis.spatial(128, j)
                vk = T.axis.reduce(128, k)
                # equals
                # vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j)
                C[vi, vj] = T.max(Y[vi, vj], T.float32(0))

print(type(MyModule))
print(type(MyModule["mm_relu"]))

# given a tensorIR, how to get other Transformation?
# actullay, low-level mm_relu can be transfomed like this one.
def lnumpy_mm_relu_v2(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    # each time compute an element
    Y = np.empty((128, 128), dtype=dtype)
    for i in range(128):
        for j0 in range(32):
            for k in range(12):
                for j1 in range(4):
                    j = j0 * 4 + j1
                    if k == 0:
                        Y[i, j] = 0
                    Y[i, j] += A[i, k] * B[k, j]
    for i in range(128):
        for j in range(128):
            C[i, j] = max(Y[i, j], 0) # relu
c_np = np.empty((128, 128), dtype=dtype)
lnumpy_mm_relu_v2(a_np, b_np, c_np)
np.testing.assert_allclose(c_mm_relu, c_np, rtol=1e-5) # assert the difference

# how to use tensorIR to automatically find this primitive function?

# first display the current Module code
print(MyModule.script())

# use schedule to manually transform the module
print("*" * 50)
sch = tvm.tir.Schedule(MyModule)
block_Y = sch.get_block("Y", func_name="mm_relu")
i, j, k = sch.get_loops(block_Y)
j0, j1 = sch.split(j, factors=[None, 4]) # split the axis
sch.reorder(j0, k, j1) # reorder the axis
print(sch.mod.script()) # current module code

# get another transformation
# actually fused C into the loop, fusion
print("*" * 50)
block_C = sch.get_block("C", "mm_relu")
sch.reverse_compute_at(block_C, j0)
print(sch.mod.script())

print("*" * 50)
# decompose the init axis in the block
sch.decompose_reduction(block_Y, k)
print(sch.mod.script())

# build and run tensorIR Module
rt_lib = tvm.build(MyModule, target="llvm") # runtime library
print(type(rt_lib))

a_nd = tvm.nd.array(a_np)
b_nd = tvm.nd.array(b_np)
c_nd = tvm.nd.empty((128, 128), dtype="float32")
print(type(c_nd))

# before transformation
func_mm_relu = rt_lib["mm_relu"]
func_mm_relu(a_nd, b_nd, c_nd)
np.testing.assert_allclose(c_mm_relu, c_nd.numpy(), rtol=1e-5)

# after transformation
rt_lib_after = tvm.build(sch.mod, target="llvm")
func_mm_relu = rt_lib_after["mm_relu"]
func_mm_relu(a_nd, b_nd, c_nd)
np.testing.assert_allclose(c_mm_relu, c_nd.numpy(), rtol=1e-5)

# compare the running time
f_timer_before = rt_lib.time_evaluator("mm_relu", tvm.cpu())
print("Time cost of MyModule %g sec" % f_timer_before(a_nd, b_nd, c_nd).mean)
f_timer_after = rt_lib_after.time_evaluator("mm_relu", tvm.cpu())
print("Time cost of transformed sch.mod %g sec" % f_timer_after(a_nd, b_nd, c_nd).mean)
