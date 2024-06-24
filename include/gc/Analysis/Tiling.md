
Let's take a look for single matmul tiling: 

```
[128 x 512] ifm         [512 x 128] weight
    \                       /
              matmul
                |
          [128 x 128] ofm
```
**ifm** stands for input feature map, **ofm** - output feature map. 

We have 2 main (most interesting) parameters for us:
  1. memory (or maybe memory hierarchy)
  2. parallelisation

## Memory 

Assume we have small amount amount of memory: 
```
  buffer = 4096 
```

Total amount of memory resuired to process this op we need: 
```
  128 * 128 + 128 * 512 + 512 * 128 = 147'456
```
what is much more than size of buffer. 

So we need tiling. In common there are 2 completely different appropaches:
  1. Tiling starts from an input
  2. Tiling strats from output

When you start tiling from input you have to create 



at first just about memory.  We are tiling starting from output. 
Also assume that smallest reasonable tile size is `[32 x 32]`. (It is used to avoid super small tiles)

Origianl layer without tiling requires `128*128 + 128*512*2 = 147'456`.
To execute it we will need: 
```
| t   load_ifm load_w
*     execute
      store_ofm
```
We need some estimation for load/store time depending on size, and execution depending on size. 
Dependency of execution time from total processing data size should be almost linear. (as in general we are changing loop boundaries)
Dependency of load/store operation time from data size can be non-trivial - it can be constant or stepped, or linear. 
So it will be something like: `max(t_li, t_lw) + t_x * 144 + t_so` . 

Let's estimate what changing in case of 4 ofm tiles. 

```
original [128 x 128] -> just divide by 2 -> tile [64 x 64]                              
```

```
input                            weight 
               512                        64          64
    +-----------------------+        +-----------|-----------+
 64 |          1            |        |           |           |
    |                       |        |           |           |
    -------------------------    512 |    3      |    4      |
 64 |          2            |        |           |           |
    |                       |        |           |           |
    +-----------------------+        +-----------|-----------+

output
       64         64        
    +-----------|-----------+
 64 |   13      |    14     |
    |           |           |
    -------------------------
 64 |   23      |    24     |
    |           |           |
    +-----------|-----------+
```
so ifm and weight sizes are `[64 x 512]` and `[512 x 64]` without additional tiling: additional required size `65'536`. It fits to memory.
So it's possible to handle several tiles in parallel 

Consider execution workflow without memory restriction:
```
0 |      load_ifm0 load_w0         
1 *      execution_t0       load_ifm1 load_w0
2        store_ofm0         execution_t1       load_ifm0 load_w1
3                           store_ofm1         execution_t2       load_ifm1 load_w1
4                                              store_ofm2         execution_t3
5                                                                 store_ofm3
```
in each row all ops can be triggered simultaneously, but expected that stage ends when slowest ends
We need to decide which of these options ares better. 

```
  assume store > load > exec 

  2 mem op < bandwidth of device -> Q to cost model: Can we use 2 ops in parallel? 

  In this case we will have: 
    max(t_li0, t_lw0) + max(t_x0, t_li1, t_lw1) + max(t_s0, t_x1, t_li2, t_lw2) + max(t_s1, t_x2, t_li3, t_lw3) + t_x * 144 + t_so

  [ buffer ... output ] {execution unit}
```

In this case we need some cost model or some estimation to decide which configuration is better.
  1. Is it possible to execute this ops simulatneously? 
  2. What a some perf estimation for load/store/execution from the size of data that will be processed?
  3. Is so simple HW model is enough or we should do some more complex? Can we have complex model as sequense of several simple ones?


In the case of tiling along reduction axis let's consider ifm and weight sizes are `[64 x 256]` and `[256 x 64]` and we will need extra space for summation
If we tile along reduction axis:

```
input                            weight 
       256        256                     64          64
    +-----------|-----------+        +-----------|-----------+
 64 |    1      |    2      |    256 |    5      |    6      |
    |           |           |        |           |           |
    -------------------------        -------------------------
 64 |    3      |    4      |    256 |    7      |    8      |
    |           |           |        |           |           |
    +-----------|-----------+        +-----------|-----------+

output
       64         64        
    +-----------|-----------+
 64 |    01     |    23     |
    |           |           |
    -------------------------
 64 |    45     |    67     |
    |           |           |
    +-----------|-----------+
```
```

                     1 * 5                                                2 * 7                                             1 * 6                                                 2 * 8
[64 (high) x 256(left)] [256(high) x 64 (left)]      [64(high) x 256(right)] [256(low) x 64(left)]     [64 (high) x 256(left)] [256(high) x 64 (right)]      [64(high) x 256(right)] [256(low) x 64(right)]
            [64 (high) x 64 (left)] part                         [64 (high) x 64 (left)] part                      [64 (high) x 64 (right)] part                         [64 (high) x 64 (right)] part
                                      [64 (high) x 64 (left)]                                                                                  [64 (high) x 64 (right)] 

                    3 * 5                                                4 * 7                                             3 * 6                                                 4 * 8
[64 (low) x 256(left)] [256(high) x 64 (left)]       [64(low) x 256(right)] [256(low) x 64(left)]      [64 (low) x 256(left)] [256(high) x 64 (right)]       [64(low) x 256(right)] [256(low) x 64(right)]
              [64 (low) x 64 (left)] part                        [64 (low) x 64 (left)] part                       [64 (low) x 64 (right)] part                          [64 (low) x 64 (right)] part
                                      [64 (low) x 64 (left)]                                                                                    [64 (low) x 64 (right)] 
```
```
 0  |     load_ifm1 load_w5         
 1  *     execution_t15       load_ifm2 load_w7 
 2                            execution_t27      load_w6
 3        sum(ofm0 ofm1)                         execution_t16      load_w8
 4        store_ofm01                                               execution_t28       load_ifm3
 5                                                sum(ofm2 ofm3)                        execution_t35      load_ifm4
 6                                                store_ofm23                                              execution_t47    
 7                                                                                      sum(ofm4 ofm5)                      execution_t36
 8                                                                                      store_ofm45                                          execution_t48
 9                                                                                                                          sum(ofm6 ofm7)
10                                                                                                                          store_ofm610
```


These 3 configurations should be chosen on the basis of HW model. Number of possibiliteies will be strictly limited with amount of memory. For example if amount of memory is very low we should tile just to fit to limitation.  

For example, ifms sizes are `256*64 = 16'384`, weight sizes similar `64*256 = 16'384` and ofm is `64*64 = 4'096`


```
 0        load_ifm1 load_w5         
   32'768 [  16'384(ifm1) |  16'384(w5)  |      free        ]
 1        execution_t15       load_ifm2 load_w7 
   69'632 [  16'384(ifm1) |  16'384(w5)  |  4'096(ofm0)   | 16'384(ifm2) |  16'384(w7)  |     free        ]
 2                            execution_t27      load_w6
   73'728 [  16'384(ifm1) |  16'384(w6)  |  4'096(ofm0)   | 16'384(ifm2) |  16'384(w7)  |  4'096(ofm1)  |     free        ]
 3        sum(ofm0 ofm1)                         execution_t16      load_w8
   73'728 [  16'384(ifm1) |  16'384(w6)  |  4'096(ofm01)  | 16'384(ifm2) |  16'384(w8)  |  4'096(ofm2)  |     free        ]
 4        store_ofm01                                               execution_t28       load_ifm3 load_w5
   77'824 [  16'384(ifm3) |  16'384(w5)  |  4'096(ofm01)  | 16'384(ifm2) |  16'384(w8)  |  4'096(ofm2)  |  4'096(ofm3)  |     free        ]
 5                                                sum(ofm2 ofm3)                        execution_t35      load_ifm4 load_w7
   77'824 [  16'384(ifm3) |  16'384(w5)  |  4'096(ofm23)  | 16'384(ifm4) |  16'384(w7)  |  4'096(ofm4)  |  4'096(ofm3)  |     free        ]
 6                                                store_ofm23                                              execution_t47     load_w6
   77'824 [  16'384(ifm3) |  16'384(w6)  |  4'096(ofm23)  | 16'384(ifm4) |  16'384(w7)  |  4'096(ofm4)  |  4'096(ofm5)  |     free        ]
 7                                                                                      sum(ofm4 ofm5)                       execution_t36   load_w8
   77'824 [  16'384(ifm3) |  16'384(w6)  |  4'096(ofm45)  | 16'384(ifm4) |  16'384(w8)  |  4'096(ofm6)  |  4'096(ofm5)  |     free        ]
 8                                                                                      store_ofm45                                          execution_t48
   77'824 [  16'384(ifm3) |  16'384(w6)  |  4'096(ofm45)  | 16'384(ifm4) |  16'384(w8)  |  4'096(ofm6)  |  4'096(ofm7)  |     free        ]
 9                                                                                                                           sum(ofm6 ofm7)
   77'824 [  16'384(ifm3) |  16'384(w6)  |  4'096(ofm67)  | 16'384(ifm4) |  16'384(w8)  |  4'096(ofm6)  |  4'096(ofm7)  |     free        ]
10                                                                                                                           store_ofm610
```
It is trivial example that reduces memory usage. Without tiling memory required `128*512*2 + 128*128 = 147'456`. Memory usage can be reduced with extra load operations. In extreme case this tiling can be used with memory for single tile and space for partial sum: `64*256*2 + 64*64 + 64*64= 40'960`. And such extreme case without tiling along reduction axis: `64*512*2 + 64*64 = 69'632`. 

Of cause such estimation ignores possible fragmatation and depends on ops possibility to execute inplace. 

### Memory with hierarchy

When we have memory hierarchy it's possible to apply tiling stage by stage from biggest buffer to smallest or to tile in one stage to the smallest buffer. (How to decide - depends on memory and what it does). In the case of multistage approach we will have several estimations that will be depend on each other. This can be very difficult to estimate effectively. 

Let's take a look for both appraoches with 3 levels of memory hierarchy: 

```
    [ 10'000 ]      1st level
        | 
   [  50'000  ]     2nd level
        |
 [    100'000    ]  3rd level

```

Consider these 2 ways of tiling for these memory hierarchy for already used matmul layer. 

#### Multistage approach

Tile for `3rd level` so size of tiled op should fit into `100'000`. For this will be enough Output tiling:

```
original [128 x 128] -> just divide by 2 -> tile [64 x 64]                              
```

```
input                            weight 
               512                        64          64
    +-----------------------+        +-----------|-----------+
 64 |          1            |        |           |           |
    |                       |        |           |           |
    -------------------------    512 |    3      |    4      |
 64 |          2            |        |           |           |
    |                       |        |           |           |
    +-----------------------+        +-----------|-----------+

output
       64         64        
    +-----------|-----------+
 64 |   13      |    14     |
    |           |           |
    -------------------------
 64 |   23      |    24     |
    |           |           |
    +-----------|-----------+
```

So required size is: `64*512*2 + 64*64 = 69'632`. 

```
0 |      3: load_ifm1 load_w3         
1 *      execution_t0          
2        store_ofm13           3: load_ifm2 
3                              execution_t1 
3                              store_ofm23   3: load_ifm1 load_w4
4                                            execution_t2          
5                                            store_ofm14           3: load_ifm2
6                                                                  execution_t3
7                                                                  store_ofm24
```

Now we should tile single tile for `2nd level`: 

```
[64 x 512] ifm         [512 x 64] weight
    \                       /
              matmul
                |
          [64 x 64] ofm
``` 

```
input                            weight 
               512                     16          16
    +-----------------------+        +-----|-----|-----|-----+
 16 |          1_2          |        |     |     |     |     |
    -------------------------        | 5_2 |     |     |     |
    |          2_2          |        |     | 6_2 |     |     |
    -------------------------    512 |     |     | 7_2 |     |
 16 |          3_2          |        |     |     |     | 8_2 |
    -------------------------        |     |     |     |     |
    |          4_2          |        |     |     |     |     |
    +-----------------------+        +-----|-----|-----|-----+

output
       16          16        
    +-----|------|------|-----+
 16 | 15_2|      |      | 18_2|
    ------|-------------|------
    |     |      |      |     |
    ------|-------------|------
 16 |     | 36_2 |      |     |
    ------|-------------|------
    |     |      | 47_2 |     |
    +-----|------|------|-----+
```

With `32x32` tile size required: `32*512*2 + 32*32 = 33'792`, but it's not possible to parrellise load/stroes of strat tile with execution of the next. It is possible if the size of tile is less than 24. 

Logic is the the following - execution stage "blocks" from modification input, weight and output, so we keep weight split and trying to split input into 2 buffers.

To have something like: 
```
 +------+------+-------------+------+
 | ifm0 | ifm1 |     w0      | ofm0 |
 +------+------+-------------+------+
```

To reuse w0, and switch input to the next tile.
So we have foolwing equation: 
```
  32*512 (w) + x*512*2 (ifm0 + ifm1) + 32*x (ofm) = 50'000 (buffer size)
```
from this equation `x = 30` should be fine, but as we have 3 tiles of input, lets make it's size more granular: `round_up(64/3) = 22`, so 3 tile: `22 22 20`
In this case weight shared between tiles, and not used tiling along reduction axes.

Let's compare:
```
input                            weight 
               512                     32    32   
    +-----------------------+        +-----|-----+
 22 |          1_2          |        |     |     |
    -------------------------        | 4_2 |     |
 22 |          2_2          |        |     | 5_2 |
    -------------------------    512 |     |     |
 20 |          3_2          |        |     |     |
    -------------------------        |     |     |
                                     |     |     |
                                     +-----|-----+
output
       32   32   
    +-----|------+
 22 | 14_2|      |
    ------|-------
 22 |     |      |
    ------|-------
 20 |     | 35_2 |
    +-----|------+
```


`32x32` tile size
```
0 |      3: load_ifm1 load_w3         
1 *      execution_t0          
2        store_ofm13           3: load_ifm2
3                              execution_t1
3                              store_ofm23           3: load_ifm1 load_w4
4                                                    execution_t2          
5                                                    store_ofm14           3: load_ifm2
6                                                                          execution_t3
7                                                                          store_ofm24
```


`22x32` tile size:
```
0 |      2: load_ifm1 load_w4         
1 *      execution_t0          2: load_ifm2 
2        store_ofm14           execution_t1   2: load_ifm3
3                              store_ofm24    execution_t2          
5                                             store_ofm34    2: load_ifm1 load_w5
6                                                            execution_t3          2: load_ifm2
7                                                            store_ofm15           execution_t4   2: load_ifm3
8                                                                                  store_ofm25    execution_t5                                               
9                                                                                                 store_ofm25
```

We can try to save full-sized weights with extra tiling in input, to have this equation:
```
  64*512 (w) + x*512*2 (ifm0 + ifm1) + 64*x (ofm) = 50'000 (buffer size)
```
from this equation: `x = floor(15.8) = 15`, so it's `rounup(64/15) = 5`, to have more similar file size - `roundup(64 / 5) = 13`, so tiles `13 13 13 13 12`. 

and tile size is: `13x64`
```
input                            weight 
               512                        64
    +-----------------------+        +----------+
 13 |          1_2          |        |          |
    -------------------------        |          |
 13 |          2_2          |        |          |
    -------------------------    512 |   6_2    |
 13 |          3_2          |        |          |
    -------------------------        |          |
 13 |          4_2          |        |          |
    -------------------------        +----------+
 12 |          5_2          |
    +-----------------------+

output
         64   
    +-----------+
 13 |           |
    -------------
 13 |           |
    -------------
 13 |           |
    -------------
 13 |           |
    -------------
 12 |           |
    +-----------+
```
```
0 |      2: load_ifm1 load_w6         
1 *      execution_t0          2: load_ifm2 
2        store_ofm16           execution_t1   2: load_ifm3
3                              store_ofm26    execution_t2   2: load_ifm4       
5                                             store_ofm36    execution_t3  2: load_ifm5
6                                                            store_ofm46   execution_t4 
7                                                                          store_ofm56 
```
This example shows, that even if time of load is similar for different tile sizes it's effective to tile even for single core. 

```
0 |      3: load_ifm1 load_w3         
1 |         2: load_ifm1_2 load_w6_2         
2 *         execution_t02             2: load_ifm2_2 
3           store_ofm16_2             execution_t12    2: load_ifm3_2
4                                     store_ofm26_2    execution_t22    2: load_ifm4_2       
5                                                      store_ofm36_2    execution_t32   2: load_ifm5_2
6                                                                       store_ofm46_2   execution_t42 
7                                                                                       store_ofm56_2 
8        store_ofm13           3: load_ifm2 
9-15                           execution_t1 
16                             store_ofm23   3: load_ifm1 load_w4
17-23                                        execution_t2          
24                                           store_ofm14           3: load_ifm2
25-31                                                              execution_t3
32                                                                 store_ofm24
```

Let's try to add tiling for `1st layer`:

```
input                            weight 
                512                      64
    +-----------------------+        +----------+
 13 |                       |        |          |
    +-----------------------+        |          |
                                     |          |
                                 512 |          |
                                     |          |
                                     |          |
                                     |          |
                                     +----------+
output
         64   
    +-----------+
 13 |           |
    +-----------+
```
If we are trying to use on this stage share input and tile only weight to have case like:
```
 +------+-------------+-------------+------+
 | ifm0 |     w0      |     w1      | ofm0 |
 +------+-------------+-------------+------+
```
corresponding equation is: 
```
  13*512 (ifm0) + x*512*2 (w0 + w1) + 13*x (ofm) = 10'000 (buffer size)
```
and resulting `x = 3`. This size looks too small. Produced amount of ticks is `n_ticks = num_tiles + 2`, `n_ticks = 22 + 2 = 24`.

Let's try to tile along reduction axes. 
```
 +------+------+-------------+-------------|--------|--------|------+
 | ifm0 | ifm1 |     w0      |     w1      | ofm0_p | ofm1_p | ofm0 |
 +------+------+-------------+-------------|--------|--------|------+
```
This is `layout 1`.


```
0 |      2: load_ifm0 load_w0         
1 *      execution_t0 (ofm0_p)  2: load_ifm1 load_w1 
2                               execution_t1 (ofm1_p)  2: load_ifm2 load_w2 
3                               sum(ofm1_p, ofm0_p)               
4                               store_ofm0             execution_t2 (ofm0_p)  2: load_ifm3 load_w3
5                                                      sum(ofm0, ofm0_p)                              
6                                                      store_ofm1_p           execution_t3 (ofm0)     2: load_ifm3 load_w3
7                                                                             sum(ofm1_p, ofm0)     
8                                                                             store_ofm0_p            execution_t4 (ofm1_p)
9                                                                                                     sum(ofm0_p, ofm1_p)
10                                                                                                    store_ofm0
                                                                              ^
                                                                              |
                                                         It depends on possibility to use sum and execute simultaniously. 
```
So number of used ticks is `n_ticks = (num_tiles - 1) * 2 + 2 + 1` . 
In this formula: `(num_tiles - 1) * 2` - is `execution + sum`, last `+ 1` is first `execution`, `+ 2` is initial load and final store.

It works when in buffer we have layout like - `layout 1`. In current case to achieve layout like this:
```
  x*13*2 (ifm0 + ifm1) + x*64*2 (w0 + w1) + 13*64*3 (ofm0_p + ofm1_p + ofm0) = 10'000 (buffer size)
```
so `x = 48`, it requires `11` tiles. Which will produce `n_ticks = 23`.

Without approach to pack more ops into single time unit: 
```
 +------+-------------+--------+--------+------+
 | ifm0 |     w0      | ofm0_p | ofm1_p | ofm0 |
 +------+-------------+--------+--------+------+
```
This is `layout 2`.

```
  x*13 (ifm0) + x*64*2 (w0) + 13*64*3 (ofm0_p + ofm1_p + ofm0) = 10'000 (buffer size)
```
from this equation: `x = 97`, which requires: `6` tiles along reduction. 
```
0 |      2: load_ifm0 load_w0         
1 *      execution_t0 (ofm0_p)  
2                               2: load_ifm0 load_w0   
3                               execution_t1 (ofm1_p)               
4                               sum(ofm1_p, ofm0_p)    2: load_ifm0 load_w0 
5                               store_ofm0             execution_t2 (ofm0_p)  
6                                                      sum(ofm0, ofm0_p)      2: load_ifm0 load_w0                        
7                                                      store_ofm1_p           execution_t3 (ofm0)     
8                                                                             sum(ofm1_p, ofm0)       2: load_ifm0 load_w0
9                                                                             store_ofm0_p            execution_t4 (ofm1_p)
10                                                                                                    sum(ofm0_p, ofm1_p)
11                                                                                                    store_ofm0
```
In this case `n_ticks = (num_tiles - 1) * 2 + 2 + 2`, so `n_ticks = 14`


Let's try to check tiling along several axes  (reduction and output w).

```
input                            weight 
       171     171       170            32    32
    +-------+---------+------+        +-----|-----+
 13 |   1   |   2     |  3   |    171 |  4  |  5  |
    +-------+---------+------+        |     |     |
                                      |-----|-----|
                                  171 |  6  |  7  |
                                      |     |     |
                                      |-----|-----|
                                  170 |  8  |  9  |
                                      +-----|-----+
output
       32    32   
    +-----|------+
 13 |     |      |
    +-----|------+
```
We can use same layout here: 
```
 +------+-------------+--------+--------+------+
 | ifm0 |     w0      | ofm0_p | ofm1_p | ofm0 |
 +------+-------------+--------+--------+------+
```
```
0 |      2: load_ifm1 load_w4         
1 *      execution_t0 (ofm0_p)  
2                               2: load_ifm2 load_w6   
3                               execution_t1 (ofm1_p)               
4                               sum(ofm1_p, ofm0_p)    2: load_ifm3 load_w8 
5                               store_ofm0             execution_t2 (ofm0_p)  
6                                                      sum(ofm0, ofm0_p)      2: load_ifm1 load_w5                        
7                                                      store_ofm1_p           execution_t3 (ofm0)     
8                                                                             sum(ofm1_p, ofm0)       2: load_ifm2 load_w7
9                                                                             store_ofm0_p            execution_t4 (ofm1_p)
10                                                                                                    sum(ofm0_p, ofm1_p)
11                                                                                                    store_ofm0
...
```
In this case we can have similar amount of ticks as we have similar amount of tiles and same layout `n_ticks = (num_tiles - 1) * 2 + 2 + 2`, so `n_ticks = 14`.
If we change amount of tiles along weight axises (instead of 2 tiles along weight w and 3 tiles along weight h, we have 3 along h and 2 along 2), we will still have similar layout in buffer. 


## Threads/Cores

To utilize threads effectively we will need an amount of them and mark memory as shared/private for specific core. 



