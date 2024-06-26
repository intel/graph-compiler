
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

In further doc I will assume several things about HW:
   1. We have one execution unit and one buffer or buffer hierarachy. 
   2. Execution unit can handle only single execution command. (we can't have 2 simultaneous executions)
      ```
      0  load_ifm0 load_w0
      1  execute (ofm0)       execute (ofm0)      # Not possible - structural hazard           
      ```
   3. Execution stage blocks from any read/modification all used parts of buffer. (we can't write into place, which used during execution)
      e.g.
      in buffer we wahe: 
      ```
         +------+-------------+------+------+
         | ifm0 |     w0      | ofm0 | ofm1 |
         +------+-------------+------+------+
      ```
      ```
      0  load_ifm0 load_w0
      1  execute (ofm0)      # cannot access on this tick to ifm0, w0 and ofm0           
      ```
   4. Schema with ticks assume, that all ops in a row triggered simultneously.
      e.g.
      ```
      0  load_ifm load_w
      1  execute              store_ofm
      ```
      means that we are loading ifm and w into buffer simultaneously and than 
      execution of op with simultaneous storing of some ofm.

To estimate performance of some tiling I will calculate amount of ticks, that will be required with this black-box execution. To make this prediction more applicable to specific HW we will need to understand how long takes load/store/execution compared to each other and depending on amount of data they are processing. 


<!-- 
Total amount of memory resuired to process this op we need: 
```
  128 * 128 + 128 * 512 + 512 * 128 = 147'456
```
what is much more than size of buffer. 

So we need tiling. In common there are 2 completely different appropaches:
  1. Tiling starts from an input
  2. Tiling strats from output

When you start tiling from input you have to create  -->



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

It takes `8` ticks. In this case layout in buffer is: 
```
 +------+-------------+------+
 | ifm0 |     w0      | ofm0 |
 +------+-------------+------+
```

Let's try to reduce amount of ticks.

```
  x*512*2 (ifm0 + ifm1) + 128*512 (w0) + x*128*2 (ofm0 + ofm1) = 100'000 (buffer size)
```
`x = 26`-> requires `5` tiles -> more regular: `x = 26`, tiles are `26 26 26 26 24`.
```
 +------+------+-------------+------+------+
 | ifm0 | ifm1 |     w0      | ofm0 | ofm1 |
 +------+------+-------------+------+------+
```

```
0 |      3: load_ifm0 load_w0         
1 *      execution_t0          3: load_ifm1 
2        store_ofm0            execution_t1   3: load_ifm0
3                              store_ofm1     execution_t2   3: load_ifm1
4                                             store_ofm0     execution_t3  3: load_ifm1
5                                                            store_ofm1    execution_t4
6                                                                          store_ofm0
```
This configuration requires `7` ticks, which is a little better. 

```
input                            weight 
               512                              128
    +-----------------------+        +----------------------+
 26 |          1            |        |                      |
    -------------------------        |                      |
 26 |          2            |    512 |          6           |
    -------------------------        |                      |
 26 |          3            |        |                      |
    -------------------------        +----------------------+
 26 |          4            |
    -------------------------
 24 |          5            |
    +-----------------------+

output
           128        
    +---------------+
 26 |       16      |
    -----------------
 26 |       26      |
    -----------------
 26 |       36      |
    -----------------
 26 |       46      |
    -----------------
 24 |       56      |
    +---------------+
```
As it's best, let's use this tiling.


Now we should calculate tile sizes for single tile for `2nd level` buffer: 

```
[26 x 512] ifm         [512 x 128] weight
    \                       /
              matmul
                |
          [26 x 128] ofm
``` 

```
input                            weight 
               512                            128
    +-----------------------+        +--------------------+
 26 |                       |        |                    |
    +-----------------------+        |                    |
                                     |                    |
                                 512 |                    |
                                     |                    |
                                     |                    |
                                     |                    |
                                     +--------------------+

output
                128        
    +----------------------+
 26 |                      |
    +----------------------+
```

As weigth is beiggest tensor used for operaion processing, let's try to split it. 

Straightforward layout in buffer: 
```
 +------+-------------+------+
 | ifm0 |     w0      | ofm0 |
 +------+-------------+------+
```
Such layout produce smallest amount of tiles.
```
  26*512 (ifm0) + x*512 (w0) + 26*x (ofm) = 50'000 (buffer size)
```
`x = 68` -> 2 tiles required -> `x = 64`. 

Corresponding execution:
```
0 |      2: load_ifm0 load_w0         
1 *      execution_t0          
2        store_ofm0           2: load_w1
3                             execution_t1
4                             store_ofm1  
```
So `5` ticks. 

Let's estimate how much will take execution with more tiles, but also more units utilization. 
```
 +------+-------------+-------------+------+------+
 | ifm0 |     w0      |     w1      | ofm0 | ofm1 |
 +------+-------------+-------------+------+------+
```
```
  26*512 (ifm0) + (x*512 + 26*x)*2 (2 ofm + 2 w) = 50'000 (buffer size)
```
`x = 34` -> 4 tiles required -> `x = 32`. 

Corresponding execution:
```
0   2: load_ifm0 load_w0         
1   execution_t0          2: load_w1 
2   store_ofm0            execution_t1  2: load_w2
3                         store_ofm1    execution_t2   2: load_w3
4                                       store_ofm0     execution_t2
5                                                      store_ofm1                             
```
`6` ticks which is worth than naive approach. 

Let's also check tiling along reduction axis. In this case extra sum op will be required so buffer layout:
```
 +------+-------------+--------+--------+------+
 | ifm0 |     w0      | ofm0_p | ofm1_p | ofm0 |
 +------+-------------+--------+--------+------+
```
```
  26*x (ifm0) + x*128 (w) + 26*128*3 (3 ofm) = 50'000 (buffer size)
```
`x = 259` -> 2 tiles required -> `x = 256`. 

Corresponding execution:
```
0   2: load_ifm0 load_w0         
1   execution_t0 (ofm0_p)
2                          2: load_w1
3                          execution_t0 (ofm1_p)
4                          sum(ofm0_p, ofm1_p)
5                          store_ofm0
```
As you see tiling along reduction axis is not very effective as produces extra stages, which can slow down execution.

Let's combine solutions for 1st level and 2nd:

```
0        3: load_ifm0 load_w0         
1           2: load_ifm0 load_w0         
2           execution_t0          
3           store_ofm0           2: load_w1
4                                execution_t1
5                                store_ofm1        3: load_ifm1 
6-10     store_ofm0                                execution_t1   3: load_ifm0
11-15                                              store_ofm1     execution_t2   3: load_ifm1
16-20                                                             store_ofm0     execution_t3  3: load_ifm1
21-25                                                                            store_ofm1    execution_t4
26                                                                                             store_ofm0
```

Finally let's try to add tiling for `1st layer`:

```
input                            weight 
                512                      64
    +-----------------------+        +----------+
 26 |                       |        |          |
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
 26 |           |
    +-----------+
```

At first let's try to avoid tiling along reduction axis. 
```
  x*512 + y*512 + x*y = 10'000
```
As `x*y` is relatively small, we can understand, that it will work if `x + y < 10'000/512 = 19`, so in general as we are interested in integer solutions,
let's try to have `5` tles for weight - `13 13 13 13 12`. So for `x + 13 < 19`  -> `x < 6`, so `5` tiles on input `6 5 5 5 5`. 
So output amount of tiles in this case is `5*5 = 25`. 

`5` tiles in w, and `5` tiles in input. 

Let's try to check some close solutions to reduce ofm amount of tiles.
`6` tiles in w - `11 11 11 11 11 9`, `x < 19 - 11 = 8 `, so `4` tiles for input -> `7 7 7 5`, and corresponding ofm tiles amount `24`.

`3` tiles for input -> `9 9 8`, so `3*512 + y*512 + 3*y = 10'000` -> `y = 10` -> `7` tiles in w: `10 10 10 10 10 10 4` ( maybe better is `10 9 9 9 9 9 9`), so ofm tiles amout is `7*3 = 21`.

`2` tiles for input -> `13 13`, so `13*512 + y*512 + 13*y = 10'000` -> `y = 6` -> `10` tiles in w: `7 7 7 7 7 7 7 7 4 4`, so ofm tiles amout is `10*2 = 20`.
```
0   1: load_ifm0 load_w0
1   execution_t0 
2   store_ofm0            1: load_w1
3                         execution_t1
4                         store_ofm0    1: load_w2
5                                       execution_t2
...
```
`n_ticks = num_tiles * 2 + 1 ` -> for `20` ofm tiles we need `41` ticks.

```
 +------+-------------+--------+--------+------+
 | ifm0 |     w0      | ofm0_p | ofm1_p | ofm0 |
 +------+-------------+--------+--------+------+
```

Let's check what will happen with tiling along reduction axis:
```
  26*x (ifm0) + x*64 (w) + 26*64*3 (3 ofm) = 10'000 (buffer size)
```
`x = 55` -> 10 tiles required -> `x = 52`. 

Corresponding execution:
```
0   2: load_ifm0 load_w0         
1   execution_t0 (ofm0_p)
2                          2: load_ifm1 load_w1
3                          execution_t1 (ofm1_p)
4                          ofm0 = sum(ofm0_p, ofm1_p)   2: load_ifm2 load_w2
5                                                       execution_t2 (ofm0_p)
6                                                       ofm1_p = sum(ofm0_p, ofm0)    2: load_ifm3 load_w3
7                                                                                     execution_t3 (ofm0_p)
8                                                                                     ofm0 = sum(ofm0_p, ofm1_p)  2: load_ifm4 load_w4
9                                                                                                                 execution_t4 (ofm0_p)   
10                                                                                                                ofm1_p = sum(ofm0_p, ofm0)
11                                                                                                                store_ofm1_p   
...
```

`n_ticks = (num_tiles - 1)*2 + 1 + 3 ` -> to process required `22` ticks. This result is much better than previous one.

```
 +------+-------------+------+-------------+--------+--------+------+--------+
 | ifm0 |     w0      | ifm1 |     w1      | ofm0_p | ofm1_p | ofm0 | ofm2_p |
 +------+-------------+------+-------------+--------+--------+------+--------+
```
Let's check what will happen with tiling along reduction axis:
```
  x*26*2 (ifm0) + x*64*2 (w) + 26*64*4 (3 ofm) = 10'000 (buffer size)
```
`x = 18` -> 29 tiles required -> `x = 18`. 

```
0   2: load_ifm0 load_w0   2: load_ifm1 load_w1      
1   execution_t0 (ofm0_p)  
2                          execution_t1 (ofm1_p)       2: load_ifm0 load_w0           
3                          ofm0 = sum(ofm0_p, ofm1_p)  execution_t2 (ofm2_p)          2: load_ifm1 load_w1
4                                                      ofm1_p = sum(ofm2_p, ofm0)     execution_t3 (ofm0_p)       2: load_ifm0 load_w1
5                                                                                     ofm0 = sum(ofm0_p, ofm1_p)  execution_t4 (ofm2_p)
6                                                                                                                 ofm1_p = sum(ofm2_p, ofm0)
7                                                                                                                 store_ofm1_p   
...
```
`n_ticks = num_tiles + 1 + 2 ` -> to process required `32` ticks.

This buffer layout can be used with tiling along second biggest dimnsion - weight w -> split `64` to `32 32`.
```
  x*26*2 (ifm0) + x*32*2 (w) + 26*32*4 (3 ofm) = 10'000 (buffer size)
```
`x = 57` -> 9 tiles required -> `x = 57`

So total amount of executions is `9*2 = 18`. So estimation for number of ticks is `21`. It's a little better than native approach.

So resulting tiling will have:

```
0        3: load_ifm0 load_w0         
1           2: load_ifm0 load_w0         
2-22        execution_t0          
23          store_ofm0           2: load_w1
24-44                            execution_t1
45                               store_ofm1        3: load_ifm1 
46-90    store_ofm0                                execution_t1   3: load_ifm0
91-135                                             store_ofm1     execution_t2   3: load_ifm1
136-180                                                           store_ofm0     execution_t3  3: load_ifm1
181-225                                                                          store_ofm1    execution_t4
226                                                                                            store_ofm0
```

So total tiled layer took about `226 ticks`. 

#### Tiling that uses smallest buffer 

TBD - Calculations here are not very accurate, need to check than, also ways of searching solution is not so simple. As in this case we have very big search space. 

Let's try to tile same sized matmul without several stages but just to fit into smallest buffer. Buffer size is `10'000`

```
input                            weight 
               512                             128
    +-----------------------+        +----------------------+
    |                       |        |                      |
    |                       |        |                      |
128 |                       |    512 |                      |
    |                       |        |                      |
    |                       |        |                      |
    +-----------------------+        +----------------------+

output
               128
    +----------------------+
    |                      |
    |                      |
128 |                      |
    |                      |
    |                      |
    +----------------------+
```

```
  x*512 (ifm0) + x*512 (w0) + x*x = 10'000 (buffer size)
```
from this equation: `x = 9`, which requires: `15` tiles for ifm and weights -> so tile sizes is `9` . This will produce `15*15 = 225` total amount of tiles.
```
0 |      1: load_ifm0 load_w0         
1 *      execution_t0
2        store_ofm0             1: load_ifm0 load_w0   
3                               execution_t1                
4                               store_ofm0             1: load_ifm0 load_w0 
5                                                      execution_t2
6                                                      store_ofm1_p           1: load_ifm0 load_w0                        
7                                                                             execution_t3 (ofm0)     
8                                                                             store_ofm0_p            1: load_ifm0 load_w0
9                                                                                                     execution_t4 (ofm1_p)
10                                                                                                    store_ofm0
```
`n_ticks = num_tiles * 2 + 1` -> `num_ticks = 450`, we will also need load to level 3, level 2, so it will be around `450 + 2 + 2 = 454`.

```
 +------+-------------+------+-------------+------+------+
 | ifm0 |     w0      | ifm1 |     w1      | ofm0 | ofm1 |
 +------+-------------+------+-------------+------+------+
```
```
  x*512*2 (ifm0) + x*512*2 (w0) + x*x*2 = 10'000 (buffer size)
```
`x = 4` -> 32 tiles in ifm and 32 tiles in weights -> total amount of tiles is `32 * 32 = 1024` -> so total amount of ticks is about `1024 + 2 + 2 = 1028` 


```
 +------+-------------+------+-------------+--------+--------+------+--------+
 | ifm0 |     w0      | ifm1 |     w1      | ofm0_p | ofm1_p | ofm0 | ofm2_p |
 +------+-------------+------+-------------+--------+--------+------+--------+
```

```
  32*y*2 (ifm0) + 32*y*2 (w0) + 32*32*4 = 10'000 (buffer size)
```
`y = 46` -> 12 tiles along reduction axis -> `y = 43`

```
0   2: load_ifm0 load_w0   2: load_ifm1 load_w1      
1   execution_t0 (ofm0_p)  
2                          execution_t1 (ofm1_p)       2: load_ifm0 load_w0           
3                          ofm0 = sum(ofm0_p, ofm1_p)  execution_t2 (ofm2_p)          2: load_ifm1 load_w1
4                                                      ofm1_p = sum(ofm2_p, ofm0)     execution_t3 (ofm0_p)       2: load_ifm0 load_w1
5                                                                                     ofm0 = sum(ofm0_p, ofm1_p)  execution_t4 (ofm2_p)
6                                                                                                                 ofm1_p = sum(ofm2_p, ofm0)
7                                                                                                                 store_ofm1_p   
...
```
In this configuration 4 tiles along input h, 12 tiles along reduction axis, 4 tiles along weight w -> total amount `4 * 4 * 12 = 192` tiles total.
number of ticks will be about `192 + 2 + 2 = 196`

## Threads/Cores

To utilize threads effectively we will need an amount of them and mark memory as shared/private for specific core. 


