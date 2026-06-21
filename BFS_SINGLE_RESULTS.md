# BFS Single-Layer (push/pull, no inner loop) Benchmark

| Dataset | GPU | ms/iter | maxVal | unvisited | iters |
|---|---:|---:|---:|---:|---:|
| soc-twitter | 2 | 42.6 | 17 | 0 | 18 |
| soc-twitter | 4 | 52.6 | 17 | 19230 | 18 |
| soc-twitter | 8 | 82.5 | 17 | 108778 | 18 |
| soc-sinaweibo | 2 | 121.2 | 6 | 39 | 7 |
| soc-sinaweibo | 4 | 96.5 | 9 | 32735070 | 10 |
| soc-sinaweibo | 8 | 98.6 | 17 | 46409686 | 18 |
| sk-2005 | 2 | 98.1 | 23 | 21017 | 24 |
| sk-2005 | 4 | 90.3 | 23 | 21017 | 24 |
| sk-2005 | 8 | 106.0 | 23 | 6115 | 24 |
| uk-2007 | 2 | FAIL | - | - | - |
| uk-2007 | 4 | 295.2 | 17 | 1053840 | 19 |
| uk-2007 | 8 | 134.1 | 17 | 1053840 | 19 |
