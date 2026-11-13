# Trellis

 - rate: 1/2
 - constraint: 3
 - Polynomials
   - 1, 0, 1
   - 1, 1, 1

## Table
| S0,1 | In | C0,1 | S'0,1 |
| ---- | -- | ---- | ----- |
| 0 0  | 0  | 0 0  | 0 0   |
|      | 1  | 1 1  | 1 0   |
| 0 1  | 0  | 1 1  | 0 0   |
|      | 1  | 0 0  | 1 0   |
| 1 0  | 0  | 0 1  | 0 1   |
|      | 1  | 1 0  | 1 1   |
| 1 1  | 0  | 1 0  | 0 1   |
|      | 1  | 0 1  | 1 1   |

```mermaid
graph TD
subgraph trellis[Trellis Diagram]
    direction LR;
    subgraph S0[0X]
        subgraph 0[00]
            I00[00]
            O00[00]
        end
        subgraph 1[01]
            I01[01]
            O01[01]
        end
    end
    subgraph S1[1X]
        subgraph 2[10]
            I10[10]
            O10[10]
        end
        subgraph 3[11]
            I11[11]
            O11[11]
        end
    end
end
I00--0/00-->O00
I00--1/11-->O10
I01--0/11-->O00
I01--1/00-->O10

I10--0/01-->O01
I10--1/10-->O11
I11--0/10-->O01
I11--1/01-->O11
```
