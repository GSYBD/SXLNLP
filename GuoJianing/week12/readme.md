|               | 位置编码 | transformer结构 | 多头机制    | ff层设计 | 归一化层选择            | 激活函数 | 是否使用bias                |
| ------------- | -------- | --------------- | ----------- | -------- | ----------------------- | -------- | --------------------------- |
| baichuan2-7b  | RoPE     | 串行            | 传统方式    | gated    | RMSnorm/pre norm        | SiLU     | 无bias                      |
| baichuan2-13b | Alibi    | 串行            | 传统方式    | gated    | RMSnorm/pre norm        | SiLU     | 无bias                      |
| chatglm2      | RoPE     | 串行            | multi query | gated    | RMSnorm/pre norm        | SiLU     | qkv有bias，其他线性层无bias |
| chatglm3      | RoPE     | 串行            | multi query | gated    | RMSnorm/pre norm        | SiLU     | qkv有bias，其他线性层无bias |
| llama2        | RoPE     | 串行            | multi query | gated    | RMSnorm/pre norm        | SiLU     | 无bias                      |
| moss          | RoPE     | 并行            | 传统方式    | 传统方式 | RMSnorm/pre norm        | GELU     | ff有bias                    |
| qwen          | RoPE     | 串行            | 传统方式    | gated    | RMSnorm/pre norm        | SiLU     | qkv有bias，其他线性层无bias |
| mixtral       | RoPE     | 串行            | multi query | MOE      | RMSnorm/pre norm        | RELU     | 无bias                      |
| grok1         | RoPE     | 串行            | multi query | MOE      | RMSnorm/sandwich norm   | RELU     | 无bias                      |
| gemma         | RoPE     | 串行            | multi query | gated    | RMSnorm/sandwich norm   | GELU     | 无bias                      |
| dbrx          | RoPE     | 串行            | multi query | MOE      | LayerNorm/sandwich norm | GELU     | 无bias                      |
| grok1         | RoPE     | 串行            | group query | MOE      | RMSnorm/sandwich norm   | GELU     | 无bias                      |
