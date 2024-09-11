使用 peft 框架 lora 方法微调 Bert 完成 NER 任务

```log
defaultdict(<class 'list'>, {'PERSON': ['宗元第'], 'TIME': ['前尚']})
defaultdict(<class 'list'>, {'PERSON': ['宗元第'], 'TIME': ['前尚']})
=+++++++++
=+++++++++
defaultdict(<class 'list'>, {'LOCATION': ['国常', '洲国', '洲国'], 'ORGANIZATION': ['合国代', '合国安理会发'], 'PERSON': ['国放2'], 'TIME': ['4日在']})
defaultdict(<class 'list'>, {'LOCATION': ['国常', '洲国', '洲国'], 'ORGANIZATION': ['合国代', '合国安理会发'], 'PERSON': ['国放2'], 'TIME': ['4日在']})
=+++++++++
2024-09-08 16:57:33,691 - __main__ - INFO - PERSON类实体，准确率：0.803109, 召回率: 0.803109, F1: 0.803104
2024-09-08 16:57:33,691 - __main__ - INFO - LOCATION类实体，准确率：0.807018, 召回率: 0.769874, F1: 0.788004
2024-09-08 16:57:33,691 - __main__ - INFO - TIME类实体，准确率：0.873563, 召回率: 0.853933, F1: 0.863631
2024-09-08 16:57:33,691 - __main__ - INFO - ORGANIZATION类实体，准确率：0.614035, 召回率: 0.736842, F1: 0.669851
2024-09-08 16:57:33,691 - __main__ - INFO - Macro-F1: 0.781148
2024-09-08 16:57:33,691 - __main__ - INFO - Micro-F1 0.793489
2024-09-08 16:57:33,691 - __main__ - INFO - --------------------


```